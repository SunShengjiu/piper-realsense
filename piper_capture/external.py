"""第三人称视频的导入与登记。

本项目**不**开发第三人称相机的驱动、录制或运动控制。这里只有登记接口：
  - 计算 SHA-256 校验值；
  - 探测分辨率/帧率/时长（cv2，必要时用 ffmpeg 补充编码信息）；
  - 按角色归位：
      side_task           -> episodes/<episode_id>/external/
      environment_overview-> scenes/<scene_id>/external/
  - 记录录制时间、时间偏移与同步状态；没有同步依据时一律 `unsynchronized`，
    绝不伪造时间对齐。
  - 默认把文件复制进数据集，使数据集整体搬移后仍可读取；`copy=False`
    只登记绝对路径（此时数据集不可搬移，会在记录里写明）。

未导入第三人称视频时，机械臂与 D435i 采集完全不受影响。
"""
from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .episode import DatasetManifest, scene_paths, validate_id
from .jsonio import read_json, sha256_file, write_json
from .schema import SCHEMA_VERSION

ROLES = ("side_task", "environment_overview")

SYNC_STATES = (
    "unsynchronized",       # 没有任何同步依据（默认）
    "offset_provided",      # 用户提供了录制时间/偏移，但未与采集时钟核验
    "synchronized",         # 有可核查的同步依据（需要外部证据，本项目不声称）
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def probe_video(path: Path, use_ffmpeg: bool = True) -> Dict[str, Any]:
    """探测视频基本信息。主用 cv2（无外部依赖），ffmpeg 只作补充。"""
    import cv2

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {path}")
    out: Dict[str, Any] = {
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        "probe_tool": ["cv2.VideoCapture"],
    }
    cap = cv2.VideoCapture(str(path))
    try:
        if cap.isOpened():
            out["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            out["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            out["fps"] = fps if fps > 0 else None
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out["frame_count"] = frames if frames > 0 else None
            if out.get("fps") and out.get("frame_count"):
                out["duration_s"] = out["frame_count"] / out["fps"]
            else:
                out["duration_s"] = None
            out["fourcc"] = _fourcc(cap.get(cv2.CAP_PROP_FOURCC))
        else:
            out["cv2_error"] = "VideoCapture 无法打开文件（可能缺少对应解码器）"
        out["probe_tool"].append("cv2.CAP_PROP_FOURCC")
    finally:
        cap.release()

    if use_ffmpeg:
        extra = _probe_ffmpeg(path)
        if extra:
            out["probe_tool"].append("ffmpeg -i")
            for k, v in extra.items():
                out.setdefault(k, v)
            out["ffmpeg"] = extra
    return out


def _fourcc(code: float) -> Optional[str]:
    try:
        n = int(code)
    except Exception:
        return None
    if n <= 0:
        return None
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))


def _probe_ffmpeg(path: Path) -> Optional[Dict[str, Any]]:
    import re
    import subprocess

    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        p = subprocess.run(
            [exe, "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return None
    text = p.stderr or ""
    out: Dict[str, Any] = {}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", text)
    if m:
        out["ffmpeg_duration_s"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"Stream #\d+:\d+.*?: Video: ([^,]+)", text)
    if m:
        out["codec"] = m.group(1).strip()
    m = re.search(r"(\d{2,5})x(\d{2,5})", text)
    if m:
        out["ffmpeg_resolution"] = f"{m.group(1)}x{m.group(2)}"
    m = re.search(r"(\d+\.?\d*)\s*fps", text)
    if m:
        out["ffmpeg_fps"] = float(m.group(1))
    return out or None


def _resolve_target(root: Path, role: str, file_name: str, scene_id: Optional[str], episode_id: Optional[str]) -> Path:
    if role == "side_task":
        if not episode_id:
            raise ValueError("role=side_task 必须提供 episode_id（侧视视频按 episode 关联）")
        ep_dir = Path(root) / "episodes" / validate_id(episode_id, "episode_id")
        if not (ep_dir / "metadata.json").is_file():
            raise FileNotFoundError(f"episode 不存在，请先采集或创建: {ep_dir / 'metadata.json'}")
        base = ep_dir / "external"
    elif role == "environment_overview":
        if not scene_id:
            raise ValueError("role=environment_overview 必须提供 scene_id（环视视频按 scene 保存）")
        sp = scene_paths(root, scene_id)
        if not sp["scene_json"].is_file():
            raise FileNotFoundError(f"scene 不存在，请先创建场景: {sp['scene_json']}")
        base = sp["external"]
    else:
        raise ValueError(f"未知角色 {role}，可用: {ROLES}")
    base.mkdir(parents=True, exist_ok=True)
    return base / file_name


def register(
    root: Path,
    video_path: Path,
    role: str,
    *,
    scene_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    copy: bool = True,
    recorded_at: Optional[str] = None,
    time_offset_s: Optional[float] = None,
    offset_basis: Optional[str] = None,
    operator: Optional[str] = None,
    notes: Optional[str] = None,
    use_ffmpeg: bool = True,
) -> Dict[str, Any]:
    """登记/导入一个第三人称视频文件。"""
    root = Path(root)
    video_path = Path(video_path).expanduser().resolve()
    if role not in ROLES:
        raise ValueError(f"未知角色 {role}，可用: {ROLES}")

    info = probe_video(video_path, use_ffmpeg=use_ffmpeg)
    checksum = sha256_file(video_path)
    target = _resolve_target(root, role, video_path.name, scene_id, episode_id)

    if copy:
        if target.resolve() != video_path:
            shutil.copy2(video_path, target)
        stored_path = target.resolve()
        path_mode = "copied_into_dataset"
    else:
        stored_path = video_path
        path_mode = "referenced_in_place"

    sync_status = "offset_provided" if time_offset_s is not None else "unsynchronized"
    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "external_video",
        "role": role,
        "registered_at": _utc_now(),
        "stored_path": str(stored_path),
        "stored_path_relative_to_dataset": (
            stored_path.relative_to(root).as_posix() if _is_relative_to(stored_path, root) else None
        ),
        "path_mode": path_mode,
        "source_path": str(video_path),
        "file_name": video_path.name,
        "sha256": checksum,
        "scene_id": scene_id,
        "episode_id": episode_id,
        "video": info,
        "recording": {
            "recorded_at": recorded_at,
            "duration_s": info.get("duration_s") or info.get("ffmpeg_duration_s"),
            "resolution": (
                f"{info.get('width')}x{info.get('height')}"
                if info.get("width") and info.get("height")
                else info.get("ffmpeg_resolution")
            ),
            "fps": info.get("fps") or info.get("ffmpeg_fps"),
        },
        "synchronization": {
            "status": sync_status,
            "time_offset_s": time_offset_s,
            "offset_basis": offset_basis,
            "clock_domain": "third_person_camera_unknown",
            "policy": "没有同步依据时标记 unsynchronized，不伪造时间对齐",
            "requirements_alignment": (
                "侧视视频按 episode 关联；环境环视视频按 scene 保存，可被多个 episode 引用；"
                "不要求与每条机械臂样本一一对应"
            ),
        },
        "operator": operator,
        "notes": notes,
        "file_checksum_verified_at_registration": True,
        "dataset_movable": bool(copy),
    }
    if not copy:
        record["dataset_movable_warning"] = (
            "以引用方式登记，文件不在数据集内；数据集整体搬移到其他机器后该路径可能失效"
        )

    _attach_record(root, record)
    registry = _registry(root)
    registry["videos"] = [v for v in registry.get("videos", []) if v.get("sha256") != checksum] + [record]
    registry["updated_at"] = _utc_now()
    write_json(root / "external_registry.json", registry)
    DatasetManifest(root).update()
    return record


def _attach_record(root: Path, record: Dict[str, Any]) -> None:
    role = record["role"]
    if role == "environment_overview":
        p = scene_paths(root, record["scene_id"])["scene_json"]
        if not p.is_file():
            raise FileNotFoundError(f"scene 不存在，请先创建场景: {p}")
        scene = read_json(p)
        # ensure_scene 会写入 environment_overview: null（尚无视频时），
        # setdefault 对已存在的 None 不生效，必须显式兜底，否则下面迭代 None 会报错
        existing = scene.get("environment_overview") or []
        scene["environment_overview"] = [
            v for v in existing if v.get("sha256") != record["sha256"]
        ] + [_brief(record)]
        scene["updated_at"] = _utc_now()
        write_json(p, scene)
    else:
        p = Path(root) / "episodes" / validate_id(record["episode_id"], "episode_id") / "metadata.json"
        if not p.is_file():
            raise FileNotFoundError(f"episode 不存在，请先采集或创建: {p}")
        meta = read_json(p)
        existing = meta.get("external_videos") or []
        meta["external_videos"] = [
            v for v in existing if v.get("sha256") != record["sha256"]
        ] + [_brief(record)]
        write_json(p, meta)


def _brief(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": record["role"],
        "stored_path_relative_to_dataset": record["stored_path_relative_to_dataset"],
        "file_name": record["file_name"],
        "sha256": record["sha256"],
        "scene_id": record["scene_id"],
        "episode_id": record["episode_id"],
        "duration_s": record["recording"]["duration_s"],
        "synchronization_status": record["synchronization"]["status"],
        "registered_at": record["registered_at"],
    }


def _is_relative_to(p: Path, base: Path) -> bool:
    try:
        p.relative_to(base)
        return True
    except ValueError:
        return False


def _registry(root: Path) -> Dict[str, Any]:
    p = Path(root) / "external_registry.json"
    if p.is_file():
        try:
            return read_json(p)
        except Exception:
            pass
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "roles": list(ROLES),
        "videos": [],
    }


def list_registered(root: Path, *, role: Optional[str] = None, scene_id: Optional[str] = None, episode_id: Optional[str] = None) -> List[Dict[str, Any]]:
    out = _registry(Path(root)).get("videos", [])
    if role:
        out = [v for v in out if v["role"] == role]
    if scene_id:
        out = [v for v in out if v.get("scene_id") == scene_id]
    if episode_id:
        out = [v for v in out if v.get("episode_id") == episode_id]
    return out


def verify_registered(root: Path, *, check_content: bool = False) -> Dict[str, Any]:
    """核对已登记文件是否仍存在；可选重新计算校验值。"""
    root = Path(root)
    results = []
    for v in list_registered(root):
        p = Path(v["stored_path"])
        entry: Dict[str, Any] = {
            "file_name": v["file_name"],
            "role": v["role"],
            "path": str(p),
            "exists": p.is_file(),
        }
        if p.is_file() and check_content:
            now = sha256_file(p)
            entry["checksum_match"] = now == v["sha256"]
            entry["sha256_now"] = now
        results.append(entry)
    bad = [r for r in results if not r["exists"] or r.get("checksum_match") is False]
    return {
        "schema_version": SCHEMA_VERSION,
        "checked_at": _utc_now(),
        "n": len(results),
        "results": results,
        "ok": not bad,
        "problems": bad,
    }