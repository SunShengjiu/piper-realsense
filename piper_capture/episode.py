"""数据集目录、episode 生命周期与样本写入。

目录结构（对齐需求第九节）：
    dataset/
      manifest.json
      calibrations/{camera,handeye,gripper}/
      scenes/<scene_id>/{scene.json,external/environment_overview.mp4}
      episodes/<episode_id>/
        metadata.json samples.jsonl robot_states.jsonl commands.jsonl
        rgb/ depth_raw/ depth_aligned/ external/side_task.mp4 logs/ quality_report.json

中断安全：
  - JSON 一律原子写（临时文件 + os.replace）。
  - JSONL 每行写后 flush，中断后已写行仍可解析。
  - metadata.json 在开始时写入 status="open"，正常结束写 "closed"，
    Ctrl-C 写 "aborted"；即使进程被 kill，status 仍是 "open" 且数据可读。
  - 所有路径都是相对 dataset root 的 POSIX 路径，整个数据集可整体搬移。
"""
from __future__ import annotations

import getpass
import json
import platform
import re
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .jsonio import JsonlWriter, relpath, sha256_json, write_json
from .schema import JOINT_NAMES, SCHEMA_VERSION, UNIT_DECLARATIONS


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def validate_id(value: str, field: str) -> str:
    if not _ID_RE.match(value) or ".." in value or "/" in value:
        raise ValueError(f"{field} 只允许字母数字和 _ . -，收到: {value!r}")
    return value


class CalibrationStore:
    """固定标定参数的独立文件存储。样本只引用 calibration_id。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, kind: str, calibration_id: str) -> Path:
        if kind not in ("camera", "handeye", "gripper"):
            raise ValueError(f"未知标定类型: {kind}")
        return self.root / "calibrations" / kind / f"{validate_id(calibration_id, 'calibration_id')}.json"

    def save(self, kind: str, calibration_id: str, payload: Dict[str, Any]) -> Path:
        payload = dict(payload)
        payload.setdefault("schema_version", SCHEMA_VERSION)
        payload.setdefault("calibration_id", calibration_id)
        payload.setdefault("kind", kind)
        payload.setdefault("saved_at", _utc_now())
        payload["content_sha256"] = sha256_json({k: v for k, v in payload.items() if k != "content_sha256"})
        return write_json(self.path(kind, calibration_id), payload)

    def load(self, kind: str, calibration_id: str) -> Dict[str, Any]:
        p = self.path(kind, calibration_id)
        if not p.is_file():
            raise FileNotFoundError(f"标定文件不存在: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def exists(self, kind: str, calibration_id: str) -> bool:
        return self.path(kind, calibration_id).is_file()

    def index(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for kind in ("camera", "handeye", "gripper"):
            d = self.root / "calibrations" / kind
            out[kind] = sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []
        return out


class DatasetManifest:
    """dataset/manifest.json 的读写。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / "manifest.json"

    def load(self) -> Dict[str, Any]:
        if self.path.is_file():
            with open(self.path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return self._empty()

    def _empty(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": f"piper-d435i-{_utc_stamp()}",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "joint_names": list(JOINT_NAMES),
            "units": UNIT_DECLARATIONS,
            "ee_pose_layout": "[x, y, z, qw, qx, qy, qz]",
            "path_convention": "所有文件路径相对本 manifest.json 所在目录，数据集可整体搬移",
            "scenes": [],
            "episodes": [],
            "calibrations": {},
        }

    def update(self, **mutator: Any) -> Dict[str, Any]:
        data = self.load()
        if "episodes" in mutator:
            known = {e["episode_id"]: e for e in data.get("episodes", [])}
            known.update({e["episode_id"]: e for e in mutator.pop("episodes")})
            data["episodes"] = sorted(known.values(), key=lambda e: e["episode_id"])
        if "scenes" in mutator:
            known = {s["scene_id"]: s for s in data.get("scenes", [])}
            known.update({s["scene_id"]: s for s in mutator.pop("scenes")})
            data["scenes"] = sorted(known.values(), key=lambda s: s["scene_id"])
        data.update(mutator)
        data["updated_at"] = _utc_now()
        store = CalibrationStore(self.root)
        data["calibrations"] = store.index()
        write_json(self.path, data)
        return data


def scene_paths(root: Path, scene_id: str) -> Dict[str, Path]:
    base = Path(root) / "scenes" / validate_id(scene_id, "scene_id")
    return {"base": base, "scene_json": base / "scene.json", "external": base / "external"}


def ensure_scene(
    root: Path,
    scene_id: str,
    *,
    camera_calibration_id: Optional[str] = None,
    handeye_calibration_id: Optional[str] = None,
    gripper_calibration_id: Optional[str] = None,
    mount: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    paths = scene_paths(root, scene_id)
    paths["external"].mkdir(parents=True, exist_ok=True)
    existing: Dict[str, Any] = {}
    if paths["scene_json"].is_file():
        with open(paths["scene_json"], "r", encoding="utf-8") as fh:
            existing = json.load(fh)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "created_at": existing.get("created_at", _utc_now()),
        "updated_at": _utc_now(),
        "camera_calibration_id": camera_calibration_id or existing.get("camera_calibration_id"),
        "handeye_calibration_id": handeye_calibration_id or existing.get("handeye_calibration_id"),
        "gripper_calibration_id": gripper_calibration_id or existing.get("gripper_calibration_id"),
        "camera_mount": mount if mount is not None else existing.get("camera_mount"),
        "environment_overview": existing.get("environment_overview") or [],
        "robots": existing.get("robots", ["piper"]),
    }
    if extra:
        payload.update(extra)
    write_json(paths["scene_json"], payload)
    return paths["scene_json"]


class EpisodeWriter:
    """单个 episode 的写入器。"""

    def __init__(
        self,
        root: Path,
        scene_id: str,
        episode_id: str,
        *,
        robot_meta: Optional[Dict[str, Any]] = None,
        camera_meta: Optional[Dict[str, Any]] = None,
        sync_config: Optional[Dict[str, Any]] = None,
        capture_config: Optional[Dict[str, Any]] = None,
        calibration_ids: Optional[Dict[str, Optional[str]]] = None,
        notes: Optional[str] = None,
        fsync_every: int = 0,
    ) -> None:
        self.root = Path(root).resolve()
        self.scene_id = validate_id(scene_id, "scene_id")
        self.episode_id = validate_id(episode_id, "episode_id")
        self.dir = self.root / "episodes" / self.episode_id
        for sub in ("rgb", "depth_raw", "depth_aligned", "external", "logs"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.calibrations = CalibrationStore(self.root)
        self.calibration_ids = calibration_ids or {}
        self.started_at = _utc_now()
        self.started_monotonic = time.monotonic()
        self._sample_writer = JsonlWriter(self.dir / "samples.jsonl", fsync_every=fsync_every)
        self._state_writer = JsonlWriter(self.dir / "robot_states.jsonl", fsync_every=fsync_every)
        self._log_fh = open(self.dir / "logs" / "session.log", "a", encoding="utf-8")
        self.sample_count = 0
        self.state_count = 0
        self.invalid_sample_count = 0
        self.first_sample_host_ns: Optional[int] = None
        self.last_sample_host_ns: Optional[int] = None
        self.status = "open"
        self._closed = False

        self.metadata: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "scene_id": self.scene_id,
            "status": "open",
            "started_at": self.started_at,
            "ended_at": None,
            "host": {
                "hostname": socket.gethostname(),
                "user": getpass.getuser(),
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            "joint_names": list(JOINT_NAMES),
            "units": UNIT_DECLARATIONS,
            "ee_pose_layout": "[x, y, z, qw, qx, qy, qz]",
            "robot": robot_meta or {},
            "camera": camera_meta or {},
            "sync": sync_config or {},
            "capture": capture_config or {},
            "calibration_ids": self.calibration_ids,
            "files": {
                "samples": "samples.jsonl",
                "robot_states": "robot_states.jsonl",
                "commands": "commands.jsonl",
                "logs": "logs/",
                "rgb": "rgb/",
                "depth_raw": "depth_raw/",
                "depth_aligned": "depth_aligned/",
                "external": "external/",
                "quality_report": "quality_report.json",
            },
            "notes": notes,
            "counters": {},
        }
        write_json(self.dir / "metadata.json", self.metadata)
        self._write_command_index()
        self.log(f"episode {self.episode_id} 开始 (scene={self.scene_id})")

    # ------------------------------------------------------------------ 日志
    def log(self, message: str) -> None:
        line = f"[{_utc_now()}] {message}\n"
        self._log_fh.write(line)
        self._log_fh.flush()

    def _write_command_index(self) -> None:
        """commands.jsonl：明确说明本项目未接入下发命令记录，不从状态反推命令。"""
        writer = JsonlWriter(self.dir / "commands.jsonl", fsync_every=0)
        writer.write(
            {
                "schema_version": SCHEMA_VERSION,
                "episode_id": self.episode_id,
                "record_type": "command_log_unavailable",
                "command_recording_enabled": False,
                "reason": "当前采集程序只读取状态，未接入任何下发命令通道的记录",
                "policy": "禁止从反馈状态反推并冒充真实下发命令",
                "written_at": _utc_now(),
            }
        )
        writer.close()

    # ------------------------------------------------------------------ 写入
    def write_robot_state(self, record: Dict[str, Any]) -> None:
        rec = dict(record)
        rec["schema_version"] = SCHEMA_VERSION
        rec["episode_id"] = self.episode_id
        self._state_writer.write(rec)
        self.state_count += 1

    def write_sample(self, record: Dict[str, Any]) -> None:
        rec = dict(record)
        rec["schema_version"] = SCHEMA_VERSION
        rec["episode_id"] = self.episode_id
        if not rec["valid"]:
            self.invalid_sample_count += 1
        # 记录样本主机接收时间跨度：实测样本率应以该跨度为准，
        # 而不是含开关相机/预热耗时的整段 episode 墙钟时长。
        ts = rec.get("timestamps") or {}
        host_ns = ts.get("reference_host_ns")
        if host_ns is not None:
            if self.first_sample_host_ns is None:
                self.first_sample_host_ns = int(host_ns)
            self.last_sample_host_ns = int(host_ns)
        self._sample_writer.write(rec)
        self.sample_count += 1

    def rel(self, path: Path | str) -> str:
        return relpath(path, self.root)

    # ------------------------------------------------------------------ 收尾
    def finalize(self, status: str = "closed", extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._closed:
            return self.metadata
        self.status = status
        self._sample_writer.close()
        self._state_writer.close()
        span_s = None
        measured_rate = None
        if self.first_sample_host_ns is not None and self.last_sample_host_ns is not None:
            span_s = (self.last_sample_host_ns - self.first_sample_host_ns) / 1e9
            if span_s > 0 and self.sample_count > 1:
                measured_rate = (self.sample_count - 1) / span_s
        self.metadata.update(
            {
                "status": status,
                "ended_at": _utc_now(),
                "duration_s": time.monotonic() - self.started_monotonic,
                "sample_span_s": span_s,
                # 实测样本率以首个到最后一个样本的主机接收时间跨度为准，
                # duration_s 含开关相机/预热，不能直接当作采样时长
                "measured_sample_rate_hz": measured_rate,
                "counters": {
                    "samples": self.sample_count,
                    "invalid_samples": self.invalid_sample_count,
                    "robot_states": self.state_count,
                },
            }
        )
        if extra:
            self.metadata.update(extra)
        write_json(self.dir / "metadata.json", self.metadata)
        DatasetManifest(self.root).update(
            episodes=[
                {
                    "episode_id": self.episode_id,
                    "scene_id": self.scene_id,
                    "status": status,
                    "started_at": self.started_at,
                    "ended_at": self.metadata["ended_at"],
                    "samples": self.sample_count,
                    "invalid_samples": self.invalid_sample_count,
                    "robot_states": self.state_count,
                }
            ]
        )
        self.log(f"episode {self.episode_id} 结束 status={status} samples={self.sample_count}")
        try:
            self._log_fh.close()
        except Exception:
            pass
        self._closed = True
        return self.metadata


def new_episode_id(prefix: str = "ep") -> str:
    return f"{prefix}-{_utc_stamp()}-{uuid.uuid4().hex[:6]}"


def build_sample_record(
    *,
    sample_id: str,
    episode_id: str,
    seq: int,
    frame: Any,
    matched: Any,
    camera_model: Any,
    paths: Dict[str, Optional[str]],
    relative_to: Path,
    handeye_calibration_id: Optional[str],
    clock_mappers: Dict[str, Any],
    camera_frame: str = "camera_color_optical_frame",
    sync_tolerance_ms: float = 33.0,
) -> Dict[str, Any]:
    """组装一条 samples.jsonl 记录。"""
    color_ts = frame.color_device_ts_ms
    depth_ts = frame.depth_device_ts_ms
    mapper_color = clock_mappers.get("color")
    mapper_depth = clock_mappers.get("depth")
    color_host_est = mapper_color.to_host_ns(color_ts) if mapper_color else None
    depth_host_est = mapper_depth.to_host_ns(depth_ts) if mapper_depth else None
    rgb_dt_ms = (
        (color_host_est - frame.frameset_host_recv_ns) / 1e6 if color_host_est is not None else None
    )
    depth_dt_ms = (
        (depth_host_est - frame.frameset_host_recv_ns) / 1e6 if depth_host_est is not None else None
    )

    depth_ok = frame.depth_raw_u16 is not None and frame.depth_raw_u16.size > 0
    aligned_ok = frame.depth_aligned_u16 is not None and frame.depth_aligned_u16.size > 0
    aligned_matches_color = (
        aligned_ok
        and frame.depth_aligned_u16.shape[0] == frame.color_bgr.shape[0]
        and frame.depth_aligned_u16.shape[1] == frame.color_bgr.shape[1]
    )
    rgb_ok = paths.get("rgb") is not None

    reasons: List[str] = []
    if not rgb_ok:
        reasons.append("RGB 未保存")
    if not depth_ok:
        reasons.append("原始深度缺失")
    if not aligned_ok:
        reasons.append("对齐深度缺失")
    elif not aligned_matches_color:
        reasons.append("对齐深度尺寸与 RGB 不一致")
    if matched.joints_rad is None:
        reasons.append("缺少机械臂关节状态")
    if matched.ee_pose is None:
        reasons.append("缺少 EE 位姿")
    if not matched.valid:
        reasons.append(matched.invalid_reason or "机器人状态匹配无效")
    if matched.stale:
        reasons.append("使用的关节反馈已过期")
    if frame.pair_dt_ms and abs(frame.pair_dt_ms) > sync_tolerance_ms:
        reasons.append(f"RGB/Depth 设备时间差 {frame.pair_dt_ms:.2f} ms 超过容差 {sync_tolerance_ms} ms")
    if rgb_dt_ms is not None and abs(rgb_dt_ms) > sync_tolerance_ms:
        reasons.append(f"RGB 设备时间与主机接收时间不一致 {rgb_dt_ms:.2f} ms")

    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "sample_id": sample_id,
        "seq": seq,
        "valid": len(reasons) == 0,
        "invalid_reasons": reasons,
        "timestamps": {
            "rgb": {
                "source_timestamp_ms": color_ts,
                "source_clock_domain": frame.color_ts_domain,
                "host_recv_ns": frame.color_host_recv_ns,
                "device_clock_mapped_host_ns": color_host_est,
                "device_vs_host_dt_ms": rgb_dt_ms,
                "clock_source": "device:realsense_frame_timestamp",
            },
            "depth": {
                "source_timestamp_ms": depth_ts,
                "source_clock_domain": frame.depth_ts_domain,
                "host_recv_ns": frame.depth_host_recv_ns,
                "device_clock_mapped_host_ns": depth_host_est,
                "device_vs_host_dt_ms": depth_dt_ms,
                "clock_source": "device:realsense_frame_timestamp",
            },
            "aligned_depth": {
                "source_timestamp_ms": depth_ts,
                "source_clock_domain": frame.depth_ts_domain,
                "note": "对齐深度继承原始深度时间戳，由同一帧集生成",
            },
            "robot_joints": {
                "source_timestamp_ns": None,
                "source_clock_domain": "unavailable",
                "host_recv_ns": None,
                "clock_source": "host_receive_only:piper_can_protocol_has_no_device_timestamp",
            },
            "reference_host_ns": frame.frameset_host_recv_ns,
            "reference_clock": "host_receive_time_frameset(time.time_ns)",
        },
        "sync": {
            "matching_method": matched.method,
            "rgb_depth_device_dt_ms": frame.pair_dt_ms,
            "rgb_depth_tolerance_ms": sync_tolerance_ms,
            "robot_joints_dt_ms": matched.dt_ms,
            "gripper_dt_ms": matched.gripper_dt_ms,
            "is_hardware_synchronized": False,
            "note": "全部为软件时间匹配",
        },
    }
    record.update(matched.to_dict())
    # 关节反馈的主机接收时间来自实际使用的状态
    if matched.gripper_source_state_id is not None:
        record["timestamps"]["robot_joints"]["host_recv_ns"] = int(
            frame.frameset_host_recv_ns + round((matched.dt_ms or 0.0) * 1e6)
        )

    if matched.joints_rad is not None and len(matched.joints_rad) != 6:
        record["valid"] = False
        record.setdefault("invalid_reasons", []).append(f"关节数量不是 6，实际 {len(matched.joints_rad)}")

    if matched.ee_pose is not None:
        record["ee_pose"] = matched.ee_pose
        record["ee_pose_frame"] = "link6"
        record["ee_pose_source"] = "feedback_joint_fk"
        record["ee_pose_units"] = {"position": "m", "quaternion": "wxyz"}

    record["cameras"] = {
        "camera_frame": camera_frame,
        "depth_frame": "camera_depth_optical_frame",
        "aligned_depth_frame": "camera_color_optical_frame",
        # 图像路径的基准和其他字段不同：这里是 episode 目录（rgb/000001.png），
        # metadata 里的标定/场景路径才是相对 dataset root。显式写出来避免读错。
        "path_base": "episode_dir",
        "path_base_note": "cameras.*.path 相对本 episode 目录；metadata 内的 calibration/scene 路径相对 dataset root",
        "color": {
            "path": paths.get("rgb"),
            "frame_number": frame.color_frame_number,
            "width": int(frame.color_bgr.shape[1]),
            "height": int(frame.color_bgr.shape[0]),
            "encoding": "bgr8",
            "format": "png (lossless)",
        },
        "depth_raw": {
            "path": paths.get("depth_raw"),
            "frame_number": frame.depth_frame_number,
            "width": int(frame.depth_raw_u16.shape[1]),
            "height": int(frame.depth_raw_u16.shape[0]),
            "dtype": "uint16",
            "format": "png (lossless)",
            "depth_scale_m": camera_model.depth_scale_m if camera_model else None,
            "depth_formula": "depth_m = raw_depth * depth_scale_m",
            "invalid_value": 0,
            "invalid_pixels": frame.invalid_depth_pixels,
        },
        "depth_aligned": {
            "path": paths.get("depth_aligned"),
            "frame_number": frame.aligned_frame_number,
            "width": int(frame.depth_aligned_u16.shape[1]) if aligned_ok else None,
            "height": int(frame.depth_aligned_u16.shape[0]) if aligned_ok else None,
            "dtype": "uint16",
            "format": "png (lossless)",
            "alignment": "realsense_rs_align_to_color (geometry based)",
            "depth_scale_m": camera_model.depth_scale_m if camera_model else None,
            "invalid_value": 0,
            "invalid_pixels": frame.aligned_invalid_pixels,
            "same_size_as_rgb": aligned_matches_color,
        },
        "is_geometric_alignment": True,
    }
    record["calibrations"] = {
        "camera_calibration_id": camera_model.calibration_id if camera_model else None,
        "handeye_calibration_id": handeye_calibration_id,
        "gripper_calibration_id": matched.gripper_calibration_id,
    }
    return record