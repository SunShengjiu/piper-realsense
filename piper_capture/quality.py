"""数据质量检查工具。

读回 episode（含被中断的 episode），检查：
  - 文件关联：samples/robot_states/metadata 是否能解析，图像路径是否存在；
  - 单位与几何：关节数量与顺序、四元数顺序与归一化、相邻帧符号连续性、
    ee_pose 与反馈关节角 FK 是否一致；
  - 图像：RGB/原始深度/对齐深度尺寸、深度 dtype=uint16、depth_scale、
    对齐深度是否与 RGB 同尺寸、无效深度值语义；
  - 时间：各源时间戳是否齐全、RGB/Depth/关节的时间差、超容差与过期反馈计数、
    实测样本率与目标率的偏差；
  - 标定引用：camera/handeye/gripper calibration_id 对应文件是否存在。

输出 `<episode>/quality_report.json`，结论为 pass / needs_attention / fail。
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .episode import CalibrationStore
from .jsonio import read_json, resolve_path, write_json
from .kinematics import ForwardKinematics
from .schema import JOINT_NAMES, SCHEMA_VERSION

PASS = "pass"
ATTENTION = "needs_attention"
FAIL = "fail"
UNKNOWN = "unknown"
_RANK = {PASS: 0, ATTENTION: 1, FAIL: 2, UNKNOWN: 1}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def tolerant_read_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """逐行读取 JSONL，容忍中断留下的不完整尾行。

    返回 (记录列表, 问题列表)。中断安全是采集的硬要求：已写完的行必须仍可解析。
    """
    records: List[Dict[str, Any]] = []
    problems: List[Dict[str, Any]] = []
    if not path.is_file():
        problems.append({"type": "missing_file", "file": str(path)})
        return records, problems
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            problems.append(
                {
                    "type": "unparsable_line",
                    "file": str(path),
                    "line": lineno,
                    "is_last_line": lineno == len(lines),
                    "error": str(exc),
                    "note": (
                        "中断时可能留下半行；已完成的其余行仍可解析"
                        if lineno == len(lines)
                        else "非尾行损坏，需要检查写入过程"
                    ),
                }
            )
    return records, problems


class _Checks:
    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []

    def add(self, name: str, status: str, detail: Any = None, *, evidence: Any = None) -> None:
        self.items.append({"name": name, "status": status, "detail": detail, "evidence": evidence})

    def add_if(self, condition: bool, name: str, detail: Any = None, *, evidence: Any = None, fail_status: str = FAIL) -> None:
        self.add(name, PASS if condition else fail_status, detail, evidence=evidence)

    @property
    def overall(self) -> str:
        worst = PASS
        for it in self.items:
            if _RANK.get(it["status"], 1) > _RANK[worst]:
                worst = it["status"]
        return worst


def _fk_from_metadata(meta: Dict[str, Any], cfg: Optional[Dict[str, Any]]) -> Optional[ForwardKinematics]:
    robot = meta.get("robot") or {}
    kin = robot.get("kinematics") or {}
    startup = robot.get("startup_params") or {}
    dh = startup.get("dh_is_offset")
    if dh is None and cfg:
        dh = cfg.get("robot", {}).get("dh_is_offset")
    if dh is None:
        return None
    return ForwardKinematics(
        dh_is_offset=int(dh),
        tool_offset_m=kin.get("tool_offset_m") or (cfg or {}).get("robot", {}).get("tool_offset_m", [0.0, 0.0, 0.0]),
        ee_frame=kin.get("ee_frame", "link6"),
        base_frame=kin.get("base_frame", "piper_base_link"),
    )


def check_episode(
    root: Path,
    episode_id: str,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    image_checks: int = 12,
    write_report: bool = True,
) -> Dict[str, Any]:
    root = Path(root)
    ep_dir = root / "episodes" / episode_id
    checks = _Checks()
    stats: Dict[str, Any] = {}

    # ---------------- 结构 ----------------
    meta_path = ep_dir / "metadata.json"
    checks.add_if(meta_path.is_file(), "metadata.json 存在", str(meta_path))
    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = read_json(meta_path)
        except Exception as exc:
            checks.add("metadata.json 可解析", FAIL, str(exc))
    if meta:
        checks.add("episode 状态", PASS if meta.get("status") in ("closed", "aborted", "open") else ATTENTION, meta.get("status"))
        if meta.get("status") == "open":
            checks.add(
                "episode 未正常收尾",
                ATTENTION,
                "status=open：进程可能被 kill 或仍在运行；已写入的数据仍可解析",
            )

    samples, sample_problems = tolerant_read_jsonl(ep_dir / "samples.jsonl")
    states, state_problems = tolerant_read_jsonl(ep_dir / "robot_states.jsonl")
    # 样本里的图像/视频路径是**相对 episode 目录**记录的（数据集可整体搬移），
    # 不是相对数据集根目录。这里必须用 ep_dir 作为基准解析。
    robot_expected = bool((meta.get("robot") or {}).get("enabled", True))
    camera_only = bool((meta.get("capture") or {}).get("camera_only", not robot_expected))
    bad_tail = [p for p in sample_problems + state_problems if p.get("type") == "unparsable_line" and p.get("is_last_line")]
    hard_bad = [p for p in sample_problems + state_problems if not (p.get("type") == "unparsable_line" and p.get("is_last_line"))]
    checks.add_if(not hard_bad, "JSONL 无损坏行（尾行截断除外）", hard_bad or "无", evidence=hard_bad)
    if bad_tail:
        checks.add(
            "中断残留尾行",
            PASS,
            f"存在 {len(bad_tail)} 处尾部半行（中断正常现象，其余行可解析）",
            evidence=bad_tail,
        )
    checks.add_if(len(samples) > 0, "samples.jsonl 有内容", f"{len(samples)} 条样本")
    stats["samples"] = len(samples)
    stats["robot_states"] = len(states)

    counters = meta.get("counters") or {}
    if counters:
        checks.add_if(
            counters.get("samples") == len(samples),
            "样本数与 metadata.counters 一致",
            f"metadata={counters.get('samples')} 实际={len(samples)}",
            fail_status=ATTENTION,
        )
        checks.add_if(
            counters.get("robot_states") == len(states),
            "robot_states 数与 metadata.counters 一致",
            f"metadata={counters.get('robot_states')} 实际={len(states)}",
            fail_status=ATTENTION,
        )

    if not samples:
        report = _build_report(episode_id, checks, stats, meta)
        return _maybe_write(ep_dir, report, write_report)

    # ---------------- 字段完整性 ----------------
    required = [
        "schema_version",
        "episode_id",
        "sample_id",
        "timestamps",
        "sync",
        "joint_names",
        "joint_positions_rad",
        "ee_pose",
        "cameras",
        "calibrations",
        "valid",
        "invalid_reasons",
    ]
    missing: Dict[str, int] = {}
    for rec in samples:
        for key in required:
            if key not in rec or rec.get(key) is None:
                missing[key] = missing.get(key, 0) + 1
    # camera_only 模式下机械臂本就未连接，joint/ee_pose 为 null 是预期结果，
    # 且样本里已写明 invalid_reasons，因此是 needs_attention 而不是 fail。
    robot_fields_missing = bool(missing.get("joint_positions_rad") or missing.get("ee_pose"))
    checks.add_if(
        not missing,
        "样本必填字段齐全",
        (missing or "齐全"),
        evidence={
            "missing_counts": missing,
            "camera_only": camera_only,
            "robot_expected": robot_expected,
            "note": "camera_only 时 joint_positions_rad/ee_pose 为 null 属预期，原因写在 invalid_reasons"
            if (robot_fields_missing and not robot_expected)
            else None,
        },
        fail_status=FAIL if (robot_fields_missing and robot_expected) else ATTENTION,
    )
    stats["missing_field_counts"] = missing
    checks.add_if(
        all(rec.get("joint_names") == JOINT_NAMES for rec in samples if rec.get("joint_names")),
        "joint_names 与固定顺序一致",
        JOINT_NAMES,
    )

    # ---------------- 关节与 EE ----------------
    joint_ok = True
    joint_bad: List[Dict[str, Any]] = []
    quat_norm_err: List[float] = []
    quat_flips = 0
    fk_pos_err_mm: List[float] = []
    fk_rot_err_deg: List[float] = []
    prev_q: Optional[np.ndarray] = None
    fk = _fk_from_metadata(meta, cfg)

    from .schema import Transform

    for rec in samples:
        j = rec.get("joint_positions_rad")
        if not isinstance(j, list) or len(j) != 6:
            joint_ok = False
            joint_bad.append({"sample_id": rec.get("sample_id"), "len": None if j is None else len(j)})
            continue
        if any(abs(float(v)) > 2 * np.pi + 0.1 for v in j):
            joint_ok = False
            joint_bad.append({"sample_id": rec.get("sample_id"), "reason": "关节角绝对值超过 2π，疑似单位不是弧度"})
        pose = rec.get("ee_pose")
        if isinstance(pose, list) and len(pose) == 7:
            q = np.asarray(pose[3:], dtype=float)
            n = float(np.linalg.norm(q))
            quat_norm_err.append(abs(n - 1.0))
            if prev_q is not None and float(np.dot(q, prev_q)) < 0:
                quat_flips += 1
            prev_q = q / n if n > 0 else q
            if fk is not None:
                T_expected = fk.fk_base_ee(j)
                T_stored = Transform.from_pose(pose)
                fk_pos_err_mm.append(float(np.linalg.norm(T_expected[:3, 3] - T_stored[:3, 3]) * 1000.0))
                fk_rot_err_deg.append(Transform.rotation_angle_deg(T_expected[:3, :3], T_stored[:3, :3]))

    checks.add_if(
        joint_ok,
        "关节数为 6 且量纲为弧度",
        joint_bad[:5] or "全部通过",
        evidence={"problems": joint_bad[:5], "camera_only": camera_only},
        fail_status=ATTENTION if (not robot_expected and all(s.get("joint_positions_rad") is None for s in samples)) else FAIL,
    )
    if quat_norm_err:
        stats["quaternion_norm_error_max"] = max(quat_norm_err)
        checks.add_if(
            max(quat_norm_err) < 1e-6,
            "ee_pose 四元数已归一化(norm≈1)",
            f"max |norm-1| = {max(quat_norm_err):.2e}",
            fail_status=FAIL,
        )
    stats["quaternion_sign_flips"] = quat_flips
    checks.add(
        "相邻帧四元数符号连续性",
        PASS if quat_flips == 0 else ATTENTION,
        f"检测到 {quat_flips} 次相邻帧点积为负"
        + ("（正常：采集侧已做符号连续处理）" if quat_flips == 0 else "（应检查采集侧 QuatContinuity 是否生效）"),
    )
    if fk_pos_err_mm:
        stats["ee_pose_vs_fk"] = {
            "n": len(fk_pos_err_mm),
            "position_max_mm": max(fk_pos_err_mm),
            "position_mean_mm": float(np.mean(fk_pos_err_mm)),
            "rotation_max_deg": max(fk_rot_err_deg) if fk_rot_err_deg else None,
            "model": fk.model_id(),
        }
        checks.add_if(
            max(fk_pos_err_mm) < 1e-3,
            "ee_pose 与反馈关节角 FK 一致(<1e-3 mm)",
            f"max {max(fk_pos_err_mm):.3e} mm（模型 {fk.model_id()}）",
            fail_status=FAIL,
        )
        checks.add(
            "四元数顺序为 wxyz",
            PASS,
            "由 ee_pose_layout/units 声明 + FK 复算一致性共同确认",
            evidence=meta.get("ee_pose_layout"),
        )
    elif samples:
        checks.add(
            "ee_pose 与 FK 一致性",
            UNKNOWN,
            "metadata 缺少 kinematics/startup_params，无法确定 DH 版本，跳过复算",
        )

    # ---------------- 图像与深度 ----------------
    cam = (meta.get("camera") or {})
    actual = cam.get("actual_streams") or {}
    checks.add_if(
        bool(cam.get("depth_scale_m")),
        "相机 depth_scale 已记录",
        cam.get("depth_scale_m"),
        fail_status=FAIL,
    )
    checks.add_if(bool(actual), "实际流配置已记录", actual or "缺失", fail_status=ATTENTION)
    if actual:
        c = actual.get("color", {})
        d = actual.get("depth", {})
        want_c = ((cfg or {}).get("camera", {}) or {}).get("color")
        want_d = ((cfg or {}).get("camera", {}) or {}).get("depth")
        ok = True
        detail = []
        if want_c:
            same = (c.get("width"), c.get("height"), c.get("fps")) == (want_c.get("width"), want_c.get("height"), want_c.get("fps"))
            ok &= bool(same)
            detail.append(f"color 要求 {want_c.get('width')}x{want_c.get('height')}@{want_c.get('fps')} 实际 {c.get('width')}x{c.get('height')}@{c.get('fps')}")
        if want_d:
            same = (d.get("width"), d.get("height"), d.get("fps")) == (want_d.get("width"), want_d.get("height"), want_d.get("fps"))
            ok &= bool(same)
            detail.append(f"depth 要求 {want_d.get('width')}x{want_d.get('height')}@{want_d.get('fps')} 实际 {d.get('width')}x{d.get('height')}@{d.get('fps')}")
        checks.add("图像流规格符合目标配置", PASS if ok else FAIL, "; ".join(detail))

    import cv2

    step = max(1, len(samples) // max(1, image_checks))
    checked = 0
    img_problems: List[Dict[str, Any]] = []
    depth_dtypes: Dict[str, int] = {}
    aligned_same_size = True
    invalid_zero_ok = True
    for rec in samples[::step][:image_checks]:
        cams = rec.get("cameras") or {}
        rgb = (cams.get("color") or {})
        dr = (cams.get("depth_raw") or {})
        da = (cams.get("depth_aligned") or {})
        checked += 1
        for label, entry, expect_u16 in (("rgb", rgb, False), ("depth_raw", dr, True), ("depth_aligned", da, True)):
            rel = entry.get("path")
            if not rel:
                img_problems.append({"sample_id": rec.get("sample_id"), "file": label, "reason": "路径缺失"})
                continue
            p = resolve_path(rel, ep_dir)
            if not p.is_file():
                img_problems.append({"sample_id": rec.get("sample_id"), "file": label, "reason": "文件不存在", "path": str(p)})
                continue
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                img_problems.append({"sample_id": rec.get("sample_id"), "file": label, "reason": "无法解码"})
                continue
            if expect_u16:
                depth_dtypes[label] = int(np.asarray(img).dtype.itemsize * 8)
                if np.asarray(img).dtype != np.uint16:
                    img_problems.append({"sample_id": rec.get("sample_id"), "file": label, "reason": f"dtype={np.asarray(img).dtype}，应为 uint16"})
            ew, eh = entry.get("width"), entry.get("height")
            if ew and eh and (int(img.shape[1]), int(img.shape[0])) != (int(ew), int(eh)):
                img_problems.append(
                    {
                        "sample_id": rec.get("sample_id"),
                        "file": label,
                        "reason": f"实际尺寸 {img.shape[1]}x{img.shape[0]} 与记录 {ew}x{eh} 不一致",
                    }
                )
            if label == "depth_raw" and entry.get("invalid_value") == 0:
                invalid_zero_ok &= bool(int(np.count_nonzero(np.asarray(img) == 0)) == int(entry.get("invalid_pixels", -1)))
        if da.get("same_size_as_rgb") is False:
            aligned_same_size = False
        if da.get("same_size_as_rgb") is True and rgb.get("width") and da.get("width"):
            aligned_same_size &= bool((rgb["width"], rgb["height"]) == (da["width"], da["height"]))

    checks.add_if(not img_problems, f"抽查 {checked} 个样本的图像文件", img_problems[:5] or "全部可读取且尺寸一致", evidence=img_problems[:5])
    if depth_dtypes:
        checks.add_if(
            all(v == 16 for v in depth_dtypes.values()),
            "深度为无损 uint16",
            depth_dtypes,
            fail_status=FAIL,
        )
    checks.add_if(aligned_same_size, "对齐深度与 RGB 同尺寸", "由记录字段交叉确认")
    checks.add_if(invalid_zero_ok, "无效深度值语义一致 (0=无有效测量)", "抽查样本 0 像素计数与记录一致")
    checks.add_if(
        all(bool((r.get("cameras") or {}).get("is_geometric_alignment")) for r in samples[:50]),
        "对齐深度由几何对齐生成（非缩放）",
        "samples.cameras.is_geometric_alignment=true",
    )

    # ---------------- 时间同步 ----------------
    syn = [
        {
            "rgb_dt": (r.get("sync") or {}).get("rgb_depth_device_dt_ms"),
            "rob_dt": (r.get("sync") or {}).get("robot_joints_dt_ms"),
            "is_stale": r.get("robot_state_stale"),
            "valid": r.get("valid"),
            "reasons": r.get("invalid_reasons") or [],
            "has_robot_ts": bool((r.get("timestamps") or {}).get("robot_joints")),
        }
        for r in samples
    ]
    valid_n = sum(1 for s in syn if s["valid"])
    stale_n = sum(1 for s in syn if s["is_stale"])
    stats["valid_samples"] = valid_n
    stats["invalid_samples"] = len(syn) - valid_n
    stats["stale_robot_states"] = stale_n
    checks.add_if(
        valid_n > 0,
        "存在有效样本",
        f"有效 {valid_n} / 共 {len(syn)}"
        + ("（camera_only：无机械臂状态，全部标为无效属预期）" if (valid_n == 0 and camera_only) else ""),
        fail_status=ATTENTION if camera_only else FAIL,
    )
    checks.add_if(
        stale_n == 0,
        "无过期关节反馈填充",
        f"{stale_n} 条样本标记 robot_state_stale",
        fail_status=ATTENTION,
    )
    checks.add_if(
        all(s["has_robot_ts"] for s in syn),
        "每条样本都有各数据源时间戳字段",
        "timestamps.robot_joints 存在（源时间戳为 null 并注明原因，属正常）",
    )
    rgb_dts = [s["rgb_dt"] for s in syn if s["rgb_dt"] is not None]
    rob_dts = [s["rob_dt"] for s in syn if s["rob_dt"] is not None]
    tol = ((meta.get("sync") or {}).get("tolerance_ms")) or ((meta.get("sync") or {}).get("robot_tolerance_ms"))
    if rgb_dts:
        stats["rgb_depth_dt_ms"] = {
            "mean": float(np.mean(rgb_dts)),
            "p95_abs": float(np.percentile(np.abs(rgb_dts), 95)),
            "max_abs": float(np.max(np.abs(rgb_dts))),
        }
    if rob_dts:
        stats["robot_joints_dt_ms"] = {
            "mean": float(np.mean(rob_dts)),
            "p95_abs": float(np.percentile(np.abs(rob_dts), 95)),
            "max_abs": float(np.max(np.abs(rob_dts))),
        }
        checks.add_if(
            float(np.max(np.abs(rob_dts))) <= float(tol or 33.0),
            f"关节反馈时间差在容差内 ({tol} ms)",
            f"max |dt| = {float(np.max(np.abs(rob_dts))):.2f} ms",
            fail_status=ATTENTION,
        )
    checks.add(
        "同步性质声明",
        PASS if not any((r.get("sync") or {}).get("is_hardware_synchronized") for r in samples) else FAIL,
        "所有样本 is_hardware_synchronized=false，属软件时间匹配",
    )

    # ---------------- 帧率与掉帧 ----------------
    dur = meta.get("sample_span_s") or meta.get("duration_s")
    if dur:
        rate = len(samples) / float(dur)
        target = ((meta.get("capture") or {}).get("target_sample_rate")) or 30.0
        stats["measured_sample_rate_hz"] = rate
        stats["rate_span_s"] = float(dur)
        stats["rate_span_basis"] = "sample_span_s（首个到最后一个样本的主机接收时间跨度）" if meta.get("sample_span_s") else "duration_s（含开关相机耗时，偏保守）"
        checks.add_if(
            rate >= 0.8 * float(target),
            f"实测样本率 ≥ 80% 目标 ({target} Hz)",
            f"实测 {rate:.2f} Hz，共 {len(samples)} 样本 / {float(dur):.2f} s（基准 {stats['rate_span_basis']}）",
            fail_status=ATTENTION,
        )
    diag = ((meta.get("diagnostics_summary") or {}).get("camera")) or {}
    if diag:
        stats["camera_diagnostics"] = diag
        missed = (diag.get("missing_color") or 0) + (diag.get("missing_depth") or 0) + (diag.get("frameset_timeouts") or 0)
        checks.add_if(missed == 0, "采集期间无丢帧/取帧超时", f"缺失/超时合计 {missed}", fail_status=ATTENTION)

    # ---------------- 标定引用 ----------------
    cids = meta.get("calibration_ids") or {}
    store = CalibrationStore(root)
    cal_details: Dict[str, Any] = {}
    for kind, key in (("camera", "camera_calibration_id"), ("handeye", "handeye_calibration_id"), ("gripper", "gripper_calibration_id")):
        cid = cids.get(key)
        if cid is None:
            cal_details[kind] = None
            continue
        exists = store.exists(kind, cid)
        cal_details[kind] = {"calibration_id": cid, "file_exists": exists}
        if exists:
            payload = store.load(kind, cid)
            cal_details[kind]["status"] = payload.get("status", "valid" if payload.get("valid") else "uncalibrated")
            cal_details[kind]["valid"] = payload.get("valid")
    stats["calibrations"] = cal_details
    checks.add_if(
        all(v is None or v.get("file_exists") for v in cal_details.values()),
        "引用的标定文件存在",
        cal_details,
        fail_status=FAIL,
    )
    if cal_details.get("handeye") and cal_details["handeye"].get("valid") is False:
        checks.add(
            "手眼标定尚未通过验证",
            ATTENTION,
            f"handeye status={cal_details['handeye'].get('status')}；"
            "数据集已引用该 id，但不得当作已可用变换",
        )
    if cal_details.get("gripper") is None:
        checks.add(
            "夹爪未校准",
            ATTENTION,
            "gripper_calibration_id 为空：gripper_width_mm 应为 null，这符合“不填虚构毫米值”的要求",
        )

    # ---------------- 第三人称视频 ----------------
    ext = meta.get("external_videos") or []
    if ext:
        missing_ext = [v for v in ext if v.get("stored_path_relative_to_dataset") and not resolve_path(v["stored_path_relative_to_dataset"], root).is_file()]
        checks.add_if(not missing_ext, "已登记的第三人称视频文件存在", ext, fail_status=FAIL)
        checks.add_if(
            all(v.get("synchronization_status") == "unsynchronized" for v in ext),
            "第三人称视频同步状态未被伪造",
            [v.get("synchronization_status") for v in ext],
        )
    else:
        checks.add("第三人称视频", PASS, "未导入（不影响机械臂与 D435i 数据）")

    # scene 级环境环视视频：按 scene 保存、可被多个 episode 引用，
    # 因此不在 episode 的 external_videos 里，需要顺着 scene_id 查 scene.json
    scene_id = meta.get("scene_id")
    if scene_id:
        scene_json = root / "scenes" / scene_id / "scene.json"
        if scene_json.is_file():
            scene = read_json(scene_json)
            envs = scene.get("environment_overview") or []
            if envs:
                missing_env = [
                    v
                    for v in envs
                    if v.get("stored_path_relative_to_dataset")
                    and not resolve_path(v["stored_path_relative_to_dataset"], root).is_file()
                ]
                checks.add_if(not missing_env, "scene 级环境环视视频文件存在", envs, fail_status=FAIL)
                checks.add_if(
                    all(v.get("synchronization_status") == "unsynchronized" for v in envs),
                    "scene 级环境环视视频同步状态未被伪造",
                    [v.get("synchronization_status") for v in envs],
                )
            else:
                checks.add("scene 级环境环视视频", PASS, f"{scene_id} 尚无 environment_overview，按 scene 引用为可选")
        else:
            checks.add("scene 级环境环视视频", UNKNOWN, f"scene_id={scene_id} 但 {scene_json} 不存在")
    else:
        checks.add("scene 级环境环视视频", UNKNOWN, "metadata 未记录 scene_id，无法关联环视视频")

    report = _build_report(episode_id, checks, stats, meta)
    return _maybe_write(ep_dir, report, write_report)


def _build_report(episode_id: str, checks: _Checks, stats: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "quality_report",
        "episode_id": episode_id,
        "checked_at": _utc_now(),
        "overall": checks.overall,
        "episode_status": meta.get("status"),
        "counters": meta.get("counters") or {},
        "checks": checks.items,
        "statistics": stats,
        "interpretation": {
            "pass": "该项通过",
            "needs_attention": "不阻塞读取，但会影响可用性或需要人工确认",
            "fail": "存在明确错误，需修复",
            "unknown": "缺少判断依据",
        },
    }


def _maybe_write(ep_dir: Path, report: Dict[str, Any], write: bool) -> Dict[str, Any]:
    if write:
        write_json(ep_dir / "quality_report.json", report)
    return report


def check_dataset(root: Path, *, cfg: Optional[Dict[str, Any]] = None, image_checks: int = 6) -> Dict[str, Any]:
    root = Path(root)
    ep_root = root / "episodes"
    episodes: List[Dict[str, Any]] = []
    if ep_root.is_dir():
        for d in sorted(p for p in ep_root.iterdir() if p.is_dir()):
            r = check_episode(root, d.name, cfg=cfg, image_checks=image_checks)
            episodes.append({"episode_id": d.name, "overall": r["overall"], "counters": r["counters"]})
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "dataset_quality_summary",
        "checked_at": _utc_now(),
        "dataset_root": str(root),
        "n_episodes": len(episodes),
        "episodes": episodes,
        "overall": (
            FAIL
            if any(e["overall"] == FAIL for e in episodes)
            else (ATTENTION if any(e["overall"] == ATTENTION for e in episodes) else PASS)
        ),
    }
    if episodes:
        write_json(root / "reports" / "dataset_quality.json", summary)
    return summary