"""夹爪诊断与开度校准（独立入口）。

设计原则：
  - **诊断只读**：`diagnose()` 只被动观察 CAN 反馈，不发送任何夹爪命令。
    主采集流程不会调用本模块，避免反复下发命令干扰采集。
  - **主动探测单独隔离**：只有 `probe()` 会发送 `GripperCtrl`，必须显式传
    `allow_motion=True`，并在报告中写明 `motion_commands_sent=true`。
  - **不伪造毫米值**：未完成实物多点测量时，校准文件 `valid=false`，
    数据集里 `gripper_width_mm` 为 null 并带原因。

已定位的可疑点（被动证据，见 diagnose 输出）：
  1. 单位/范围：驱动反馈 `grippers_angle` 单位 0.001 mm，全行程 [0, 70] mm
     对应原始值 [0, 70000]；官方 `piper_param_manager.gripper_range=[0.0,0.07]`。
     超过该范围的原始值说明量纲或零位有问题，直接报出来。
  2. 控制模式/使能：status_code bit6 是使能位、bit5 驱动错误、bit4 传感器异常。
     使能位为 0 时下发的行程指令不会被执行，表现为“指令后开合不稳定”。
  3. 发送频率/并发：官方示例 `piper_ctrl_gripper.py` 每 5 ms（约 200 Hz）
    连续流式下发同一目标值。若上位程序同时有多个线程/进程周期发包，
    夹爪会在多个目标间抖动。诊断会统计实际反馈更新频率与是否有夹爪独占
    反馈帧（`gripper_only_updates`），用于判断是否存在并发下发。
  4. 反馈延迟：夹爪反馈是独立的 CAN 帧（0x2A8），与关节帧不同步；诊断输出
     夹爪反馈更新率与最近一次更新的时间差。
"""
from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .episode import CalibrationStore
from .jsonio import write_json
from .robot import GripperCalibration, RobotReader, RobotState
from .schema import SCHEMA_VERSION, decode_gripper_status


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def reports_dir(root: Path) -> Path:
    d = Path(root) / "reports" / "gripper"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- 被动诊断


def _summarize(samples: Sequence[RobotState], raw_unit_mm: float, driver_range_mm: Sequence[float]) -> Dict[str, Any]:
    raws = [s.gripper_feedback_raw for s in samples if s.gripper_feedback_raw is not None]
    efforts = [s.gripper_effort_raw for s in samples if s.gripper_effort_raw is not None]
    codes = [s.gripper_status_code for s in samples if s.gripper_status_code is not None]
    times = [s.gripper_host_recv_ns for s in samples if s.gripper_host_recv_ns is not None]

    distinct_codes = sorted(set(int(c) for c in codes))
    bit_hist: Dict[str, int] = {}
    for c in codes:
        for name, val in (decode_gripper_status(int(c)) or {}).items():
            if val:
                bit_hist[name] = bit_hist.get(name, 0) + 1

    out: Dict[str, Any] = {
        "samples": len(samples),
        "gripper_feedback_frames": len(raws),
        "distinct_status_codes": distinct_codes,
        "status_bit_counts": bit_hist,
        "status_bit_decode": {str(c): decode_gripper_status(c) for c in distinct_codes},
    }
    if raws:
        arr = np.asarray(raws, dtype=float)
        lo, hi = float(np.min(arr)), float(np.max(arr))
        out["raw"] = {
            "min": lo,
            "max": hi,
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "range": hi - lo,
            "distinct_values": int(len(set(raws))),
            "converted_mm_min": lo * raw_unit_mm,
            "converted_mm_max": hi * raw_unit_mm,
            "converted_mm_span": (hi - lo) * raw_unit_mm,
        }
        drv_lo, drv_hi = float(driver_range_mm[0]), float(driver_range_mm[1])
        raw_lo, raw_hi = drv_lo / raw_unit_mm, drv_hi / raw_unit_mm
        out["range_check"] = {
            "driver_range_mm": [drv_lo, drv_hi],
            "expected_raw_range": [raw_lo, raw_hi],
            "raw_unit_mm": raw_unit_mm,
            "below_range_count": int(np.count_nonzero(arr < raw_lo)),
            "above_range_count": int(np.count_nonzero(arr > raw_hi)),
            "verdict": (
                "范围内"
                if np.count_nonzero(arr < raw_lo) == 0 and np.count_nonzero(arr > raw_hi) == 0
                else "存在超出驱动标称行程的原始值：需要核对量纲/零位/回零状态"
            ),
        }
    if efforts:
        e = np.asarray(efforts, dtype=float)
        out["effort"] = {
            "raw_unit": "0.001 N*m",
            "min": float(np.min(e)),
            "max": float(np.max(e)),
            "std": float(np.std(e)),
            "near_stall_count": int(np.count_nonzero(e >= 4000)),
            "note": "接近 5000 的扭矩意味着堵转/夹紧，长时间保持会触发过流或过温",
        }
    if len(times) >= 2:
        t = np.asarray(sorted(int(v) for v in times), dtype=np.int64)
        gaps_ms = np.diff(t) / 1e6
        span = (int(t[-1]) - int(t[0])) / 1e9
        out["feedback_timing"] = {
            "span_s": span,
            "update_rate_hz": (len(t) - 1) / span if span > 0 else None,
            "gap_median_ms": float(np.median(gaps_ms)),
            "gap_p95_ms": float(np.percentile(gaps_ms, 95)),
            "gap_max_ms": float(np.max(gaps_ms)),
            "note": "夹爪反馈是独立 CAN 帧(0x2A8)，其更新率不等于关节反馈率",
        }
    out["enable_state"] = {
        "driver_enabled_ever_true": bool(bit_hist.get("driver_enabled", 0) > 0),
        "homing_done_ever_true": bool(bit_hist.get("homing_done", 0) > 0),
        "note": "bit6=使能，bit7=回零状态；未使能时行程指令不会执行",
    }
    return out


def diagnose(
    cfg: Dict[str, Any],
    root: Path,
    *,
    duration_s: float = 10.0,
    reader: Optional[RobotReader] = None,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """只读被动诊断：观察 N 秒内夹爪反馈，不发送任何命令。"""
    root = Path(root)
    raw_unit_mm = float(cfg["gripper"].get("raw_unit_mm", 0.001))
    driver_range_mm = cfg["gripper"].get("driver_range_mm", [0.0, 70.0])

    own_reader = reader is None
    if own_reader:
        rc = cfg["robot"]
        reader = RobotReader(
            can_interface=rc["can_interface"],
            dh_is_offset=int(rc["dh_is_offset"]),
            poll_hz=float(rc.get("poll_hz", 200.0)),
            queries_on_connect=True,
            feedback_timeout_s=float(rc.get("feedback_timeout_s", 1.0)),
            gripper_calibration=GripperCalibration(),
        )
        if not reader.open(timeout_s=5.0):
            report = {
                "schema_version": SCHEMA_VERSION,
                "kind": "gripper_diagnosis",
                "generated_at": _utc_now(),
                "mode": "passive_readonly",
                "status": "failed",
                "reason": reader.open_error,
                "motion_commands_sent": False,
                "duration_s": duration_s,
                "feedback_sustained": False,
                "driver_counters": reader.diagnostics(),
                "summary": {
                    "samples": 0,
                    "gripper_feedback_frames": 0,
                    "note": "连接阶段就没有收到关节反馈帧，因此没有进入观察窗口；"
                    "行程/使能等字段无数据，不填默认值",
                },
                "findings": [
                    {
                        "severity": "blocker",
                        "finding": "CAN 上读不到任何关节/夹爪反馈帧，无法做夹爪诊断",
                        "evidence": reader.open_error,
                        "next_step": "确认机械臂上电与 CAN 线连接；本项目不会自行激活或修改 CAN 配置",
                    }
                ],
            }
            reader.close()
            return _write_diagnosis(root, report, report_path)

    assert reader is not None
    samples: List[RobotState] = []
    t_end = time.monotonic() + max(0.0, duration_s)
    try:
        while time.monotonic() < t_end:
            st = reader.latest()
            if st is not None and (not samples or samples[-1].state_id != st.state_id):
                samples.append(st)
            time.sleep(0.005)
    finally:
        if own_reader:
            reader.close()

    summary = _summarize(samples, raw_unit_mm, driver_range_mm)
    findings: List[Dict[str, str]] = []
    timing = summary.get("feedback_timing")
    sustained = bool(timing and timing.get("update_rate_hz") is not None)
    if not sustained:
        # 只有 0/1 帧（或时间跨度 0）说明观察窗口内根本没有持续的夹爪反馈。
        # 此时下面的行程/使能结论都建立在"没有数据的默认值 0"上，不能当作实测结论。
        findings.append(
            {
                "severity": "blocker",
                "finding": "观察窗口内没有持续的夹爪反馈帧",
                "evidence": f"gripper_feedback_frames={summary['gripper_feedback_frames']}，"
                f"samples={summary['samples']}，feedback_timing={timing}",
                "next_step": "确认机械臂上电、CAN 线连接、夹爪已接线（0x2A8 反馈）；"
                "在此之前所有行程/使能数值都是默认值 0，不代表实测状态",
            }
        )
    else:
        rng = summary.get("range_check")
        if rng and (rng["below_range_count"] or rng["above_range_count"]):
            findings.append(
                {
                    "severity": "high",
                    "finding": "夹爪原始反馈超出驱动标称行程对应的原始值范围",
                    "evidence": f"expected_raw_range={rng['expected_raw_range']}，"
                    f"实际 [{summary['raw']['min']:.0f}, {summary['raw']['max']:.0f}]",
                    "next_step": "核对 raw_unit_mm 与回零状态(bit7)，确认是否做过 set_zero",
                }
            )
        if not summary["enable_state"]["driver_enabled_ever_true"]:
            findings.append(
                {
                    "severity": "high",
                    "finding": "观察窗口内夹爪驱动器从未处于使能状态（status_code bit6=0）",
                    "evidence": f"distinct_status_codes={summary['distinct_status_codes']}",
                    "next_step": "下发指令前先确认使能（官方流程先发 status_code=0x01 使能）；"
                    "未使能时行程指令不会执行，表现为“发了指令但开合不稳定/无响应”",
                }
            )
        if summary["status_bit_counts"].get("driver_error") or summary["status_bit_counts"].get("sensor_abnormal"):
            findings.append(
                {
                    "severity": "high",
                    "finding": "夹爪反馈中出现驱动器错误或传感器异常位",
                    "evidence": f"status_bit_counts={summary['status_bit_counts']}",
                    "next_step": "先清除错误（官方 status_code=0x02 失能清错）再做开合测试",
                }
            )
        eff = summary.get("effort")
        if eff and eff["near_stall_count"] > 0:
            findings.append(
                {
                    "severity": "medium",
                    "finding": "夹爪扭矩接近上限，存在堵转/夹紧",
                    "evidence": f"effort max={eff['max']} (0.001 N*m)，near_stall_count={eff['near_stall_count']}",
                    "next_step": "降低目标扭矩或减小行程，避免过流保护导致的间歇动作",
                }
            )
        tm = summary.get("feedback_timing")
        if tm and tm["update_rate_hz"] is not None and tm["update_rate_hz"] < 5.0:
            findings.append(
                {
                    "severity": "medium",
                    "finding": "夹爪反馈更新率偏低，上位看到的开度可能滞后",
                    "evidence": f"update_rate_hz={tm['update_rate_hz']:.2f}",
                    "next_step": "对比关节反馈率(约 200 Hz)与夹爪反馈率，判断是总线负载还是夹爪侧上报慢",
                }
            )
        if not findings:
            findings.append(
                {
                    "severity": "info",
                    "finding": "被动反馈未发现明显异常",
                    "evidence": f"raw 范围 [{summary['raw']['min']:.0f}, {summary['raw']['max']:.0f}]，"
                    f"status={summary['distinct_status_codes']}",
                    "next_step": "“下发后开合不稳定”属于动态行为，需要用 probe 做主动探测才能复现",
                }
            )

    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "gripper_diagnosis",
        "generated_at": _utc_now(),
        "mode": "passive_readonly",
        "motion_commands_sent": False,
        "duration_s": duration_s,
        "feedback_sustained": sustained,
        "gripper_config": {
            "raw_unit_mm": raw_unit_mm,
            "driver_range_mm": [float(v) for v in driver_range_mm],
            "calibration_id": cfg["gripper"].get("calibration_id"),
        },
        "driver_counters": reader.diagnostics(),
        "summary": summary,
        "findings": findings,
        "checks_required_by_requirement": {
            "unit_conversion": "raw_unit_mm 来自 piper_sdk 报文定义（0.001 mm），已写入配置并在此核对",
            "range": "与 piper_param_manager.gripper_range=[0.0,0.07] m 对照",
            "control_mode": "由 status_code 位与 GetArmStatus.ctrl_mode 判断（见 driver_counters）",
            "enable_state": "status_code bit6",
            "send_frequency": "官方示例约 200 Hz 连续流式下发；本项目诊断不发送，仅统计被动反馈率",
            "concurrent_commands": "无法从被动反馈直接证明；需用 probe 或在总线上抓包（见 probe 输出）",
            "feedback_latency": "夹爪反馈与关节反馈是不同 CAN 帧，分别统计更新率与间隔",
        },
        "status": "ok" if sustained else "failed",
    }
    return _write_diagnosis(root, report, report_path)


def _write_diagnosis(root: Path, report: Dict[str, Any], report_path: Optional[Path]) -> Dict[str, Any]:
    path = report_path or (reports_dir(root) / f"{_utc_stamp()}-diagnosis.json")
    write_json(path, report)
    report["report_path"] = str(path)
    return report


# --------------------------------------------------------------------------- 主动探测


def probe(
    cfg: Dict[str, Any],
    root: Path,
    *,
    allow_motion: bool,
    targets_mm: Sequence[float] = (0.0, 20.0, 40.0, 60.0, 0.0),
    dwell_s: float = 1.2,
    effort_raw: int = 1000,
    send_hz: float = 200.0,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """主动探测：逐点下发目标行程并记录反馈，用于复现“开合不稳定”。

    ⚠️ 会真实驱动夹爪。必须 allow_motion=True，且现场需确认夹爪行程内无遮挡、
    机械臂周围安全。报告会写明 motion_commands_sent=true。
    """
    if not allow_motion:
        raise PermissionError(
            "probe 会真实驱动夹爪，必须显式传入 allow_motion=True。"
            "请先确认机械臂周围安全、夹爪行程内无遮挡，再运行。"
        )
    from piper_sdk import C_PiperInterface_V2

    root = Path(root)
    rc = cfg["robot"]
    raw_unit_mm = float(cfg["gripper"].get("raw_unit_mm", 0.001))
    driver_range_mm = [float(v) for v in cfg["gripper"].get("driver_range_mm", [0.0, 70.0])]

    piper = C_PiperInterface_V2(
        can_name=rc["can_interface"],
        judge_flag=True,
        can_auto_init=True,
        dh_is_offset=0x01 if int(rc["dh_is_offset"]) else 0x00,
        logger_level=40,
    )
    piper.ConnectPort(can_init=True, piper_init=True, start_thread=True)
    time.sleep(0.3)

    steps: List[Dict[str, Any]] = []
    period = 1.0 / max(1.0, send_hz)
    for target in targets_mm:
        if target < driver_range_mm[0] or target > driver_range_mm[1]:
            steps.append({"target_mm": float(target), "skipped": "超出驱动标称行程"})
            continue
        raw_target = int(round(float(target) / raw_unit_mm))
        sent = 0
        feedback: List[Dict[str, Any]] = []
        t_end = time.monotonic() + max(0.0, dwell_s)
        while time.monotonic() < t_end:
            # 与官方示例一致：先使能(0x01)，再连续流式下发目标
            piper.GripperCtrl(abs(raw_target), int(effort_raw), 0x01, 0)
            sent += 1
            g = piper.GetArmGripperMsgs().gripper_state
            feedback.append(
                {
                    "t_ns": time.time_ns(),
                    "raw": int(g.grippers_angle),
                    "effort": int(g.grippers_effort),
                    "status_code": int(g.status_code),
                }
            )
            time.sleep(period)
        arr = np.asarray([f["raw"] for f in feedback], dtype=float)
        codes = sorted({f["status_code"] for f in feedback})
        steps.append(
            {
                "target_mm": float(target),
                "target_raw": raw_target,
                "commands_sent": sent,
                "sent_hz": sent / max(1e-9, dwell_s),
                "feedback_samples": len(feedback),
                "feedback_raw_final": int(arr[-1]) if arr.size else None,
                "feedback_raw_mean": float(arr.mean()) if arr.size else None,
                "feedback_raw_std": float(arr.std()) if arr.size else None,
                "feedback_raw_span": float(arr.max() - arr.min()) if arr.size else None,
                "feedback_mm_final": float(arr[-1] * raw_unit_mm) if arr.size else None,
                "settle_error_mm": float(abs(arr[-1] * raw_unit_mm - float(target))) if arr.size else None,
                "status_codes": codes,
                "status_decode": {str(c): decode_gripper_status(c) for c in codes},
                "feedback_observations": feedback,
            }
        )

    piper.DisconnectPort()
    unstable = [s for s in steps if s.get("feedback_raw_span") is not None and s["feedback_raw_span"] > 2000]
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "gripper_probe",
        "generated_at": _utc_now(),
        "mode": "active_motion",
        "motion_commands_sent": True,
        "warning": "本报告来自真实下发夹爪命令的探测，需现场确认安全后方可运行",
        "driver": {
            "backend": "piper_sdk_direct_can",
            "can_interface": rc["can_interface"],
            "gripper_ctrl_can_id": "0x159",
            "effort_raw": int(effort_raw),
            "effort_unit": "0.001 N*m, 范围 0-5000",
            "send_hz": send_hz,
        },
        "steps": steps,
        "conclusions": [
            {
                "severity": "high" if unstable else "info",
                "finding": (
                    "在保持目标不变时反馈仍在抖动，指向并发下发或反馈/机械回差问题"
                    if unstable
                    else "各目标点静止窗口内的反馈抖动在阈值内"
                ),
                "evidence": f"抖动 > 2000 raw(2 mm) 的目标点: {[s['target_mm'] for s in unstable]}",
            }
        ],
    }
    path = report_path or (reports_dir(root) / f"{_utc_stamp()}-probe.json")
    write_json(path, report)
    report["report_path"] = str(path)
    return report


# --------------------------------------------------------------------------- 开度校准


def fit_calibration(
    points: Sequence[Tuple[float, float]],
    *,
    raw_unit_mm: float = 0.001,
    driver_range_mm: Sequence[float] = (0.0, 70.0),
) -> Dict[str, Any]:
    """由 (原始反馈, 实测两指间距 mm) 多点测量拟合线性映射。

    返回拟合参数与误差报告；不写入数据集，由 `calibrate()` 负责保存。
    """
    if len(points) < 3:
        raise ValueError(f"线性拟合至少需要 3 个测量点，收到 {len(points)}")
    arr = np.asarray(points, dtype=float)
    raw = arr[:, 0]
    meas = arr[:, 1]
    if len(set(raw.tolist())) < 3:
        raise ValueError("原始反馈值至少要有 3 个互不相同的点，否则无法拟合")

    slope, intercept = np.polyfit(raw, meas, 1)
    pred = slope * raw + intercept
    resid = pred - meas
    rms = float(np.sqrt(np.mean(resid ** 2)))
    max_abs = float(np.max(np.abs(resid)))
    n = len(points)
    # 自由度 n-2；n>=3 保证分母为正
    std_err = float(np.sqrt(np.sum(resid ** 2) / (n - 2)))
    span_mm = float(meas.max() - meas.min())
    drv_lo, drv_hi = float(driver_range_mm[0]), float(driver_range_mm[1])

    return {
        "model": "linear_total_opening: width_mm = slope * raw + intercept",
        "slope_mm_per_raw_unit": float(slope),
        "intercept_mm": float(intercept),
        "residual_rms_mm": rms,
        "residual_max_abs_mm": max_abs,
        "residual_std_error_mm": std_err,
        "n_points": n,
        "measured_span_mm": span_mm,
        "measured_raw_range": [float(raw.min()), float(raw.max())],
        "valid_raw_range": [float(raw.min()), float(raw.max())],
        "driver_range_mm": [drv_lo, drv_hi],
        "unit_check": {
            "expected_slope_if_raw_is_total_opening_0p001mm": raw_unit_mm,
            "fitted_slope": float(slope),
            "slope_ratio_vs_expected": float(slope) / raw_unit_mm,
            "note": (
                "斜率偏离 0.001 说明驱动原始值不是简单的总开度，或存在连杆/回差；"
                "比值同时用于检查是否把单侧行程当成总行程（比值≈0.5）"
            ),
        },
        "coverage_warning": (
            None
            if span_mm >= 0.5 * (drv_hi - drv_lo)
            else f"实测跨度只有 {span_mm:.1f} mm，不足标称行程的一半，范围外为外推，不应用于数据集"
        ),
        "points": [{"raw": float(r), "measured_width_mm": float(m)} for r, m in points],
    }


def calibrate(
    cfg: Dict[str, Any],
    root: Path,
    *,
    points: Sequence[Tuple[float, float]],
    calibration_id: str,
    operator: Optional[str] = None,
    notes: Optional[str] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """保存夹爪开度校准。测量点必须来自实物测量。"""
    root = Path(root)
    fit = fit_calibration(
        points,
        raw_unit_mm=float(cfg["gripper"].get("raw_unit_mm", 0.001)),
        driver_range_mm=cfg["gripper"].get("driver_range_mm", [0.0, 70.0]),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "gripper",
        "calibration_id": calibration_id,
        "valid": True,
        "calibrated": True,
        "method": "manual_physical_measurement_linear_fit",
        "operator": operator,
        "notes": notes,
        "measured_at": _utc_now(),
        "raw_unit_mm": float(cfg["gripper"].get("raw_unit_mm", 0.001)),
        "unit_of_measured_width": "mm (两指外侧间距，总开度)",
        "status_reason": None,
        **fit,
    }
    if save:
        path = CalibrationStore(root).save("gripper", calibration_id, payload)
        payload["saved_path"] = str(path)
    return payload


def pending_calibration(calibration_id: Optional[str], reason: str) -> Dict[str, Any]:
    """未完成实物测量时的占位校准记录（不写数据集里的毫米值）。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "gripper",
        "calibration_id": calibration_id,
        "valid": False,
        "calibrated": False,
        "status_reason": reason,
        "gripper_width_mm_policy": "未校准时数据集写 null 并给出原因，不填虚构毫米值",
        "required_steps": [
            "1. 保持机械臂上电、夹爪接线，程序只读观察（diagnose）确认有 0x2A8 反馈",
            "2. 用游标卡尺在至少 3 个不同目标行程处测量两指实际间距（推荐 5 点：0/15/30/45/60 mm）",
            "3. 每个点等夹爪静止后记录反馈原始值，重复 3 次取中位数，记录原始重复性",
            "4. 把 (原始值, 实测 mm) 交给 gripper calibrate --point raw:measured_mm ...",
            "5. 检查拟合残差 RMS 与覆盖范围，确认后再把 calibration_id 写入配置",
        ],
    }


def load_calibration(root: Path, calibration_id: Optional[str]) -> GripperCalibration:
    if not calibration_id:
        return GripperCalibration()
    store = CalibrationStore(Path(root))
    if not store.exists("gripper", calibration_id):
        return GripperCalibration()
    return GripperCalibration(store.load("gripper", calibration_id))