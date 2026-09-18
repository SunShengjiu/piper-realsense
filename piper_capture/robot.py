"""只读 PiPER 机械臂反馈采集。

复用当前可用的 piper_sdk 0.6.2 + can0，不重建连接、不改固件、不动零位。
本模块**只读取状态**：
  - 不调用 GripperCtrl / JointCtrl / EnablePiper / MotionCtrl 等任何下发接口；
  - 连接时的 PiperInit 只发送官方定义的“查询”帧（读固件、读电机限位/加速度），
    可用 `queries_on_connect=False` 关闭。

时间戳口径（重要）：
  PiPER 的 CAN 反馈协议不带设备侧时间戳。SDK 暴露的 `time_stamp` 来自
  python-can 的报文接收时刻，属于**主机接收时间**，不是机械臂时钟。
  因此本模块对关节/夹爪反馈只写 `host_recv_ns`，`source_timestamp_ns` 保持
  null 并注明原因，不伪造设备时间。
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

from .kinematics import ForwardKinematics, QuatContinuityTracker
from .schema import JOINT_NAMES, decode_gripper_status

_MDEG_TO_RAD = math.pi / 180.0 / 1000.0  # SDK 反馈单位 0.001 度 -> rad


@dataclass
class RobotState:
    """一帧机械臂状态快照。关节与夹爪各自带主机接收时间。"""

    state_id: int
    # --- 关节反馈 ---
    joints_rad: List[float]
    joints_raw_mdeg: List[int]
    joints_host_recv_ns: int
    joints_source_timestamp_ns: Optional[int]
    joints_clock_source: str
    # --- 夹爪反馈（独立字段，绝不混入弧度数组）---
    gripper_feedback_raw: Optional[int]
    gripper_effort_raw: Optional[int]
    gripper_status_code: Optional[int]
    gripper_host_recv_ns: Optional[int]
    gripper_valid: bool
    gripper_width_mm: Optional[float]
    gripper_calibration_id: Optional[str]
    # --- 末端位姿（由反馈关节角做 FK）---
    ee_pose: Optional[List[float]]
    ee_pose_frame: str
    ee_pose_source: str
    # --- 驱动自带末端反馈，独立字段 ---
    driver_end_pose_raw: Optional[Dict[str, Any]]
    # --- 状态 ---
    arm_status: Optional[Dict[str, Any]]
    enable: Optional[List[bool]]
    feedback_hz: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": None,  # 由 episode 层写入
            "robot_state_id": self.state_id,
            "joint_names": list(JOINT_NAMES),
            "joint_positions_rad": [float(v) for v in self.joints_rad],
            "joint_positions_raw_0p001deg": [int(v) for v in self.joints_raw_mdeg],
            "joints_host_recv_ns": int(self.joints_host_recv_ns),
            "joints_source_timestamp_ns": self.joints_source_timestamp_ns,
            "joints_clock_source": self.joints_clock_source,
            "gripper_feedback_raw": self.gripper_feedback_raw,
            "gripper_feedback_raw_unit": "0.001 mm (driver raw)",
            "gripper_effort_raw": self.gripper_effort_raw,
            "gripper_effort_raw_unit": "0.001 N*m",
            "gripper_status_code": self.gripper_status_code,
            "gripper_status_bits": decode_gripper_status(self.gripper_status_code),
            "gripper_host_recv_ns": self.gripper_host_recv_ns,
            "gripper_valid": bool(self.gripper_valid),
            "gripper_width_mm": self.gripper_width_mm,
            "gripper_calibration_id": self.gripper_calibration_id,
            "ee_pose": self.ee_pose,
            "ee_pose_frame": self.ee_pose_frame,
            "ee_pose_source": self.ee_pose_source,
            "driver_end_pose_raw": self.driver_end_pose_raw,
            "arm_status": self.arm_status,
            "enable": self.enable,
            "feedback_hz": self.feedback_hz,
        }


class GripperCalibration:
    """夹爪原始反馈 -> 总开度(mm) 的校准模型。

    没有实测数据时 `calibrated=False`，`width_mm()` 返回 None，
    数据集写 null 并给出原因，绝不填虚构毫米值。
    """

    def __init__(self, payload: Optional[Dict[str, Any]] = None) -> None:
        self.payload = payload or {}
        self.valid = bool(self.payload.get("valid", False))
        self.calibration_id = self.payload.get("calibration_id")
        self.slope = self.payload.get("slope_mm_per_raw_unit")
        self.intercept = self.payload.get("intercept_mm", 0.0)
        rng = self.payload.get("valid_raw_range")
        self.valid_raw_range = rng
        self.reason = None if self.valid else (
            self.payload.get("status_reason") or "夹爪未完成物理开度校准，未保存校准文件"
        )

    def width_mm(self, raw: Optional[int]) -> Optional[float]:
        """返回校准后的两指实际间距(mm)；未校准或超范围时返回 None。"""
        if not self.valid or raw is None or self.slope is None:
            return None
        if self.valid_raw_range is not None:
            lo, hi = self.valid_raw_range
            if lo is not None and raw < lo:
                return None
            if hi is not None and raw > hi:
                return None
        return float(self.slope) * float(raw) + float(self.intercept)


class RobotReader:
    """只读机械臂反馈读取器。

    参数:
        can_interface: socketcan 接口名（默认 can0）
        dh_is_offset: piper_sdk DH 2° 偏置开关，与当前型号/固件代际对应
        queries_on_connect: 是否在连接时发送官方查询帧（读固件/限位），不发运动指令
        on_state: 每产生一个状态快照时回调（在轮询线程内调用，需自行保证轻量）
    """

    def __init__(
        self,
        can_interface: str = "can0",
        *,
        dh_is_offset: int = 0x01,
        poll_hz: float = 200.0,
        queries_on_connect: bool = True,
        feedback_timeout_s: float = 1.0,
        tool_offset_m: Sequence[float] = (0.0, 0.0, 0.0),
        ee_frame: str = "link6",
        base_frame: str = "piper_base_link",
        sdk_joint_limit: bool = False,
        sdk_gripper_limit: bool = False,
        logger_level: int = 40,
        on_state: Optional[Callable[[RobotState], None]] = None,
        buffer_len: int = 4096,
        gripper_calibration: Optional[GripperCalibration] = None,
    ) -> None:
        self.can_interface = can_interface
        self.poll_hz = float(poll_hz)
        self.queries_on_connect = bool(queries_on_connect)
        self.feedback_timeout_s = float(feedback_timeout_s)
        self.logger_level = logger_level
        self.on_state = on_state
        self.sdk_joint_limit = bool(sdk_joint_limit)
        self.sdk_gripper_limit = bool(sdk_gripper_limit)
        self.gripper_cal = gripper_calibration or GripperCalibration()

        self.fk = ForwardKinematics(
            dh_is_offset=dh_is_offset,
            tool_offset_m=tool_offset_m,
            ee_frame=ee_frame,
            base_frame=base_frame,
        )
        self._quat = QuatContinuityTracker()

        self._piper = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._buffer: Deque[RobotState] = deque(maxlen=int(buffer_len))
        self._state_id = 0

        # 各源最新值与更新时间
        self._last_joint_stamp: Any = None
        self._last_gripper_stamp: Any = None
        self._last_gripper: Dict[str, Any] = {
            "raw": None,
            "effort": None,
            "status_code": None,
            "host_ns": None,
        }
        self._last_arm_status: Optional[Dict[str, Any]] = None
        self._last_enable: Optional[List[bool]] = None
        self._last_driver_end_pose: Optional[Dict[str, Any]] = None
        self._last_feedback_hz: Optional[float] = None

        # 统计
        self.counters: Dict[str, Any] = {
            "joint_frames": 0,
            "gripper_frames": 0,
            "state_updates": 0,
            "gripper_only_updates": 0,
            "read_errors": 0,
            "dropped_gaps": 0,
            "max_gap_ms": 0.0,
            "quat_sign_flips": 0,
            "first_joint_host_ns": None,
            "last_joint_host_ns": None,
        }
        self.versions: Dict[str, Any] = {}
        self.open_error: Optional[str] = None

    # ------------------------------------------------------------------ 生命周期
    def open(self, timeout_s: float = 5.0) -> bool:
        """建立只读连接，并等待首帧反馈。返回是否收到反馈。"""
        from piper_sdk import C_PiperInterface_V2

        try:
            self._piper = C_PiperInterface_V2(
                can_name=self.can_interface,
                judge_flag=True,
                can_auto_init=True,
                dh_is_offset=0x01 if self.fk.dh_is_offset else 0x00,
                start_sdk_joint_limit=self.sdk_joint_limit,
                start_sdk_gripper_limit=self.sdk_gripper_limit,
                logger_level=self.logger_level,
            )
        except Exception as exc:  # pragma: no cover - 依赖硬件
            self.open_error = f"创建 CAN 接口失败: {exc}"
            return False
        try:
            self._piper.ConnectPort(can_init=True, piper_init=self.queries_on_connect, start_thread=True)
        except Exception as exc:  # pragma: no cover - 依赖硬件
            self.open_error = f"连接 CAN 失败: {exc}"
            return False

        self.versions = {
            "backend": "piper_sdk_direct_can",
            "sdk_version": self._safe(lambda: str(self._piper.GetCurrentSDKVersion())),
            "interface_version": self._safe(lambda: str(self._piper.GetCurrentInterfaceVersion())),
            "protocol_version": self._safe(lambda: str(self._piper.GetCurrentProtocolVersion())),
            "can_interface": self.can_interface,
            "can_bitrate": 1000000,
            "connect_status": self._safe(self._piper.get_connect_status),
            "dh_is_offset": self.fk.dh_is_offset,
            "queries_on_connect": self.queries_on_connect,
            "read_only": True,
        }

        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="piper-reader", daemon=True)
        self._thread.start()
        return self.wait_for_feedback(timeout_s)

    def _safe(self, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception:
            return None

    def wait_for_feedback(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.counters["joint_frames"] > 0:
                return True
            time.sleep(0.05)
        self.open_error = (
            f"{timeout_s:.1f}s 内未收到任何关节反馈帧。"
            "请检查机械臂是否上电、CAN 线是否接到 %s、总线是否有其他节点。" % self.can_interface
        )
        return False

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._piper is not None:
            try:
                self._piper.DisconnectPort()
            except Exception:
                pass
            self._piper = None

    @property
    def connected(self) -> bool:
        return self._piper is not None and bool(self._safe(self._piper.get_connect_status))

    # ------------------------------------------------------------------ 轮询
    def _poll_loop(self) -> None:
        period = 1.0 / max(1.0, self.poll_hz)
        prev_ns: Optional[int] = None
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self._poll_once()
            except Exception:
                self.counters["read_errors"] += 1
            cur_ns = self.counters["last_joint_host_ns"]
            if cur_ns is not None and (prev_ns is None or cur_ns != prev_ns):
                if prev_ns is not None:
                    gap_ms = (int(cur_ns) - int(prev_ns)) / 1e6
                    if gap_ms > self.feedback_timeout_s * 1000.0:
                        self.counters["dropped_gaps"] += 1
                    if gap_ms > self.counters["max_gap_ms"]:
                        self.counters["max_gap_ms"] = gap_ms
                prev_ns = cur_ns
            dt = period - (time.monotonic() - t0)
            if dt > 0:
                time.sleep(dt)

    def _poll_once(self) -> None:
        piper = self._piper
        if piper is None:
            return
        joint_msg = piper.GetArmJointMsgs()
        js = joint_msg.joint_state
        joint_stamp = getattr(joint_msg, "time_stamp", None)
        joint_raw = [
            int(js.joint_1),
            int(js.joint_2),
            int(js.joint_3),
            int(js.joint_4),
            int(js.joint_5),
            int(js.joint_6),
        ]

        gripper_msg = piper.GetArmGripperMsgs()
        gs = gripper_msg.gripper_state
        gripper_stamp = getattr(gripper_msg, "time_stamp", None)

        # piper_sdk 用 `msg.time_stamp = rx_can_frame.timestamp` 填这两个时间戳
        # （见 piper_protocol_v2.py），默认值是 0。因此 0 表示"这一路反馈帧从未收到过"，
        # 不是"收到了一个时间戳为 0 的帧"。若不排除 0，首次轮询会因 None->0 被记成
        # 收到 1 帧，机械臂没上电时也会让 open() 误判为连接成功并把全 0 关节角写进数据集。
        joint_changed = joint_stamp not in (None, 0) and joint_stamp != self._last_joint_stamp
        gripper_changed = gripper_stamp not in (None, 0) and gripper_stamp != self._last_gripper_stamp

        if joint_changed:
            now_ns = time.time_ns()
            self.counters["joint_frames"] += 1
            first = self.counters["first_joint_host_ns"]
            if first is None:
                first = now_ns
                self.counters["first_joint_host_ns"] = now_ns
            else:
                if now_ns - int(first) > 1_000_000_000:
                    span = (now_ns - int(first)) / 1e9
                    self._last_feedback_hz = self.counters["joint_frames"] / span
            self.counters["last_joint_host_ns"] = now_ns
            self._last_joint_stamp = joint_stamp

        if gripper_changed:
            self.counters["gripper_frames"] += 1
            self._last_gripper = {
                "raw": int(gs.grippers_angle),
                "effort": int(gs.grippers_effort),
                "status_code": int(gs.status_code),
                "host_ns": time.time_ns(),
            }
            self._last_gripper_stamp = gripper_stamp

        if not (joint_changed or gripper_changed):
            return
        if not joint_changed:
            self.counters["gripper_only_updates"] += 1

        self._refresh_slow_status(piper)

        joints_rad = [int(v) * _MDEG_TO_RAD for v in joint_raw]
        ee_pose = None
        if joint_changed:
            T = self.fk.fk_base_ee(joints_rad)
            from .schema import Transform

            ee_pose = self._quat.apply_pose(Transform.to_pose(T))

        gripper_raw = self._last_gripper["raw"]
        status_code = self._last_gripper["status_code"]
        # gripper_valid 只表示“反馈是否可信”：传感器正常且驱动器无错误。
        # 使能状态(bit6)单独记录，不用它来判反馈有效性。
        status_bits = decode_gripper_status(status_code)
        gripper_valid = bool(
            status_code is not None
            and status_bits is not None
            and not status_bits["sensor_abnormal"]
            and not status_bits["driver_error"]
        )

        with self._lock:
            self._state_id += 1
            state = RobotState(
                state_id=self._state_id,
                joints_rad=joints_rad,
                joints_raw_mdeg=joint_raw,
                joints_host_recv_ns=time.time_ns(),
                joints_source_timestamp_ns=None,
                joints_clock_source="none:piper_can_protocol_has_no_device_timestamp",
                gripper_feedback_raw=gripper_raw,
                gripper_effort_raw=self._last_gripper["effort"],
                gripper_status_code=status_code,
                gripper_host_recv_ns=self._last_gripper["host_ns"],
                gripper_valid=gripper_valid,
                gripper_width_mm=self.gripper_cal.width_mm(gripper_raw),
                gripper_calibration_id=self.gripper_cal.calibration_id if self.gripper_cal.valid else None,
                ee_pose=ee_pose,
                ee_pose_frame=self.fk.ee_frame,
                ee_pose_source="feedback_joint_fk",
                driver_end_pose_raw=self._last_driver_end_pose,
                arm_status=self._last_arm_status,
                enable=self._last_enable,
                feedback_hz=self._last_feedback_hz,
            )
            # 关节未变化时沿用上一次的姿态与四元数（不重复插值），保持状态自洽
            if not joint_changed and self._buffer:
                prev = self._buffer[-1]
                state.joints_rad = list(prev.joints_rad)
                state.joints_raw_mdeg = list(prev.joints_raw_mdeg)
                state.joints_host_recv_ns = prev.joints_host_recv_ns
                state.ee_pose = list(prev.ee_pose) if prev.ee_pose else None
            self._buffer.append(state)
            self.counters["state_updates"] += 1
            self.counters["quat_sign_flips"] = self._quat.flips

        if self.on_state is not None:
            self.on_state(state)

    def _refresh_slow_status(self, piper) -> None:
        """低频刷新驱动状态/使能/自带末端反馈，用于驱动记录。"""
        try:
            st = piper.GetArmStatus()
            s = st.arm_status
            self._last_arm_status = {
                "ctrl_mode": int(s.ctrl_mode),
                "arm_status": int(s.arm_status),
                "mode_feed": int(s.mode_feed),
                "teach_status": int(s.teach_status),
                "motion_status": int(s.motion_status),
                "trajectory_num": int(s.trajectory_num),
                "err_code": int(getattr(s, "err_code", 0)),
                "host_recv_ns": time.time_ns(),
                "source_timestamp_ns": None,
            }
        except Exception:
            pass
        try:
            self._last_enable = [bool(v) for v in piper.GetArmEnableStatus()]
        except Exception:
            pass
        try:
            ep = piper.GetArmEndPoseMsgs()
            p = ep.end_pose
            self._last_driver_end_pose = {
                "source": "driver_end_pose_feedback_can_0x2A5_type",
                "X_axis_raw": int(p.X_axis),
                "Y_axis_raw": int(p.Y_axis),
                "Z_axis_raw": int(p.Z_axis),
                "RX_axis_raw": int(p.RX_axis),
                "RY_axis_raw": int(p.RY_axis),
                "RZ_axis_raw": int(p.RZ_axis),
                "unit": {"xyz": "0.001 mm", "rpy": "0.001 deg"},
                "note": "驱动自带反馈，仅作独立参考，不用于数据集 ee_pose",
                "host_recv_ns": time.time_ns(),
            }
        except Exception:
            pass

    # ------------------------------------------------------------------ 查询
    def latest(self) -> Optional[RobotState]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def snapshot(self) -> List[RobotState]:
        with self._lock:
            return list(self._buffer)

    def clear_buffer(self) -> None:
        with self._lock:
            self._buffer.clear()

    def export_driver_record(self) -> Dict[str, Any]:
        """可复现运行环境所需的驱动记录。"""
        firmware = self._safe(lambda: self._piper.GetPiperFirmwareVersion()) if self._piper else None
        limits = None
        if self._piper is not None:
            limits = self._safe(lambda: self._piper.GetAllMotorAngleLimitMaxSpd())
        return {
            "backend": "piper_sdk_direct_can",
            "sdk_version": self.versions.get("sdk_version"),
            "interface_version": self.versions.get("interface_version"),
            "protocol_version": self.versions.get("protocol_version"),
            "firmware_version": firmware,
            "firmware_read_note": None
            if isinstance(firmware, str)
            else "固件版本未能读取（GetPiperFirmwareVersion 返回非字符串）",
            "can": {
                "interface": self.can_interface,
                "bitrate": 1000000,
                "activate_note": "沿用现有 can0 配置，本程序不激活/不修改 CAN",
            },
            "startup_params": {
                "dh_is_offset": self.fk.dh_is_offset,
                "queries_on_connect": self.queries_on_connect,
                "sdk_joint_limit": False,
                "sdk_gripper_limit": False,
                "read_only": True,
                "motion_commands_sent": False,
            },
            "kinematics": self.fk.describe(),
            "counters": dict(self.counters),
            "feedback_rate_hz": self._last_feedback_hz,
            "command_interface": {
                "available": False,
                "note": "当前采集程序不接入下发命令记录，命令数据不由状态反推生成",
            },
            "motor_angle_limit_feedback": self._safe(lambda: str(limits)),
            "open_error": self.open_error,
        }

    def diagnostics(self) -> Dict[str, Any]:
        c = self.counters
        first = c["first_joint_host_ns"]
        last = c["last_joint_host_ns"]
        span = (int(last) - int(first)) / 1e9 if first and last else None
        return {
            "joint_frames": c["joint_frames"],
            "gripper_frames": c["gripper_frames"],
            "state_updates": c["state_updates"],
            "gripper_only_updates": c["gripper_only_updates"],
            "measured_joint_rate_hz": (c["joint_frames"] / span) if span else None,
            "measured_gripper_rate_hz": (c["gripper_frames"] / span) if span else None,
            "read_errors": c["read_errors"],
            "feedback_timeout_events": c["dropped_gaps"],
            "max_feedback_gap_ms": c["max_gap_ms"],
            "quaternion_sign_flips": c["quat_sign_flips"],
            "span_s": span,
            "connected": self.connected,
            "open_error": self.open_error,
        }