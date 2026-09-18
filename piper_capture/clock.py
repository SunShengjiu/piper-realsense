"""时钟域、设备时钟到主机时间的映射，以及样本级时间匹配。

明确区分：
  - 源时间戳（source timestamp）：由设备自己产生。RealSense 帧时间戳属于
    hardware_clock / system_time 等时钟域；PiPER 的 CAN 反馈协议**没有**
    设备侧时间戳，因此关节反馈只有主机接收时间。
  - 主机接收时间（host receive time）：本程序收到数据时的 time.time_ns()。

RGB、Depth、关节反馈三者都在主机时间轴上对齐，属于**软件时间匹配**，
不等于硬件同步。所有输出都带 matching_method 与各源时间差。
"""
from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np

from .kinematics import ForwardKinematics
from .robot import RobotState
from .schema import JOINT_NAMES


class DeviceClockMapper:
    """把设备时钟（ms）映射到主机时钟（ns）。

    估计量 offset = host_recv_ns - device_ts_ms * 1e6，用滑动中位数抑制抖动。
    这是**在线估计**，用于排查时钟漂移；写入数据集的仍是原始设备时间戳与
    原始主机接收时间，映射值只作为附加字段。
    """

    def __init__(self, name: str, domain: str, window: int = 240) -> None:
        self.name = name
        self.domain = domain
        self._offsets: Deque[float] = deque(maxlen=window)
        self.method = "sliding_median(host_receive_ns - device_timestamp_ms*1e6)"
        self.samples = 0

    def observe(self, device_ts_ms: float, host_recv_ns: int) -> None:
        self._offsets.append(float(host_recv_ns) - float(device_ts_ms) * 1e6)
        self.samples += 1

    @property
    def offset_ns(self) -> Optional[float]:
        return statistics.median(self._offsets) if self._offsets else None

    @property
    def jitter_ms(self) -> Optional[float]:
        if len(self._offsets) < 3:
            return None
        return statistics.pstdev(self._offsets) / 1e6

    def to_host_ns(self, device_ts_ms: float) -> Optional[float]:
        off = self.offset_ns
        return None if off is None else float(device_ts_ms) * 1e6 + off

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_clock_domain": self.domain,
            "mapping_method": self.method,
            "offset_ns": self.offset_ns,
            "jitter_ms": self.jitter_ms,
            "observations": self.samples,
            "note": "软件时钟映射，不是硬件同步",
        }


@dataclass
class MatchedRobot:
    valid: bool
    method: str
    dt_ms: Optional[float]
    source_state_ids: List[int]
    joints_rad: Optional[List[float]]
    ee_pose: Optional[List[float]]
    gripper_feedback_raw: Optional[int]
    gripper_width_mm: Optional[float]
    gripper_valid: bool
    gripper_calibration_id: Optional[str]
    gripper_dt_ms: Optional[float]
    gripper_source_state_id: Optional[int]
    stale: bool
    invalid_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_joints_match_method": self.method,
            "robot_joints_dt_ms": self.dt_ms,
            "robot_state_ids_used": self.source_state_ids,
            "joint_names": list(JOINT_NAMES),
            "joint_positions_rad": self.joints_rad,
            "ee_pose": self.ee_pose,
            "gripper_feedback_raw": self.gripper_feedback_raw,
            "gripper_width_mm": self.gripper_width_mm,
            "gripper_valid": self.gripper_valid,
            "gripper_calibration_id": self.gripper_calibration_id,
            "gripper_dt_ms": self.gripper_dt_ms,
            "gripper_source_state_id": self.gripper_source_state_id,
            "robot_state_stale": self.stale,
        }


class RobotStateMatcher:
    """把图像帧的主机时间匹配到最近的机械臂反馈状态。

    - mode="nearest"：取 |dt| 最小的反馈帧，超过 robot_tolerance_ms 判为无效。
    - mode="linear"：在两侧反馈帧之间对关节角线性插值（同时重算 FK，
      保证 EE 位姿与所用关节角自洽），并记录插值方法与所用原始状态 id。

    无论哪种模式，样本都会写明匹配方法、时间差和是否过期；过期状态不会被
    静默复用。
    """

    def __init__(
        self,
        fk: ForwardKinematics,
        *,
        tolerance_ms: float = 33.0,
        max_state_age_ms: float = 50.0,
        mode: str = "nearest",
    ) -> None:
        self.fk = fk
        self.tolerance_ms = float(tolerance_ms)
        self.max_state_age_ms = float(max_state_age_ms)
        if mode not in ("nearest", "linear"):
            raise ValueError("robot_match_mode 只支持 nearest / linear")
        self.mode = mode
        self.counters: Dict[str, int] = {
            "matched": 0,
            "out_of_tolerance": 0,
            "stale": 0,
            "interpolated": 0,
            "no_state": 0,
        }

    def match(self, states: Sequence[RobotState], target_host_ns: int) -> MatchedRobot:
        if not states:
            self.counters["no_state"] += 1
            return MatchedRobot(
                False, f"{self.mode}_no_robot_state", None, [], None, None, None, None, False, None, None, None, False,
                "采集时刻没有任何机械臂反馈状态",
            )
        arr = np.array([s.joints_host_recv_ns for s in states], dtype=np.int64)
        target = int(target_host_ns)
        if self.mode == "nearest":
            idx = int(np.argmin(np.abs(arr - target)))
            st = states[idx]
            dt_ms = (int(st.joints_host_recv_ns) - target) / 1e6
            aged = abs(dt_ms) > self.max_state_age_ms
            in_tol = abs(dt_ms) <= self.tolerance_ms
            gripper = self._gripper_of(states, target)
            if not in_tol:
                self.counters["out_of_tolerance"] += 1
            if aged:
                self.counters["stale"] += 1
            self.counters["matched"] += 1
            return MatchedRobot(
                valid=in_tol,
                method="nearest_feedback_frame_by_host_recv_time",
                dt_ms=dt_ms,
                source_state_ids=[st.state_id],
                joints_rad=list(st.joints_rad),
                ee_pose=list(st.ee_pose) if st.ee_pose else None,
                gripper_feedback_raw=gripper[0].gripper_feedback_raw if gripper[0] else None,
                gripper_width_mm=gripper[0].gripper_width_mm if gripper[0] else None,
                gripper_valid=gripper[0].gripper_valid if gripper[0] else False,
                gripper_calibration_id=gripper[0].gripper_calibration_id if gripper[0] else None,
                gripper_dt_ms=gripper[1],
                gripper_source_state_id=gripper[0].state_id if gripper[0] else None,
                stale=aged,
                invalid_reason=None if in_tol else f"最近关节反馈时间差 {dt_ms:.2f} ms 超过容差 {self.tolerance_ms} ms",
            )

        # linear interpolation
        order = np.argsort(arr)
        sorted_states = [states[int(i)] for i in order]
        sorted_ts = arr[order]
        if target <= int(sorted_ts[0]):
            st = sorted_states[0]
            dt_ms = (int(st.joints_host_recv_ns) - target) / 1e6
            return self._nearest_fallback(st, dt_ms, states, target, "target早于最早反馈帧")
        if target >= int(sorted_ts[-1]):
            st = sorted_states[-1]
            dt_ms = (int(st.joints_host_recv_ns) - target) / 1e6
            return self._nearest_fallback(st, dt_ms, states, target, "target晚于最新反馈帧")
        hi = int(np.searchsorted(sorted_ts, target))
        lo = hi - 1
        s0, s1 = sorted_states[lo], sorted_states[hi]
        t0, t1 = int(s0.joints_host_recv_ns), int(s1.joints_host_recv_ns)
        span = t1 - t0
        if span <= 0 or span / 1e6 > self.max_state_age_ms * 2:
            dt0 = (t0 - target) / 1e6
            dt1 = (t1 - target) / 1e6
            st = s0 if abs(dt0) <= abs(dt1) else s1
            dt_ms = dt0 if st is s0 else dt1
            return self._nearest_fallback(st, dt_ms, states, target, "两侧反馈间隔过大，退化为最近帧")
        alpha = (target - t0) / span
        joints = [float(a) + alpha * (float(b) - float(a)) for a, b in zip(s0.joints_rad, s1.joints_rad)]
        ee = self.fk.fk_base_ee(joints)
        from .schema import Transform

        ee_pose = Transform.to_pose(ee)
        gripper = self._gripper_of(states, target)
        worst = max(abs((t0 - target) / 1e6), abs((t1 - target) / 1e6))
        in_tol = worst <= self.tolerance_ms
        aged = worst > self.max_state_age_ms
        if not in_tol:
            self.counters["out_of_tolerance"] += 1
        if aged:
            self.counters["stale"] += 1
        self.counters["interpolated"] += 1
        self.counters["matched"] += 1
        return MatchedRobot(
            valid=in_tol,
            method="linear_interpolation_between_two_feedback_frames",
            dt_ms=(t0 - target) / 1e6,
            source_state_ids=[s0.state_id, s1.state_id],
            joints_rad=joints,
            ee_pose=ee_pose,
            gripper_feedback_raw=gripper[0].gripper_feedback_raw if gripper[0] else None,
            gripper_width_mm=gripper[0].gripper_width_mm if gripper[0] else None,
            gripper_valid=gripper[0].gripper_valid if gripper[0] else False,
            gripper_calibration_id=gripper[0].gripper_calibration_id if gripper[0] else None,
            gripper_dt_ms=gripper[1],
            gripper_source_state_id=gripper[0].state_id if gripper[0] else None,
            stale=aged,
            invalid_reason=None
            if in_tol
            else f"插值两端反馈距离 {worst:.2f} ms 超过容差 {self.tolerance_ms} ms",
        )

    def _nearest_fallback(
        self,
        st: RobotState,
        dt_ms: float,
        states: Sequence[RobotState],
        target: int,
        reason: str,
    ) -> MatchedRobot:
        gripper = self._gripper_of(states, target)
        in_tol = abs(dt_ms) <= self.tolerance_ms
        aged = abs(dt_ms) > self.max_state_age_ms
        if not in_tol:
            self.counters["out_of_tolerance"] += 1
        if aged:
            self.counters["stale"] += 1
        self.counters["matched"] += 1
        return MatchedRobot(
            valid=in_tol,
            method=f"nearest_feedback_frame_by_host_recv_time ({reason})",
            dt_ms=dt_ms,
            source_state_ids=[st.state_id],
            joints_rad=list(st.joints_rad),
            ee_pose=list(st.ee_pose) if st.ee_pose else None,
            gripper_feedback_raw=gripper[0].gripper_feedback_raw if gripper[0] else None,
            gripper_width_mm=gripper[0].gripper_width_mm if gripper[0] else None,
            gripper_valid=gripper[0].gripper_valid if gripper[0] else False,
            gripper_calibration_id=gripper[0].gripper_calibration_id if gripper[0] else None,
            gripper_dt_ms=gripper[1],
            gripper_source_state_id=gripper[0].state_id if gripper[0] else None,
            stale=aged,
            invalid_reason=None if in_tol else f"关节反馈时间差 {dt_ms:.2f} ms 超过容差 {self.tolerance_ms} ms",
        )

    @staticmethod
    def _gripper_of(states: Sequence[RobotState], target: int):
        """夹爪取时间上最近的一次夹爪反馈（不做插值），并给出它自己的时间差。"""
        best = None
        best_dt = None
        for st in states:
            if st.gripper_host_recv_ns is None:
                continue
            dt = (int(st.gripper_host_recv_ns) - int(target)) / 1e6
            if best_dt is None or abs(dt) < abs(best_dt):
                best, best_dt = st, dt
        return best, best_dt

    def describe(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "tolerance_ms": self.tolerance_ms,
            "max_robot_state_age_ms": self.max_state_age_ms,
            "notes": [
                "软件时间匹配，不是硬件同步",
                "过期反馈不会被静默复用：超容差样本标记为无效并计数",
                "linear 模式会记录所用原始状态 id，EE 位姿由插值后的关节角重算",
            ],
            "counters": dict(self.counters),
        }


def host_now_ns() -> int:
    return time.time_ns()