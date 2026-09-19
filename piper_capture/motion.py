"""Explicit PiPER motion helpers.

The capture path remains read-only unless the caller opts into the
``arm-button-return-zero`` workflow.  This module contains the small amount of
CAN motion code needed by that opt-in path and by the standalone ``robot
go-zero`` command.  ``zero`` here means *move to the configured joint pose*;
it does not call ``JointConfig(..., set_zero=0xAE)`` and therefore never
rewrites the motor encoder zero point.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from threading import Event
from typing import Any, Dict, List, Optional, Sequence


STANDBY_CTRL_MODE = 0x00
CAN_CTRL_MODE = 0x01
TEACHING_CTRL_MODES = (0x02, 0x06)

# Feedback values from CAN ID 0x2A1.  The J6 teaching button changes the
# recording state, but it does not itself leave ``ctrl_mode == 0x02``.  A
# separate reset-to-standby command is required before CAN motion can start.
TEACHING_RECORD_ARM_STATUS = 0x0B
TEACHING_EXECUTION_ARM_STATUS = 0x0C
TEACHING_PAUSE_ARM_STATUS = 0x0D
TEACHING_START_STATUS = 0x01
TEACHING_STOP_STATUS = 0x02

# ``ResetPiper`` briefly removes motor power.  During that handoff the
# feedback frame can report joint communication abnormal (0x05) even though
# the CAN bus is healthy and the next frames recover to normal.  It is safe to
# wait for this bounded transition, but never to send a joint target while it
# is asserted.
JOINT_COMMUNICATION_ARM_STATUS = 0x05
RECOVERABLE_ARM_STATUSES = frozenset(
    (JOINT_COMMUNICATION_ARM_STATUS, 0x0B, 0x0C, 0x0D)
)

# PiPER SDK feedback/command units are 0.001 degree.
_DEG_TO_MDEG = 1000.0


class MotionError(RuntimeError):
    """A motion precondition or communication failure."""


@dataclass
class MotionResult:
    """Result of a bounded joint-position move."""

    status: str
    target_deg: List[float]
    final_deg: Optional[List[float]]
    elapsed_s: float
    command_count: int
    max_error_deg: Optional[float] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "target_joint_positions_deg": list(self.target_deg),
            "final_joint_positions_deg": list(self.final_deg) if self.final_deg is not None else None,
            "elapsed_s": float(self.elapsed_s),
            "command_count": int(self.command_count),
            "max_error_deg": self.max_error_deg,
            "reason": self.reason,
            "zero_semantics": "move_to_pose; motor encoder zero is unchanged",
        }


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def read_arm_status(piper: Any) -> Dict[str, Optional[int]]:
    """Return the status fields used as motion safety preconditions."""

    msg = piper.GetArmStatus()
    status = getattr(msg, "arm_status", msg)
    stamp = getattr(msg, "time_stamp", None)
    return {
        "status_timestamp": stamp,
        "ctrl_mode": _int_or_none(getattr(status, "ctrl_mode", None)),
        "arm_status": _int_or_none(getattr(status, "arm_status", None)),
        "teach_status": _int_or_none(getattr(status, "teach_status", None)),
        "motion_status": _int_or_none(getattr(status, "motion_status", None)),
        "err_code": _int_or_none(getattr(status, "err_code", None)),
    }


def read_joint_positions_deg(piper: Any) -> Optional[List[float]]:
    """Read six joint positions in degrees from the SDK feedback message."""

    msg = piper.GetArmJointMsgs()
    stamp = getattr(msg, "time_stamp", None)
    # piper_sdk uses zero until a real feedback frame has arrived.
    if stamp in (None, 0):
        return None
    state = getattr(msg, "joint_state", msg)
    values: List[float] = []
    for index in range(1, 7):
        value = _int_or_none(getattr(state, f"joint_{index}", None))
        if value is None:
            return None
        values.append(float(value) / _DEG_TO_MDEG)
    return values


def _validate_pose(pose_deg: Sequence[float]) -> List[float]:
    if len(pose_deg) != 6:
        raise ValueError("回零目标必须恰好包含 6 个关节角（单位：度）")
    pose = [float(v) for v in pose_deg]
    if not all(math.isfinite(v) for v in pose):
        raise ValueError("回零目标不能包含 NaN 或无穷大")
    # Limits documented by piper_sdk.JointCtrl.  The default all-zero pose is
    # valid for every joint; validating here avoids sending an unsafe target
    # when a custom pose is supplied in a config file.
    limits = ((-150.0, 150.0), (0.0, 180.0), (-170.0, 0.0),
              (-100.0, 100.0), (-70.0, 70.0), (-120.0, 120.0))
    for index, (value, (lo, hi)) in enumerate(zip(pose, limits), start=1):
        if value < lo or value > hi:
            raise ValueError(f"回零目标 joint{index}={value:g}° 超出 SDK 限位 [{lo:g}, {hi:g}]°")
    return pose


def validate_target_pose(pose_deg: Sequence[float]) -> List[float]:
    """Validate and normalize a six-joint target before starting capture."""

    return _validate_pose(pose_deg)


class PiperMotionController:
    """Bounded CAN joint motion on an already connected SDK interface."""

    def __init__(
        self,
        piper: Any,
        *,
        target_deg: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        speed_percent: int = 20,
        command_hz: float = 100.0,
        timeout_s: float = 20.0,
        enable_timeout_s: float = 3.0,
        tolerance_deg: float = 1.0,
        settle_s: float = 0.5,
        reset_teaching: bool = False,
        reset_timeout_s: float = 3.0,
    ) -> None:
        if piper is None:
            raise MotionError("PiPER CAN 接口尚未连接")
        self.piper = piper
        self.target_deg = _validate_pose(target_deg)
        self.target_raw = [int(round(value * _DEG_TO_MDEG)) for value in self.target_deg]
        self.speed_percent = int(speed_percent)
        if not 1 <= self.speed_percent <= 100:
            raise ValueError("回零速度百分比必须在 1..100 之间")
        self.command_hz = float(command_hz)
        if not math.isfinite(self.command_hz) or self.command_hz <= 0:
            raise ValueError("回零 command_hz 必须为正数")
        self.timeout_s = float(timeout_s)
        self.enable_timeout_s = float(enable_timeout_s)
        self.tolerance_deg = float(tolerance_deg)
        self.settle_s = float(settle_s)
        self.reset_teaching = bool(reset_teaching)
        self.reset_timeout_s = float(reset_timeout_s)
        if (
            self.timeout_s <= 0
            or self.enable_timeout_s <= 0
            or self.tolerance_deg <= 0
            or self.settle_s < 0
            or self.reset_timeout_s <= 0
        ):
            raise ValueError("回零超时、误差和稳定时间参数必须有效")

    @staticmethod
    def _recording_status(status: Dict[str, Optional[int]]) -> bool:
        """Whether feedback says that a drag-teach recording is active."""

        return bool(
            status.get("teach_status") == TEACHING_START_STATUS
            or status.get("arm_status") == TEACHING_RECORD_ARM_STATUS
        )

    @staticmethod
    def _real_status(status: Dict[str, Optional[int]]) -> bool:
        """Ignore the SDK's all-zero startup placeholder frame.

        Some older SDK builds and test doubles do not expose ``time_stamp``;
        ``None`` is therefore treated as a usable status, while the SDK's
        explicit numeric zero remains the startup placeholder.
        """

        return status.get("status_timestamp") != 0

    def wait_for_teach_button(
        self,
        *,
        timeout_s: float = 60.0,
        poll_hz: float = 20.0,
        stop_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        """Wait for the physical teaching button to stop a recording.

        On PiPER the button's single click ends a recording while feedback
        remains in teaching mode.  We therefore watch the teaching/arm status
        edge instead of requiring the impossible ``0x02 -> 0x01`` transition.
        A direct teaching-to-CAN transition is still accepted for installations
        that use the Windows pendant to change modes.
        """

        deadline = time.monotonic() + float(timeout_s)
        period = 1.0 / max(1.0, float(poll_hz))
        previous: Optional[Dict[str, Optional[int]]] = None
        initial: Optional[int] = None
        seen_teaching = False
        seen_recording = False
        while time.monotonic() < deadline:
            if stop_event is not None and stop_event.is_set():
                raise MotionError("等待示教按钮时收到停止请求")
            status = read_arm_status(self.piper)
            if not self._real_status(status):
                time.sleep(period)
                continue
            mode = status.get("ctrl_mode")
            if initial is None:
                initial = mode
            if mode in TEACHING_CTRL_MODES:
                seen_teaching = True
            recording = self._recording_status(status)
            if recording:
                seen_recording = True

            direct_can = bool(
                seen_teaching
                and mode == CAN_CTRL_MODE
                and previous is not None
                and previous.get("ctrl_mode") not in (None, CAN_CTRL_MODE)
            )
            stopped_recording = bool(
                previous is not None
                and (
                    # The stop command can be reported for only one feedback
                    # frame; do not require that a preceding 0x0B frame was
                    # sampled by this process.
                    (
                        mode in TEACHING_CTRL_MODES
                        and status.get("teach_status") == TEACHING_STOP_STATUS
                        and previous.get("teach_status") != TEACHING_STOP_STATUS
                    )
                    or (
                        seen_recording
                        and self._recording_status(previous)
                        and not recording
                        and status.get("arm_status")
                        not in (TEACHING_EXECUTION_ARM_STATUS, TEACHING_PAUSE_ARM_STATUS)
                    )
                )
            )
            if stopped_recording or direct_can:
                return {
                    "status": "triggered",
                    "trigger": "teach_record_stop" if stopped_recording else "teach_to_can",
                    "initial_ctrl_mode": initial,
                    "final_ctrl_mode": mode,
                    "final_arm_status": status.get("arm_status"),
                    "final_teach_status": status.get("teach_status"),
                    "seen_teaching": seen_teaching,
                    "seen_recording": seen_recording,
                }
            previous = status
            time.sleep(period)
        raise MotionError(
            f"{timeout_s:.1f}s 内没有检测到示教按钮结束记录；"
            "单击 J5/J6 之间的示教按钮结束记录，或用上位机切换到 CAN"
        )

    def wait_for_teach_to_can(
        self,
        *,
        timeout_s: float = 60.0,
        poll_hz: float = 20.0,
        stop_event: Optional[Event] = None,
    ) -> Dict[str, Any]:
        """Wait for the physical teaching→CAN mode transition.

        A plain ``ctrl_mode == CAN`` is deliberately not enough: the arm may
        already be in CAN mode when the listener connects.  Requiring a
        teaching mode first makes the physical button an unambiguous trigger.
        """

        deadline = time.monotonic() + float(timeout_s)
        previous: Optional[int] = None
        seen_teaching = False
        initial: Optional[int] = None
        period = 1.0 / max(1.0, float(poll_hz))
        while time.monotonic() < deadline:
            if stop_event is not None and stop_event.is_set():
                raise MotionError("等待示教→CAN切换时收到停止请求")
            status = read_arm_status(self.piper)
            # piper_sdk initializes status fields to zero before the first
            # 0x2A1 feedback frame. Do not mistake that startup placeholder for
            # a real standby-mode report.
            if not self._real_status(status):
                time.sleep(period)
                continue
            mode = status.get("ctrl_mode")
            if initial is None:
                initial = mode
            if mode in TEACHING_CTRL_MODES:
                seen_teaching = True
            if seen_teaching and mode == CAN_CTRL_MODE and previous not in (None, CAN_CTRL_MODE):
                return {
                    "status": "triggered",
                    "initial_ctrl_mode": initial,
                    "final_ctrl_mode": mode,
                    "seen_teaching": True,
                }
            previous = mode
            time.sleep(period)
        raise MotionError(
            f"{timeout_s:.1f}s 内没有检测到示教模式→CAN模式切换；"
            "请先进入示教模式，再按机械臂按钮切换到 CAN"
        )

    def _reset_teaching_to_standby(self) -> None:
        """Leave drag-teach mode using the SDK reset command.

        The official PiPER manual requires ``piper_ctrl_reset.py`` before a
        second command program can select CAN mode.  ResetPiper clears the
        teaching state and may release motor power briefly, so this method is
        only called when ``reset_teaching=True`` was explicitly selected by a
        motion workflow.
        """

        reset = getattr(self.piper, "ResetPiper", None)
        if reset is None:
            reset = lambda: self.piper.MotionCtrl_1(0x02, 0x00, 0x00)
        reset()
        deadline = time.monotonic() + self.reset_timeout_s
        last_err: Optional[int] = None
        last_mode: Optional[int] = None
        last_arm_status: Optional[int] = None
        while time.monotonic() < deadline:
            status = read_arm_status(self.piper)
            if self._real_status(status):
                last_mode = status.get("ctrl_mode")
                err_code = status.get("err_code")
                last_arm_status = status.get("arm_status")
                if err_code not in (None, 0):
                    # ResetPiper clears the teaching/limit flags asynchronously.
                    # The first 0x2A1 frame can still carry the old 0x3f value;
                    # do not select CAN until a later frame confirms it is gone.
                    last_err = err_code
                if last_mode not in TEACHING_CTRL_MODES and err_code in (None, 0):
                    if last_arm_status == 0:
                        return
                    if last_arm_status in RECOVERABLE_ARM_STATUSES:
                        time.sleep(0.05)
                        continue
                    raise MotionError(
                        f"重置示教模式后机械臂状态异常（arm_status={last_arm_status}）；"
                        "拒绝切换 CAN"
                    )
            time.sleep(0.05)
        if last_err not in (None, 0):
            raise MotionError(
                f"重置示教模式后故障码仍非零（err_code={last_err}，ctrl_mode={last_mode}）；"
                "请确认机械臂没有真实关节限位或通信故障后再重试"
            )
        if last_arm_status not in (None, 0):
            raise MotionError(
                f"重置示教模式后关节通信仍未恢复（arm_status={last_arm_status}）；"
                "请检查机械臂电源、急停和 CAN 接线后再重试"
            )
        raise MotionError(
            f"{self.reset_timeout_s:.1f}s 内无法退出示教模式（ctrl_mode 仍为示教状态）；"
            "请先用上位机重置到待机，再选择 CAN 模式"
        )

    def _enter_can_mode(self) -> None:
        """Select CAN/MOVE-J after the arm is in standby."""

        deadline = time.monotonic() + self.reset_timeout_s
        last_err: Optional[int] = None
        last_mode: Optional[int] = None
        last_arm_status: Optional[int] = None
        while time.monotonic() < deadline:
            status = read_arm_status(self.piper)
            if not self._real_status(status):
                time.sleep(0.05)
                continue
            last_mode = status.get("ctrl_mode")
            err_code = status.get("err_code")
            last_arm_status = status.get("arm_status")
            if err_code not in (None, 0):
                # A ResetPiper -> standby transition may expose the previous
                # angle-limit bitmap for one or more feedback frames.  Keep
                # polling inside the bounded handshake, but never send a CAN
                # mode or joint target while a fault remains asserted.
                last_err = err_code
                time.sleep(0.05)
                continue
            if last_mode == CAN_CTRL_MODE:
                if last_arm_status == 0:
                    return
                if last_arm_status in RECOVERABLE_ARM_STATUSES:
                    time.sleep(0.05)
                    continue
                raise MotionError(
                    f"切换 CAN 后机械臂状态异常（arm_status={last_arm_status}）；"
                    "拒绝发送关节目标"
                )
            if last_mode in TEACHING_CTRL_MODES:
                raise MotionError("重置后机械臂仍处于示教模式，拒绝发送关节目标")
            if last_arm_status not in (None, 0):
                if last_arm_status in RECOVERABLE_ARM_STATUSES:
                    time.sleep(0.05)
                    continue
                raise MotionError(
                    f"切换 CAN 前机械臂状态异常（arm_status={last_arm_status}）；"
                    "拒绝发送关节目标"
                )
            # ModeCtrl is the documented post-reset command.  MotionCtrl_2 is
            # repeated as well because some firmware versions acknowledge the
            # latter only after the first standby feedback frame.
            self.piper.ModeCtrl(CAN_CTRL_MODE, 0x01, self.speed_percent, 0x00)
            self.piper.MotionCtrl_2(CAN_CTRL_MODE, 0x01, self.speed_percent, 0x00)
            time.sleep(0.05)
        if last_err not in (None, 0):
            raise MotionError(
                f"切换 CAN 前故障码仍非零（err_code={last_err}，ctrl_mode={last_mode}）；"
                "请确认机械臂没有真实关节限位或通信故障后再重试"
            )
        if last_arm_status not in (None, 0):
            raise MotionError(
                f"切换 CAN 后关节通信仍未恢复（arm_status={last_arm_status}）；"
                "请检查机械臂电源、急停和 CAN 接线后再重试"
            )
        raise MotionError(f"{self.reset_timeout_s:.1f}s 内无法切换到 CAN 控制模式")

    def _ensure_can_and_normal(self) -> Dict[str, Optional[int]]:
        # Motor enabling and the first position frame can briefly expose the
        # same joint-communication status (0x05) as ResetPiper.  Poll that
        # bounded transition here too; otherwise the next loop iteration can
        # abort immediately after a successful CAN handshake.
        deadline = time.monotonic() + self.reset_timeout_s
        last: Optional[Dict[str, Optional[int]]] = None
        while time.monotonic() < deadline:
            status = read_arm_status(self.piper)
            last = status
            if not self._real_status(status):
                time.sleep(0.05)
                continue
            if status.get("ctrl_mode") != CAN_CTRL_MODE:
                if self.reset_teaching:
                    mode = status.get("ctrl_mode")
                    if mode in TEACHING_CTRL_MODES:
                        self._reset_teaching_to_standby()
                    # A motor-enable transition can fall back to standby for
                    # one feedback frame.  Re-select CAN and re-enable before
                    # allowing the caller to publish the next JointCtrl.
                    self._enter_can_mode()
                    self._enable()
                    continue
                raise MotionError(
                    f"回零前机械臂不在 CAN 控制模式（ctrl_mode={status.get('ctrl_mode')}）；"
                    "请先重置到待机并切换到 CAN 模式"
                )
            err_code = status.get("err_code")
            arm_status = status.get("arm_status")
            if err_code not in (None, 0):
                if self.reset_teaching:
                    time.sleep(0.05)
                    continue
                raise MotionError(f"回零前机械臂故障码非零（err_code={err_code}）")
            if arm_status == 0:
                try:
                    enabled = [bool(v) for v in self.piper.GetArmEnableStatus()]
                except Exception:
                    enabled = None
                if enabled is not None and len(enabled) >= 6 and all(enabled[:6]):
                    return status
                if self.reset_teaching:
                    # The controller can remain in CAN while a motor-enable
                    # bit drops.  Do not publish another JointCtrl in that
                    # state; repeat the official enable handshake first.
                    self._enable()
                    continue
                raise MotionError("回零前六个关节未全部使能，拒绝发送关节目标")
            if arm_status in RECOVERABLE_ARM_STATUSES:
                time.sleep(0.05)
                continue
            raise MotionError(f"回零前机械臂状态异常（arm_status={arm_status}）")
        if last is not None and last.get("err_code") not in (None, 0):
            raise MotionError(f"回零前机械臂故障码仍非零（err_code={last.get('err_code')}）")
        if last is not None and last.get("arm_status") not in (None, 0):
            raise MotionError(f"回零前关节通信仍未恢复（arm_status={last.get('arm_status')}）")
        raise MotionError("回零前未收到有效的 CAN 状态反馈")

    def _wait_until_ready(self, timeout_s: float = 2.0) -> None:
        """Wait for normal CAN feedback, optionally exiting teaching mode."""

        # Exiting teaching mode involves a reset and a second mode handshake;
        # include those bounded phases in the readiness budget.
        budget = float(timeout_s) + (2.0 * self.reset_timeout_s if self.reset_teaching else 0.0)
        deadline = time.monotonic() + budget
        transitional = RECOVERABLE_ARM_STATUSES
        last_err: Optional[int] = None
        last_arm_status: Optional[int] = None
        while time.monotonic() < deadline:
            status = read_arm_status(self.piper)
            if not self._real_status(status):
                time.sleep(0.05)
                continue
            if status.get("ctrl_mode") != CAN_CTRL_MODE:
                if self.reset_teaching:
                    if status.get("ctrl_mode") in TEACHING_CTRL_MODES:
                        self._reset_teaching_to_standby()
                    # Explicit motion workflows may start in either teaching
                    # mode or standby.  ModeCtrl is safe here because no
                    # target is sent until CAN feedback is confirmed below.
                    self._enter_can_mode()
                    continue
                raise MotionError(
                    f"回零前机械臂不在 CAN 控制模式（ctrl_mode={status.get('ctrl_mode')}）；"
                    "请先重置到待机并切换到 CAN 模式"
                )
            arm_status = status.get("arm_status")
            last_arm_status = arm_status
            if status.get("err_code") not in (None, 0):
                last_err = status.get("err_code")
                # Wait for a transient reset/limit bitmap to clear.  The
                # deadline keeps a persistent hardware fault fail-safe.
                time.sleep(0.05)
                continue
            if arm_status == 0:
                return
            if arm_status is None:
                time.sleep(0.05)
                continue
            if arm_status not in transitional:
                raise MotionError(f"回零前机械臂状态异常（arm_status={arm_status}）")
            time.sleep(0.05)
        if last_err not in (None, 0):
            raise MotionError(f"回零前机械臂故障码仍非零（err_code={last_err}）")
        if last_arm_status not in (None, 0):
            raise MotionError(f"回零前关节通信仍未恢复（arm_status={last_arm_status}）")
        raise MotionError(f"{timeout_s:.1f}s 内机械臂未从示教过渡状态恢复正常")

    def _enable(self) -> None:
        deadline = time.monotonic() + self.enable_timeout_s
        last: Optional[List[bool]] = None
        last_status: Optional[Dict[str, Optional[int]]] = None
        while time.monotonic() < deadline:
            status = read_arm_status(self.piper)
            last_status = status
            if not self._real_status(status):
                time.sleep(0.05)
                continue
            mode = status.get("ctrl_mode")
            if mode != CAN_CTRL_MODE:
                if self.reset_teaching:
                    if mode in TEACHING_CTRL_MODES:
                        self._reset_teaching_to_standby()
                    self._enter_can_mode()
                    continue
                raise MotionError(
                    f"使能前机械臂不在 CAN 控制模式（ctrl_mode={mode}）；"
                    "请先重置到待机并切换到 CAN 模式"
                )
            err_code = status.get("err_code")
            arm_status = status.get("arm_status")
            if err_code not in (None, 0):
                if self.reset_teaching:
                    time.sleep(0.05)
                    continue
                raise MotionError(f"使能前机械臂故障码非零（err_code={err_code}）")
            if arm_status != 0:
                if arm_status in RECOVERABLE_ARM_STATUSES:
                    time.sleep(0.05)
                    continue
                raise MotionError(f"使能前机械臂状态异常（arm_status={arm_status}）")

            try:
                last = [bool(v) for v in self.piper.GetArmEnableStatus()]
            except Exception:
                last = None
            if last and len(last) >= 6 and all(last[:6]):
                # Require a normal status frame after the enable transition;
                # this is the point at which it is safe to send JointCtrl.
                verify = read_arm_status(self.piper)
                if not self._real_status(verify):
                    time.sleep(0.05)
                    continue
                verify_mode = verify.get("ctrl_mode")
                verify_arm = verify.get("arm_status")
                verify_err = verify.get("err_code")
                if verify_mode != CAN_CTRL_MODE:
                    if self.reset_teaching:
                        if verify_mode in TEACHING_CTRL_MODES:
                            self._reset_teaching_to_standby()
                        self._enter_can_mode()
                        continue
                    raise MotionError(
                        f"使能后机械臂不在 CAN 控制模式（ctrl_mode={verify_mode}）；"
                        "拒绝发送关节目标"
                    )
                if verify_err not in (None, 0):
                    if self.reset_teaching:
                        time.sleep(0.05)
                        continue
                    raise MotionError(f"使能后机械臂故障码非零（err_code={verify_err}）")
                if verify_arm == 0:
                    return
                if verify_arm in RECOVERABLE_ARM_STATUSES:
                    time.sleep(0.05)
                    continue
                raise MotionError(f"使能后机械臂状态异常（arm_status={verify_arm}）")
            # The SDK's EnablePiper() returns the state observed *before* its
            # frame.  The official control loop therefore calls it repeatedly
            # until a later feedback frame reports all six motors enabled;
            # one frame is not sufficient after ResetPiper or a mode drop.
            self.piper.EnablePiper()
            time.sleep(0.05)
        status_text = (
            f", 最后状态 arm_status={last_status.get('arm_status')}"
            if last_status is not None
            else ""
        )
        raise MotionError(
            f"{self.enable_timeout_s:.1f}s 内无法使能六个关节，反馈={last}{status_text}"
        )

    def move_to_target(self) -> MotionResult:
        """Move to the target and require settled feedback before returning."""

        started = time.monotonic()
        self._wait_until_ready()
        self._enable()
        period = 1.0 / self.command_hz
        deadline = started + self.timeout_s
        settled_since: Optional[float] = None
        command_count = 0
        final: Optional[List[float]] = None
        max_error: Optional[float] = None

        while time.monotonic() < deadline:
            self._ensure_can_and_normal()
            # Position/velocity mode + MOVE J.  Sending both frames at a
            # bounded rate is the pattern used by the official SDK examples.
            self.piper.MotionCtrl_2(CAN_CTRL_MODE, 0x01, self.speed_percent, 0x00)
            self.piper.JointCtrl(*self.target_raw)
            command_count += 1

            final = read_joint_positions_deg(self.piper)
            if final is not None:
                errors = [abs(a - b) for a, b in zip(final, self.target_deg)]
                max_error = max(errors)
                if max_error <= self.tolerance_deg:
                    if settled_since is None:
                        settled_since = time.monotonic()
                    elif time.monotonic() - settled_since >= self.settle_s:
                        return MotionResult(
                            status="success",
                            target_deg=list(self.target_deg),
                            final_deg=list(final),
                            elapsed_s=time.monotonic() - started,
                            command_count=command_count,
                            max_error_deg=max_error,
                        )
                else:
                    settled_since = None
            time.sleep(period)

        return MotionResult(
            status="timeout",
            target_deg=list(self.target_deg),
            final_deg=list(final) if final is not None else None,
            elapsed_s=time.monotonic() - started,
            command_count=command_count,
            max_error_deg=max_error,
            reason=f"{self.timeout_s:.1f}s 内未达到 ±{self.tolerance_deg:g}° 并稳定 {self.settle_s:g}s",
        )
