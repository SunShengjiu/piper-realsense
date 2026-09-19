"""Motion safety and feedback handling without connecting to CAN hardware."""
from types import SimpleNamespace

import pytest

from piper_capture.motion import MotionError, PiperMotionController


class FakePiper:
    def __init__(self, *, mode=1, joints=None, enabled=True):
        self.mode = mode
        self.joints = list(joints or [0, 0, 0, 0, 0, 0])
        self.enabled = enabled
        self.commands = []
        self._stamp = 1

    def GetArmStatus(self):
        return SimpleNamespace(
            arm_status=SimpleNamespace(ctrl_mode=self.mode, arm_status=0, teach_status=0, motion_status=0, err_code=0)
        )

    def GetArmEnableStatus(self):
        return [self.enabled] * 6

    def EnablePiper(self):
        self.commands.append(("enable",))
        self.enabled = True

    def MotionCtrl_2(self, *args):
        self.commands.append(("mode", args))

    def ModeCtrl(self, *args):
        self.commands.append(("mode_ctrl", args))
        self.mode = 1

    def ResetPiper(self):
        self.commands.append(("reset",))
        self.mode = 0

    def JointCtrl(self, *args):
        self.commands.append(("joint", args))
        self.joints = [int(v) for v in args]

    def GetArmJointMsgs(self):
        state = SimpleNamespace(**{f"joint_{i}": value for i, value in enumerate(self.joints, start=1)})
        return SimpleNamespace(time_stamp=self._stamp, joint_state=state)


def test_move_to_zero_requires_can_mode():
    piper = FakePiper(mode=2)
    with pytest.raises(MotionError, match="CAN 控制模式"):
        PiperMotionController(piper, timeout_s=0.2).move_to_target()
    assert not [c for c in piper.commands if c[0] in ("mode", "joint")]


def test_move_to_zero_sends_position_commands_and_waits_for_feedback():
    piper = FakePiper(joints=[45000, 0, 0, 0, 0, 0])
    result = PiperMotionController(piper, timeout_s=1.0, settle_s=0.01, command_hz=100).move_to_target()
    assert result.status == "success"
    assert result.final_deg == [0.0] * 6
    assert any(c[0] == "mode" for c in piper.commands)
    assert any(c[0] == "joint" and c[1] == (0, 0, 0, 0, 0, 0) for c in piper.commands)
    assert result.to_dict()["zero_semantics"].startswith("move_to_pose")


def test_teach_to_can_transition_is_required_for_button_trigger():
    piper = FakePiper(mode=2)
    controller = PiperMotionController(piper)
    statuses = iter([2, 2, 1])

    def status():
        piper.mode = next(statuses)
        return SimpleNamespace(
            arm_status=SimpleNamespace(ctrl_mode=piper.mode, arm_status=0, teach_status=0, motion_status=0, err_code=0)
        )

    piper.GetArmStatus = status
    trigger = controller.wait_for_teach_to_can(timeout_s=0.5, poll_hz=100)
    assert trigger["status"] == "triggered"
    assert trigger["initial_ctrl_mode"] == 2
    assert trigger["final_ctrl_mode"] == 1


def test_teach_button_stop_is_detected_without_can_transition():
    piper = FakePiper(mode=2)
    statuses = iter([
        (2, 0x0B, 0x01),  # recording
        (2, 0x00, 0x02),  # single-click stop; mode remains teaching
    ])

    def status():
        mode, arm_status, teach_status = next(statuses)
        return SimpleNamespace(
            arm_status=SimpleNamespace(
                ctrl_mode=mode,
                arm_status=arm_status,
                teach_status=teach_status,
                motion_status=0,
                err_code=0,
            )
        )

    piper.GetArmStatus = status
    trigger = PiperMotionController(piper).wait_for_teach_button(timeout_s=0.5, poll_hz=100)
    assert trigger["status"] == "triggered"
    assert trigger["trigger"] == "teach_record_stop"
    assert trigger["final_ctrl_mode"] == 2


def test_teach_stop_event_does_not_require_recording_frame():
    piper = FakePiper(mode=2)
    statuses = iter([
        (2, 0x00, 0x00),
        (2, 0x00, 0x02),  # stop feedback edge sampled; start edge was missed
    ])

    def status():
        mode, arm_status, teach_status = next(statuses)
        return SimpleNamespace(
            arm_status=SimpleNamespace(
                ctrl_mode=mode,
                arm_status=arm_status,
                teach_status=teach_status,
                motion_status=0,
                err_code=0,
            )
        )

    piper.GetArmStatus = status
    trigger = PiperMotionController(piper).wait_for_teach_button(timeout_s=0.5, poll_hz=100)
    assert trigger["trigger"] == "teach_record_stop"


def test_explicit_teaching_reset_enters_can_before_motion():
    piper = FakePiper(mode=2, joints=[0, 0, 0, 0, 0, 0])
    result = PiperMotionController(
        piper, reset_teaching=True, timeout_s=1.0, settle_s=0.01, command_hz=100
    ).move_to_target()
    assert result.status == "success"
    assert ("reset",) in piper.commands
    assert any(c[0] == "mode_ctrl" for c in piper.commands)


def test_explicit_motion_can_start_from_standby():
    piper = FakePiper(mode=0, joints=[0, 0, 0, 0, 0, 0])
    result = PiperMotionController(
        piper, reset_teaching=True, timeout_s=1.0, settle_s=0.01, command_hz=100
    ).move_to_target()
    assert result.status == "success"
    assert any(c[0] == "mode_ctrl" for c in piper.commands)


def test_can_handshake_waits_for_transient_fault_clear():
    piper = FakePiper(mode=0, joints=[0, 0, 0, 0, 0, 0])
    statuses = iter([(0, 63), (0, 0), (1, 0)])

    def status():
        mode, err_code = next(statuses, (1, 0))
        piper.mode = mode
        return SimpleNamespace(
            arm_status=SimpleNamespace(
                ctrl_mode=mode,
                arm_status=0,
                teach_status=0,
                motion_status=0,
                err_code=err_code,
            )
        )

    piper.GetArmStatus = status
    result = PiperMotionController(
        piper, reset_teaching=True, timeout_s=1.0, settle_s=0.01, command_hz=100
    ).move_to_target()
    assert result.status == "success"
    assert any(c[0] == "mode_ctrl" for c in piper.commands)


def test_can_handshake_waits_for_joint_communication_recovery():
    piper = FakePiper(mode=0, joints=[0, 0, 0, 0, 0, 0])
    statuses = iter([(0, 5), (0, 0), (1, 5), (1, 0)])

    def status():
        mode, arm_status = next(statuses, (1, 0))
        piper.mode = mode
        return SimpleNamespace(
            arm_status=SimpleNamespace(
                ctrl_mode=mode,
                arm_status=arm_status,
                teach_status=0,
                motion_status=0,
                err_code=0,
            )
        )

    piper.GetArmStatus = status
    result = PiperMotionController(
        piper, reset_teaching=True, timeout_s=1.0, settle_s=0.01, command_hz=100
    ).move_to_target()
    assert result.status == "success"
    assert any(c[0] == "mode_ctrl" for c in piper.commands)


def test_enable_waits_for_post_enable_communication_recovery():
    piper = FakePiper(mode=1, joints=[0, 0, 0, 0, 0, 0], enabled=False)
    statuses = iter([(1, 0), (1, 5), (1, 0), (1, 0)])

    def status():
        mode, arm_status = next(statuses, (1, 0))
        return SimpleNamespace(
            arm_status=SimpleNamespace(
                ctrl_mode=mode,
                arm_status=arm_status,
                teach_status=0,
                motion_status=0,
                err_code=0,
            )
        )

    piper.GetArmStatus = status
    result = PiperMotionController(
        piper, reset_teaching=True, timeout_s=1.0, settle_s=0.01, command_hz=100
    ).move_to_target()
    assert result.status == "success"
    assert ("enable",) in piper.commands


def test_target_publish_recovers_if_mode_drops_to_standby():
    piper = FakePiper(mode=1, joints=[0, 0, 0, 0, 0, 0], enabled=True)
    statuses = iter([(1, 0), (1, 0), (1, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0)])

    def status():
        mode, arm_status = next(statuses, (1, 0))
        piper.mode = mode
        return SimpleNamespace(
            arm_status=SimpleNamespace(
                ctrl_mode=mode,
                arm_status=arm_status,
                teach_status=0,
                motion_status=0,
                err_code=0,
            )
        )

    piper.GetArmStatus = status
    result = PiperMotionController(
        piper, reset_teaching=True, timeout_s=1.0, settle_s=0.01, command_hz=100
    ).move_to_target()
    assert result.status == "success"
    assert sum(c[0] == "mode_ctrl" for c in piper.commands) >= 1


def test_target_publish_reenables_motors_if_enable_bit_drops():
    class DropEnablePiper(FakePiper):
        def __init__(self):
            super().__init__(mode=1, joints=[45000, 0, 0, 0, 0, 0], enabled=True)
            self.joint_calls = 0

        def JointCtrl(self, *args):
            super().JointCtrl(*args)
            self.joint_calls += 1
            if self.joint_calls == 1:
                self.enabled = False

    piper = DropEnablePiper()
    result = PiperMotionController(
        piper, reset_teaching=True, timeout_s=1.0, settle_s=0.01, command_hz=100
    ).move_to_target()
    assert result.status == "success"
    assert sum(c[0] == "enable" for c in piper.commands) >= 1


def test_invalid_target_pose_is_rejected_before_any_command():
    piper = FakePiper()
    with pytest.raises(ValueError, match="joint2"):
        PiperMotionController(piper, target_deg=[0, -1, 0, 0, 0, 0])
    assert piper.commands == []
