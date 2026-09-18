"""运行配置。

默认值都能在“没有实物标定”的前提下安全运行：只读、不发送运动指令、
不把未标定的量填成数字。实物相关字段留空时下游必须写 null 并给出原因。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset_root": "dataset",
    # ---------------- 机械臂 ----------------
    "robot": {
        "enabled": True,
        "backend": "piper_sdk_direct_can",
        "can_interface": "can0",
        "can_bitrate": 1000000,
        # piper_sdk 的 DH 是否有 2° 偏置（0x00/0x01），对应 sdk/型号代际
        "dh_is_offset": 1,
        "sdk_joint_limit": False,
        "sdk_gripper_limit": False,
        "poll_hz": 200.0,
        # 只读：PiPER 协议无设备侧时间戳，关节反馈只有主机接收时刻
        "read_only": True,
        "command_interface": {"available": False, "note": "本项目未接入下发命令记录"},
        "feedback_timeout_s": 1.0,
        "ee_frame": "link6",
        "base_frame": "piper_base_link",
        # 用户约定的数采起始姿态；仅保存目标，不在开始采集时自动运动。
        "start_pose": {
            "name": "capture_start",
            "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            "joint_positions_deg": [90.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "reference": "existing_robot_joint_zero",
            "direction_convention": "joint1 positive: counterclockwise viewed from above base +Z toward origin",
            "source": "user_requested_joint1_left_90_degrees",
            "auto_move_on_capture": False,
        },
        # 官方 URDF，用于 FK 三方交叉校验（存在才校验）
        "urdf_path": "/home/robot/codeaspolicy/src/robot_pick_place_agent/assets/piper/upstream/piper_description.urdf",
        # 工具偏移：T_link6_tool。未实测时保持全零并记录来源
        "tool_offset_m": [0.0, 0.0, 0.0],
        "tool_offset_source": "unconfigured_default_zero",
    },
    # ---------------- 夹爪 ----------------
    "gripper": {
        "calibration_id": None,
        "combined_gripper_joint": True,
        # 参考值：官方 piper_sdk 参数夹爪行程 [0, 0.07] m，即 0..70 mm 总开度
        "driver_range_mm": [0.0, 70.0],
        "raw_unit_mm": 0.001,
    },
    # ---------------- 相机 ----------------
    "camera": {
        "enabled": True,
        "serial": None,
        "color": {"width": 1280, "height": 720, "fps": 30, "format": "bgr8"},
        "depth": {"width": 1280, "height": 720, "fps": 30, "format": "z16"},
        # 需求要求 1280x720@30；若不满足则必须报错，不得静默降级
        "allow_spec_downgrade": False,
        "save_rgb": True,
        "rgb_format": "png",
        "depth_format": "png",
        # PNG 压缩级别 0-9 都是无损，只影响编码耗时。实测 1280x720 三张图
        # 串行编码：0 级 31ms、1 级 64ms、3 级 88ms。0 级无损且最快。
        "png_compression": 0,
        # 落盘与取帧解耦：主循环只入队，PNG 编码在后台线程完成。
        # 并行 3 张（compression=0）实测约 16ms < 33ms 帧周期，才能到 30 样本/秒。
        "save_workers": 3,
        "max_pending_saves": 64,
        "warmup_frames": 30,
        "frame_timeout_ms": 5000,
        "calibration_id": None,
        "mount": {
            "link": None,
            "source": "unconfigured",
            "note": "官方打印件参数只作参考，实测以手眼标定结果为准；相机固定在末端则用 eye_in_hand",
            # 参考值，来源：Agilex-College/piper/handpose_det/models/modified_piper.urdf
            # 只作复核依据，绝不写进数据集当作实测标定结果
            "official_reference": {
                "source_file": "agilexrobotics/Agilex-College: piper/handpose_det/models/modified_piper.urdf",
                "camera_base2link6": {
                    "xyz_m": [0.0, 0.075, 0.03],
                    "rpy_rad": [-1.5708, -1.5708, 0.0],
                },
                "camera_link2camera_base": {
                    "xyz_m": [0.0, 0.01, 0.0],
                    "rpy_rad": [0.0, 0.45, 0.0],
                },
                "what_it_implies": "官方示例把相机基座挂在 link6，请现场确认实际安装连杆",
                "usage_policy": "仅作参考与复核；不是实测标定结果，不得作为 T_ee_camera 写入数据集",
            },
        },
    },
    # ---------------- 采样与时间同步 ----------------
    "capture": {
        "target_sample_rate": 30.0,
        # 只有平均样本率超过 target*(1+tolerance) 才跳过帧；用于吸收帧间隔抖动
        "rate_tolerance": 0.05,
        "sync": {
            "tolerance_ms": 33.0,
            "robot_tolerance_ms": 33.0,
            # nearest: 只取最近反馈并标记 dt；linear: 在两帧间线性插值（记录所用原始状态）
            "robot_match_mode": "nearest",
            "max_robot_state_age_ms": 50.0,
            "allow_stale_fill": False,
        },
        "log_robot_states": True,
        "fsync_every": 0,
        "stop_on_robot_timeout": False,
    },
    # ---------------- 手眼标定 ----------------
    "handeye": {
        "mode": "eye_in_hand",
        "board": {
            "type": "aruco_single",
            # 来源：Agilex-College `piper/handeye/README.md`
            #   "建议使用 Original ArUco 字典的标定板"
            #   ros2 launch aruco_ros single.launch.py marker_id:=582 marker_size:=0.0677
            # 对应 OpenCV 的 DICT_ARUCO_ORIGINAL（1024 个 5x5 码）。
            # 注意：同一个 id 在不同字典下图案完全不同（DICT_4X4_1000 是 1000 个
            # 4x4 码），字典选错会检测不到或检测到别的 id，必须与手上实物板一致。
            "dictionary": "DICT_ARUCO_ORIGINAL",
            "marker_id": 582,
            # 官方示例值，必须用卡尺实测实物板边长后复核
            "marker_size_m": 0.0677,
            "length_unit": "m",
            "source": "agilexrobotics/Agilex-College: piper/handeye/README.md",
            "note": "字典/ID/边长均为官方默认值，需以实物标定板核对；边长未实测前不得当作实测值",
        },
        "solve_method": "TSAI",
        "compare_methods": ["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"],
        "min_samples": 12,
        "min_rotation_span_deg": 30.0,
        "max_translation_span_m": 0.6,
        "verify_holdout_ratio": 0.25,
        # 留出验证阈值：超过任一项则 status=invalid（不会用单位矩阵顶替）
        "verify_max_position_rms_mm": 10.0,
        "verify_max_rotation_rms_deg": 2.0,
        "ee_frame_for_solve": "link6",
        "camera_optical_frame": "camera_color_optical_frame",
    },
    # ---------------- 第三人称视频 ----------------
    "external": {
        "roles": ["side_task", "environment_overview"],
        "probe_with_ffmpeg": True,
    },
}

_SECTIONS = tuple(DEFAULT_CONFIG.keys())


class ConfigError(RuntimeError):
    pass


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, *, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise ConfigError(f"配置文件不存在: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            cfg = _merge(cfg, json.load(fh))
    if overrides:
        cfg = _merge(cfg, overrides)
    if cfg["robot"]["dh_is_offset"] not in (0, 1, 0x00, 0x01):
        raise ConfigError("robot.dh_is_offset 只能是 0 或 1")
    if cfg["handeye"]["mode"] not in ("eye_in_hand", "eye_to_hand"):
        raise ConfigError("handeye.mode 只能是 eye_in_hand 或 eye_to_hand")
    return cfg


def dataset_root(cfg: Dict[str, Any], base: Path | None = None) -> Path:
    root = Path(cfg["dataset_root"]).expanduser()
    if not root.is_absolute():
        root = (base or Path.cwd()) / root
    return root.resolve()
