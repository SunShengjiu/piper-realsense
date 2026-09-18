"""数据集 schema 常量与几何/时间约定。

约定（与官方手眼标定教程一致）：
  - 关节角单位为弧度，顺序固定 JOINT_NAMES。
  - ee_pose = [x, y, z, qw, qx, qy, qz]，位置单位米，四元数 wxyz 且单位化。
  - 齐次变换 T_A_B 的含义是 p_A = T_A_B @ p_B（把 B 系下的点变换到 A 系）。
    因此 T_ee_camera 表示 p_ee = T_ee_camera @ p_camera。
  - URDF/标定 rpy 使用固定轴 roll-pitch-yaw，即 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = "1.0.0"

#: 固定关节顺序，所有数组、标定与数据集字段都遵循该顺序。
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

#: 单位声明，写入每个数据集的 units 段，避免下游猜单位。
UNIT_DECLARATIONS = {
    "joint_positions_rad": "rad",
    "ee_pose.position": "m",
    "ee_pose.quaternion": "wxyz, unit norm",
    "ee_pose.tool_offset_m": "m",
    "gripper_width_mm": "mm (total two-finger opening)",
    "gripper_feedback_raw": "driver raw int, unit 0.001 mm",
    "depth_m": "m = raw_depth_uint16 * depth_scale",
    "timestamp": "ns, host clock domains recorded per field",
}


#: 夹爪反馈 status_code 位定义，来自 piper_sdk 0.6.2
#: `piper_msgs/msg_v2/feedback/arm_feedback_gripper.py`（CAN 0x2A8, Byte 6）。
GRIPPER_STATUS_BITS = {
    0: "voltage_too_low",
    1: "motor_overheating",
    2: "driver_overcurrent",
    3: "driver_overheating",
    4: "sensor_abnormal",
    5: "driver_error",
    6: "driver_enabled",
    7: "homing_done",
}


def decode_gripper_status(code: int | None) -> dict | None:
    """把夹爪 status_code 拆成各状态位。

    bit6(driver_enabled) 是“使能状态”，bit4/bit5 是异常位；
    使能与否不影响反馈是否可信，因此 gripper_valid 只看异常位。
    """
    if code is None:
        return None
    return {name: bool(int(code) & (1 << bit)) for bit, name in GRIPPER_STATUS_BITS.items()}


class Transform:
    """4x4 齐次变换的小工具集。"""

    @staticmethod
    def identity() -> np.ndarray:
        return np.eye(4, dtype=float)

    @staticmethod
    def from_rt(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
        T = np.eye(4, dtype=float)
        T[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
        T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
        return T

    @staticmethod
    def from_pose(pose_7: Sequence[float]) -> np.ndarray:
        """[x, y, z, qw, qx, qy, qz] -> 4x4。"""
        x, y, z, qw, qx, qy, qz = [float(v) for v in pose_7]
        return Transform.from_rt(quat_to_matrix([qw, qx, qy, qz]), [x, y, z])

    @staticmethod
    def to_pose(T: np.ndarray) -> List[float]:
        """4x4 -> [x, y, z, qw, qx, qy, qz]。"""
        qw, qx, qy, qz = matrix_to_quat(np.asarray(T, dtype=float)[:3, :3])
        t = np.asarray(T, dtype=float)[:3, 3]
        return [float(t[0]), float(t[1]), float(t[2]), qw, qx, qy, qz]

    @staticmethod
    def invert(T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        R = T[:3, :3]
        t = T[:3, 3]
        out = np.eye(4)
        out[:3, :3] = R.T
        out[:3, 3] = -R.T @ t
        return out

    @staticmethod
    def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
        """固定轴 roll-pitch-yaw -> 旋转矩阵，R = Rz(yaw) Ry(pitch) Rx(roll)。"""
        r, p, y = [float(v) for v in rpy]
        cr, sr = math.cos(r), math.sin(r)
        cp, sp = math.cos(p), math.sin(p)
        cy, sy = math.cos(y), math.sin(y)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
        return Rz @ Ry @ Rx

    @staticmethod
    def from_rpy_xyz(rpy: Sequence[float], xyz: Sequence[float]) -> np.ndarray:
        return Transform.from_rt(Transform.rpy_to_matrix(rpy), xyz)

    @staticmethod
    def matrix_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
        R = np.asarray(R, dtype=float)
        pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
        if abs(math.cos(pitch)) < 1e-9:
            roll = math.atan2(R[0, 1], R[1, 1])
            yaw = 0.0
        else:
            roll = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(R[1, 0], R[0, 0])
        return roll, pitch, yaw

    @staticmethod
    def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
        """两个旋转矩阵之间的测地角，单位度。"""
        R = np.asarray(R_a, dtype=float)[:3, :3].T @ np.asarray(R_b, dtype=float)[:3, :3]
        cos_theta = (np.trace(R) - 1.0) / 2.0
        return math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))


def normalize_quat(q: Sequence[float]) -> Tuple[float, float, float, float]:
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        raise ValueError("零四元数无法归一化")
    q = q / n
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def matrix_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """旋转矩阵 -> 四元数 (qw, qx, qy, qz)，Shepperd 分支法，数值稳定。"""
    R = np.asarray(R, dtype=float)[:3, :3]
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return normalize_quat([qw, qx, qy, qz])


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    """四元数 (qw, qx, qy, qz) -> 旋转矩阵。"""
    qw, qx, qy, qz = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


class QuatContinuity:
    """处理相邻帧 q 与 -q 的符号连续性。

    同一姿态的 q 和 -q 表示同一旋转，但做插值/差分时必须保持半球一致。
    规则：若当前帧与上一帧的内积为负，取反当前帧。
    """

    def __init__(self) -> None:
        self._last: Tuple[float, float, float, float] | None = None
        self.flips = 0

    def apply(self, q: Sequence[float]) -> Tuple[float, float, float, float]:
        qw, qx, qy, qz = normalize_quat(q)
        if self._last is not None:
            dot = qw * self._last[0] + qx * self._last[1] + qy * self._last[2] + qz * self._last[3]
            if dot < 0.0:
                qw, qx, qy, qz = -qw, -qx, -qy, -qz
                self.flips += 1
        self._last = (qw, qx, qy, qz)
        return self._last

    def reset(self) -> None:
        self._last = None