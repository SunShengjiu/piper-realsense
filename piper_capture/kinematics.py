"""正运动学与 FK 交叉校验。

数据集的 ee_pose 必须由**反馈关节角**经正运动学算出，不能用目标关节角，
也不能用相机测量代替。这里提供两条互相独立的实现：

1. `dh_fk_link6`：按 piper_sdk 0.6.2 `kinematics/piper_fk.py` 的 DH 参数
   在本项目内重新实现（不改动 SDK），输出米制 4x4。
2. `urdf_fk_link6`：按官方 URDF 的关节 origin/axis 链式计算。

两者与 piper_sdk 自带的 `C_PiperForwardKinematics.CalFK` 三方比对，
用于验证关节顺序、零位偏移和单位约定（见 tools/verify_fk.py）。

SDK 原始提示：`C_PiperForwardKinematics.CalFK` 的 XYZ 单位是 mm，
RX/RY/RZ 单位是度，返回 6 个关节相对 base_link 的位姿。
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .schema import JOINT_NAMES, Transform

# ---------------------------------------------------------------------------
# DH 参数：数值来自 piper_sdk 0.6.2 kinematics/piper_fk.py（未修改上游文件）。
# _a: 连杆长度(mm)  _alpha: 连杆扭角(rad)  _theta: 关节零位偏置(rad)  _d: 连杆偏置(mm)
# ---------------------------------------------------------------------------
DH_TABLES: Dict[int, Dict[str, List[float]]] = {
    0x00: {
        "a": [0.0, 0.0, 285.03, -21.98, 0.0, 0.0],
        "alpha": [0.0, -math.pi / 2, 0.0, math.pi / 2, -math.pi / 2, math.pi / 2],
        "theta": [0.0, -math.pi * 174.22 / 180.0, -100.78 / 180.0 * math.pi, 0.0, 0.0, 0.0],
        "d": [123.0, 0.0, 0.0, 250.75, 0.0, 91.0],
    },
    0x01: {
        "a": [0.0, 0.0, 285.03, -21.98, 0.0, 0.0],
        "alpha": [0.0, -math.pi / 2, 0.0, math.pi / 2, -math.pi / 2, math.pi / 2],
        "theta": [0.0, -math.pi * 172.22 / 180.0, -102.78 / 180.0 * math.pi, 0.0, 0.0, 0.0],
        "d": [123.0, 0.0, 0.0, 250.75, 0.0, 91.0],
    },
}


def dh_link_transform(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    """单连杆变换，逐元素复刻 piper_sdk 的 __LinkTransformtion（含其旋转写法）。

    SDK 展开后的矩阵为：
        R = [[ct, -st, 0], [st*ca, ct*ca, -sa], [st*sa, ct*sa, ca]]
        p = [a, -sa*d, ca*d]
    与教科书标准 DH 的 R 写法不同，这里以 SDK 源码为准。
    """
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    T = np.eye(4)
    T[:3, :3] = [
        [ct, -st, 0.0],
        [st * ca, ct * ca, -sa],
        [st * sa, ct * sa, ca],
    ]
    T[:3, 3] = [a, -sa * d, ca * d]
    return T


def dh_fk_link_matrices(joints_rad: Sequence[float], dh_is_offset: int = 0x01) -> List[np.ndarray]:
    """返回 base_link 到 link1..link6 的 4x4（单位：米）。"""
    if len(joints_rad) != 6:
        raise ValueError(f"需要 6 个关节角，收到 {len(joints_rad)}")
    table = DH_TABLES[int(dh_is_offset)]
    T = np.eye(4)
    out: List[np.ndarray] = []
    for i in range(6):
        theta = float(joints_rad[i]) + table["theta"][i]
        link_mm = dh_link_transform(table["alpha"][i], table["a"][i], theta, table["d"][i])
        link = np.eye(4)
        link[:3, :3] = link_mm[:3, :3]
        link[:3, 3] = link_mm[:3, 3] / 1000.0  # mm -> m
        T = T @ link
        out.append(T.copy())
    return out


def dh_fk_link6(joints_rad: Sequence[float], dh_is_offset: int = 0x01) -> np.ndarray:
    """base_link -> link6 的 4x4（单位：米）。"""
    return dh_fk_link_matrices(joints_rad, dh_is_offset)[5]


# ---------------------------------------------------------------------------
# URDF 链式 FK（官方 piper_description.urdf）
# ---------------------------------------------------------------------------

def parse_urdf_chain(urdf_path: str | Path) -> List[Dict[str, object]]:
    """从 URDF 中提取 base_link -> link6 的 6 个转动关节。"""
    root = ET.parse(str(urdf_path)).getroot()
    joints: Dict[str, Dict[str, object]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        jtype = joint.get("type")
        if name is None or jtype is None:
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        axis = joint.find("axis")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            if origin.get("xyz"):
                xyz = [float(v) for v in origin.get("xyz").split()]
            if origin.get("rpy"):
                rpy = [float(v) for v in origin.get("rpy").split()]
        joints[name] = {
            "type": jtype,
            "parent": parent.get("link") if parent is not None else None,
            "child": child.get("link") if child is not None else None,
            "xyz": xyz,
            "rpy": rpy,
            "axis": [float(v) for v in axis.get("xyz").split()] if axis is not None else [0.0, 0.0, 1.0],
        }
    chain: List[Dict[str, object]] = []
    link = "base_link"
    for _ in range(8):
        match = [j for j in joints.values() if j["parent"] == link and j["type"] == "revolute"]
        if not match:
            break
        joint = match[0]
        chain.append(joint)
        link = str(joint["child"])
        if link == "link6":
            break
    if len(chain) != 6:
        raise ValueError(f"URDF 中未能解析出 6 个转动关节链，实际 {len(chain)}: {urdf_path}")
    return chain


def urdf_fk_link6(joints_rad: Sequence[float], chain: Sequence[Dict[str, object]]) -> np.ndarray:
    """按 URDF 关节 origin/axis 链式计算 base_link -> link6 的 4x4（单位：米）。"""
    if len(joints_rad) != 6:
        raise ValueError(f"需要 6 个关节角，收到 {len(joints_rad)}")
    T = np.eye(4)
    for q, joint in zip(joints_rad, chain):
        T = T @ Transform.from_rpy_xyz(joint["rpy"], joint["xyz"])
        axis = np.asarray(joint["axis"], dtype=float)
        axis = axis / np.linalg.norm(axis)
        K = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        R = np.eye(3) + math.sin(float(q)) * K + (1.0 - math.cos(float(q))) * (K @ K)
        T = T @ Transform.from_rt(R, [0.0, 0.0, 0.0])
    return T


# ---------------------------------------------------------------------------
# 组合：反馈关节角 -> 数据集 ee_pose
# ---------------------------------------------------------------------------

class ForwardKinematics:
    """按配置把反馈关节角转成数据集 EE 位姿。

    base_frame -> ee_frame 由运动学模型给出；tool_offset_m 是 ee_frame(默认 link6)
    到工具 TCP 的平移偏移，未实测时保持零向量并在数据集里记录来源。
    """

    def __init__(
        self,
        dh_is_offset: int = 0x01,
        tool_offset_m: Sequence[float] = (0.0, 0.0, 0.0),
        ee_frame: str = "link6",
        base_frame: str = "piper_base_link",
        urdf_chain: Sequence[Dict[str, object]] | None = None,
    ) -> None:
        self.dh_is_offset = int(dh_is_offset)
        self.tool_offset_m = [float(v) for v in tool_offset_m]
        self.ee_frame = ee_frame
        self.base_frame = base_frame
        self.urdf_chain = list(urdf_chain) if urdf_chain else None

    def fk_base_link6(self, joints_rad: Sequence[float]) -> np.ndarray:
        if self.urdf_chain:
            return urdf_fk_link6(joints_rad, self.urdf_chain)
        return dh_fk_link6(joints_rad, self.dh_is_offset)

    def tool_transform(self) -> np.ndarray:
        return Transform.from_rt(np.eye(3), self.tool_offset_m)

    def fk_base_ee(self, joints_rad: Sequence[float]) -> np.ndarray:
        """T_base_ee：ee_frame 为 link6 时等价于 T_base_link6，再叠加工具偏移。"""
        T = self.fk_base_link6(joints_rad)
        if any(abs(v) > 0.0 for v in self.tool_offset_m):
            T = T @ self.tool_transform()
        return T

    def model_id(self) -> str:
        chain = "urdf_chain" if self.urdf_chain else f"dh_is_offset_{self.dh_is_offset}"
        return f"piper_sdk_0.6.2_dh[{chain}]"

    def describe(self) -> Dict[str, object]:
        return {
            "kinematics_model": self.model_id(),
            "base_frame": self.base_frame,
            "ee_frame": self.ee_frame,
            "tool_offset_m": self.tool_offset_m,
            "joint_names": list(JOINT_NAMES),
            "joint_units": "rad",
            "ee_pose_layout": "[x, y, z, qw, qx, qy, qz]",
            "ee_pose_source": "feedback_joint_fk",
        }


def sdk_cal_fk(joints_rad: Sequence[float], dh_is_offset: int = 0x01) -> List[List[float]]:
    """调用 piper_sdk 自带 FK，仅用于交叉校验（XYZ: mm，RPY: deg）。"""
    from piper_sdk import C_PiperForwardKinematics

    return C_PiperForwardKinematics(dh_is_offset).CalFK([float(v) for v in joints_rad])


class QuatContinuityTracker:
    """保持 EE 四元数相邻帧符号连续（复用 schema.QuatContinuity）。"""

    def __init__(self) -> None:
        from .schema import QuatContinuity

        self._impl = QuatContinuity()

    @property
    def flips(self) -> int:
        return self._impl.flips

    def apply_pose(self, pose_7: Sequence[float]) -> List[float]:
        x, y, z, qw, qx, qy, qz = [float(v) for v in pose_7]
        qw, qx, qy, qz = self._impl.apply([qw, qx, qy, qz])
        return [x, y, z, qw, qx, qy, qz]

    def reset(self) -> None:
        self._impl.reset()