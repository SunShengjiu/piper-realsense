"""手眼标定：采样、求解与验证。

沿用官方流程（Agilex-College `piper/handeye` 与
`agilexrobotics/handeye_calibration_ros`）：
  - eye-in-hand：相机固定在末端随末端运动，求 T_ee_camera；
  - 机械臂静止、标定板检测有效时记录成对样本；
  - `cv2.calibrateHandEye` 求解，TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS 对比；
  - 用未参与求解的姿态做留出验证，输出可量化误差。

坐标与方向约定（与 schema.py 一致）：
  T_A_B 表示 p_A = T_A_B @ p_B。
  T_cam_target 由 ArUco 或棋盘格检测得到（板坐标系在相机系下的位姿）。
  ArUco 原点为码中心；棋盘格原点为固定排序的首个内角点。
  T_base_ee 由**反馈关节角**正运动学算出（默认 ee=link6）。
  求解得到 X = T_ee_camera，满足 p_ee = X @ p_camera。

验证指标（不依赖标定板真值）：
  标定板在基座系下是静止的，因此对每个样本
      T_base_target = T_base_ee @ T_ee_camera @ T_cam_target
  应当恒定。留出样本上 T_base_target 的位置标准差/最大偏差(mm)与旋转
  标准差/最大测地偏差(deg)即为可量化误差。

未通过验证时 `status` 只会是 `pending`/`invalid`，`matrix` 不会被
单位矩阵顶替。
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera import RealsenseCamera
from .episode import CalibrationStore, validate_id
from .jsonio import matrix_to_list, write_json
from .kinematics import ForwardKinematics
from .robot import RobotReader
from .schema import SCHEMA_VERSION, JOINT_NAMES, Transform, matrix_to_quat, normalize_quat, quat_to_matrix

_DICTS = {
    "DICT_4X4_50": 0,
    "DICT_4X4_100": 1,
    "DICT_4X4_250": 2,
    "DICT_4X4_1000": 3,
    "DICT_5X5_50": 4,
    "DICT_5X5_100": 5,
    "DICT_5X5_250": 6,
    "DICT_5X5_1000": 7,
    "DICT_6X6_50": 8,
    "DICT_6X6_100": 9,
    "DICT_6X6_250": 10,
    "DICT_6X6_1000": 11,
    "DICT_7X7_50": 12,
    "DICT_7X7_100": 13,
    "DICT_7X7_250": 14,
    "DICT_7X7_1000": 15,
    # 官方教程用的是 "Original ArUco" 字典（Agilex-College piper/handeye：
    # "建议使用 Original ArUco 字典的标定板"），对应 OpenCV 的 DICT_ARUCO_ORIGINAL。
    # 注意：同一个 marker_id 在不同字典下图案完全不同（DICT_ARUCO_ORIGINAL 是 1024 个
    # 5x5 码，DICT_4X4_1000 是 1000 个 4x4 码），标定板必须与这里的字典一致，
    # 否则会检测到别的 id 或完全检测不到。
    "DICT_ARUCO_ORIGINAL": 16,
}

SOLVE_METHODS = {
    "TSAI": "CALIB_HAND_EYE_TSAI",
    "PARK": "CALIB_HAND_EYE_PARK",
    "HORAUD": "CALIB_HAND_EYE_HORAUD",
    "ANDREFF": "CALIB_HAND_EYE_ANDREFF",
    "DANIILIDIS": "CALIB_HAND_EYE_DANIILIDIS",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def handeye_dir(root: Path) -> Path:
    d = Path(root) / "calibrations" / "handeye"
    d.mkdir(parents=True, exist_ok=True)
    return d


def session_dir(root: Path, session_id: str) -> Path:
    d = handeye_dir(root) / "sessions" / validate_id(session_id, "session_id")
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- 板检测


def build_detector(board_cfg: Dict[str, Any]) -> Dict[str, Any]:
    import cv2

    btype = board_cfg.get("type", "aruco_single")
    if btype == "checkerboard":
        shape = board_cfg.get("inner_corners")
        if (not isinstance(shape, (list, tuple)) or len(shape) != 2
                or any(type(n) is not int or n < 3 for n in shape)):
            raise ValueError("棋盘格 inner_corners 应为两个至少为 3 的整数（内角点列数、行数）")
        cols, rows = shape
        if cols % 2 == rows % 2:
            raise ValueError("当前棋盘格需一奇一偶的内角点数，以用黑白格消除 180° 排序歧义；本板为 [10, 7]")
        size = float(board_cfg.get("square_size_m", 0))
        if not np.isfinite(size) or size <= 0:
            raise ValueError("square_size_m 必须是正的有限数，单位米")
        objp = np.zeros((cols * rows, 3), dtype=np.float64)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * size
        return {"type": btype, "pattern_size": (cols, rows), "object_points": objp}
    if btype != "aruco_single":
        # 不要在这里静默按单码处理：charuco 的配置字段（square_length_m 等）
        # 与单码不同，静默走单码分支只会抛出难懂的 KeyError
        raise ValueError(
            f"当前 detect_board 支持 aruco_single 和 checkerboard；收到 type={btype}。"
            "如需 charuco，请先补齐 board 配置字段与检测实现"
        )
    dict_name = board_cfg.get("dictionary", "DICT_ARUCO_ORIGINAL")
    if dict_name not in _DICTS:
        raise ValueError(f"不支持的 ArUco 字典: {dict_name}，可用: {sorted(_DICTS)}")
    dictionary = cv2.aruco.getPredefinedDictionary(_DICTS[dict_name])
    params = cv2.aruco.DetectorParameters()
    return {"type": btype, "dictionary": dictionary, "params": params, "board": None, "dict_name": dict_name}


def _detect_checkerboard(image_bgr: Any, board_cfg: Dict[str, Any], camera_model: Any,
                         det: Dict[str, Any]) -> Dict[str, Any]:
    """完整棋盘格检测；以首角点负 x/负 y 象限为黑格，固定棋盘坐标方向。"""
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cols, rows = det["pattern_size"]
    out = {"found": False, "board_type": "checkerboard", "image_size": [w, h],
           "inner_corners": [cols, rows], "square_size_m": float(board_cfg["square_size_m"])}
    ok, corners = cv2.findChessboardCornersSB(
        gray, (cols, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        out["reason"] = f"未找到完整 {cols}×{rows} 内角点棋盘格，请让整块平整棋盘进入视野"
        return out
    imgp = corners.reshape(-1, 2).astype(np.float64)
    grid = det["object_points"][:, :2] / float(board_cfg["square_size_m"])
    H, _ = cv2.findHomography(grid, imgp, 0)
    if H is None:
        out["reason"] = "棋盘格平面拟合失败"
        return out

    # 不用图像左上角确定棋盘原点：相机旋转时检测顺序可能倒置。
    # 一奇一偶的内角点板在旋转 180° 后黑白颜色互换，可据此固定顺序。
    offsets = np.array([(x, y) for y in (-.1, 0, .1) for x in (-.1, 0, .1)])
    probes = np.vstack((offsets + [-.5, -.5], offsets + [.5, -.5])).astype(np.float64)
    pixels = cv2.perspectiveTransform(probes.reshape(-1, 1, 2), H).reshape(-1, 2)
    if not np.isfinite(pixels).all() or np.any(pixels < 0) or np.any(pixels >= [w - 1, h - 1]):
        out["reason"] = "棋盘格外圈未完整入镜，无法确定固定角点方向"
        return out
    intensities = cv2.remap(gray, pixels[:, 0].astype(np.float32).reshape(1, -1),
                           pixels[:, 1].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel()
    contrast = float(np.mean(intensities[9:]) - np.mean(intensities[:9]))
    if abs(contrast) < 30:
        out["reason"] = "棋盘格黑白对比不足，无法确定固定角点方向；请检查反光、遮挡和清晰度"
        return out
    if contrast < 0:
        imgp = imgp[::-1].copy()

    intr = camera_model.color_intrinsics
    K = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]], dtype=float)
    D = np.asarray(intr["coeffs"], dtype=float)
    objp = det["object_points"]
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.isfinite(tvec).all() or not np.isfinite(rvec).all():
        out["reason"] = "棋盘格 solvePnP 失败"
        return out
    R, _ = cv2.Rodrigues(rvec)
    if np.any((objp @ R.T + tvec.reshape(3))[:, 2] <= 0):
        out["reason"] = "棋盘格解算位于相机后方"
        return out
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
    reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - imgp) ** 2, axis=1))))
    # 检查整个棋盘外边界，而不仅仅是内角点边界。
    size = float(board_cfg["square_size_m"])
    boundary = np.array([[-1, -1, 0], [cols, -1, 0], [cols, rows, 0], [-1, rows, 0]], dtype=float) * size
    edge, _ = cv2.projectPoints(boundary, rvec, tvec, K, D)
    edge = edge.reshape(-1, 2)
    margin = float(min(edge[:, 0].min(), edge[:, 1].min(), w - 1 - edge[:, 0].max(), h - 1 - edge[:, 1].max()))
    area = float(cv2.contourArea(cv2.convexHull(imgp.astype(np.float32))))
    out.update({"found": True, "corner_count": len(imgp), "corners_px": imgp.tolist(),
                "rvec": rvec.reshape(3).tolist(), "tvec_m": tvec.reshape(3).tolist(),
                "T_cam_target": matrix_to_list(Transform.from_rt(R, tvec.reshape(3))),
                "corner_reproj_error_px": reproj, "corner_area_px2": area,
                "border_margin_px": margin, "orientation_contrast": abs(contrast),
                "target_frame": "checkerboard: origin=first inner corner; +x along columns; +y along rows; negative-x/negative-y adjacent square is black; +z=x cross y",
                "quality_ok": bool(reproj < 1.0 and margin > 2.0 and area > 400.0)})
    return out


def detect_board(image_bgr: Any, board_cfg: Dict[str, Any], camera_model: Any, det: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """检测标定板并给出 T_cam_target。

    aruco_single：单个 ArUco 码。物体点使用 ArUco 的标准定义（与 OpenCV
    `estimatePoseSingleMarkers` 一致）：
        point0 [-s, +s, 0] / point1 [+s, +s, 0] / point2 [+s, -s, 0] / point3 [-s, -s, 0]
    与 `cv2.aruco.detectMarkers` 返回的角点顺序（图像内顺时针，从左上开始）成对使用。
    得到的 marker 坐标系即 OpenCV/ArUco 约定：原点在码中心，+x 在图像中朝右、
    +y 在图像中朝上、**+z 由板面指向相机**（板正对相机时 R = Rx(180°)）。

    求解 flag 用 SOLVEPNP_ITERATIVE 而不是 SOLVEPNP_IPPE_SQUARE。已实测（合成码，
    fx=fy=900，板 0.0677 m，rvec 真值 = Rx(180°)+倾角旋扫）：
      - ITERATIVE 在倾角 0 / 0.001 / 0.005 / … / 0.4 rad 下重投影 RMS 全为 0.0000 px，
        且与 `cv2.aruco.estimatePoseSingleMarkers` 的结果完全一致；
      - IPPE_SQUARE / IPPE 在**板面正对相机（倾角=0）**时退化为非精确解：
        重投影 RMS 27.15 px、位置误差 4.96 mm（板距 0.17 m，即 ~3% 相对误差），
        倾角 ≥0.01 rad 才恢复精确。
    正对相机恰好是手眼采样里常见姿态，因此改用 ITERATIVE；重投影误差仍会
    作为质量判据输出，不依赖求解器"应该是对的"。

    注意：单个正方码在"绕板面内轴翻转 180°"上存在歧义（翻过来的码重投影误差
    同样很小），因此这里只用它给出目标在相机系下的位姿，不在元数据里声明
    板面内的 x/y 朝向。
    """
    import cv2

    det = det or build_detector(board_cfg)
    if det["type"] == "checkerboard":
        return _detect_checkerboard(image_bgr, board_cfg, camera_model, det)
    dictionary = det["dictionary"]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=det["params"])
    h, w = gray.shape[:2]
    out: Dict[str, Any] = {"found": False, "image_size": [int(w), int(h)]}
    if ids is None or len(ids) == 0:
        # 字典与实物板不一致时最常见的结果就是"一个码都检测不到"，
        # 所以这里也要把当前字典报出来，否则只会看到一句无信息量的失败。
        out["reason"] = (
            f"未检测到任何 ArUco 码（当前字典 {det['dict_name']}，"
            f"期望 id={board_cfg.get('marker_id')}）。"
            "若实物板是别的字典（官方教程用 Original ArUco），"
            "请核对 handeye.board.dictionary —— 同一 id 在不同字典下图案不同"
        )
        out["dictionary"] = det["dict_name"]
        return out

    K = np.array(
        [
            [camera_model.color_intrinsics["fx"], 0.0, camera_model.color_intrinsics["ppx"]],
            [0.0, camera_model.color_intrinsics["fy"], camera_model.color_intrinsics["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    D = np.asarray(camera_model.color_intrinsics["coeffs"], dtype=float).reshape(-1, 1)

    wanted = board_cfg.get("marker_id")
    chosen = None
    for idx, mid in enumerate(ids.flatten()):
        if wanted is None or int(mid) == int(wanted):
            chosen = idx
            break
    if chosen is None:
        detected = [int(v) for v in ids.flatten()]
        # 字典配错时典型表现：检测到的是"别的 id"，或干脆一个都检测不到。
        # 同一个 id 在不同字典下是不同的图案，因此这里要把字典列为可疑项。
        out["reason"] = (
            f"检测到 {detected}，但没有 id={wanted}"
            f"（当前字典 {det['dict_name']}）。"
            "若标定板实际用的是别的字典（官方教程用 Original ArUco），请核对 "
            "handeye.board.dictionary —— 同一 id 在不同字典下图案不同"
        )
        out["detected_ids"] = detected
        out["dictionary"] = det["dict_name"]
        return out

    size = float(board_cfg["marker_size_m"])
    s = size / 2.0
    objp = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], dtype=float)
    imgp = corners[chosen].reshape(4, 2).astype(float)

    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        out["reason"] = "solvePnP 失败"
        return out
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
    reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - imgp) ** 2, axis=1))))
    R, _ = cv2.Rodrigues(rvec)

    # 角点面积与是否贴边，用于判断检测质量
    area = float(abs(cv2.contourArea(imgp.astype(np.float32))))
    margin = float(min(imgp[:, 0].min(), imgp[:, 1].min(), w - 1 - imgp[:, 0].max(), h - 1 - imgp[:, 1].max()))

    out.update(
        {
            "found": True,
            "marker_id": int(ids.flatten()[chosen]),
            "detected_ids": [int(v) for v in ids.flatten()],
            "marker_size_m": size,
            "rvec": [float(v) for v in rvec.reshape(3)],
            "tvec_m": [float(v) for v in tvec.reshape(3)],
            "T_cam_target": matrix_to_list(Transform.from_rt(R, tvec.reshape(3))),
            "corner_reproj_error_px": reproj,
            "corner_area_px2": area,
            "border_margin_px": margin,
            "target_frame": (
                f"aruco_marker_{int(ids.flatten()[chosen])} "
                "(OpenCV/ArUco 约定：原点在码中心，+z 由板面指向相机)"
            ),
            "quality_ok": bool(reproj < 1.0 and margin > 2.0 and area > 400.0),
        }
    )
    return out


# --------------------------------------------------------------------------- 采样


def _static_check(reader: RobotReader, window_ms: float = 400.0) -> Dict[str, Any]:
    """用最近 window_ms 内的反馈判断机械臂是否静止。"""
    now = time.time_ns()
    win = reader.snapshot()
    sel = [s for s in win if abs(s.joints_host_recv_ns - now) <= window_ms * 1e6]
    if len(sel) < 3:
        return {"enough_samples": False, "samples": len(sel), "static": None}
    arr = np.asarray([s.joints_rad for s in sel], dtype=float)
    spread = arr.max(axis=0) - arr.min(axis=0)
    max_spread_deg = float(np.degrees(np.max(spread)))
    return {
        "enough_samples": True,
        "samples": len(sel),
        "window_ms": window_ms,
        "max_joint_spread_deg": max_spread_deg,
        "static": bool(max_spread_deg < 0.05),
        "threshold_deg": 0.05,
    }


def sample(
    cfg: Dict[str, Any],
    root: Path,
    *,
    session_id: Optional[str] = None,
    notes: Optional[str] = None,
    color_format: Optional[str] = None,
    resume: bool = False,
    preview: bool = False,
    auto_seconds: Optional[float] = None,
    prompt_fn: Any = None,
    print_fn: Any = print,
) -> Dict[str, Any]:
    """交互式采样：人工调整姿态 → 回车记录一对样本。

    自动规划/扫描运动**不在本项目范围内**：这里只按回车记录当前姿态。
    """
    root = Path(root)
    session_id = session_id or f"he-{_utc_stamp()}"
    sdir = session_dir(root, session_id)
    existing_meta = None
    existing_samples = []
    if resume:
        existing_meta, existing_samples = load_session(root, session_id)
        if existing_meta.get("board") != cfg["handeye"]["board"] or existing_meta.get("mode") != cfg["handeye"]["mode"]:
            raise ValueError("续采模式或标定板配置与原会话不一致")
        if existing_meta.get("ee_frame_for_solve") != cfg["handeye"].get("ee_frame_for_solve", "link6"):
            raise ValueError("续采末端参考坐标系与原会话不一致")
    elif (sdir / "session.json").exists() or any(sdir.glob("sample_*.json")):
        raise ValueError(f"采样会话已存在，请使用新 session 名称: {session_id}")
    (sdir / "images").mkdir(parents=True, exist_ok=True)
    board_cfg = cfg["handeye"]["board"]
    build_detector(board_cfg)
    solve_frame = cfg["handeye"].get("ee_frame_for_solve", "link6")
    if solve_frame != "link6":
        raise ValueError("采样统一使用 link6 正运动学，请设 ee_frame_for_solve=link6")
    prompt_fn = prompt_fn or input

    cam_cfg = cfg["camera"]
    camera = RealsenseCamera(
        serial=cam_cfg.get("serial"),
        color_width=cam_cfg["color"]["width"],
        color_height=cam_cfg["color"]["height"],
        color_fps=cam_cfg["color"]["fps"],
        color_format=color_format or cam_cfg["color"]["format"],
        depth_width=cam_cfg["depth"]["width"],
        depth_height=cam_cfg["depth"]["height"],
        depth_fps=cam_cfg["depth"]["fps"],
        depth_format=cam_cfg["depth"]["format"],
        allow_spec_downgrade=False,
        warmup_frames=int(cam_cfg.get("warmup_frames", 30)),
        calibration_id=cam_cfg.get("calibration_id"),
    )
    camera_model = camera.open()
    if existing_meta:
        old_camera = existing_meta["camera"]
        current_camera = camera_model.to_dict()
        if any(old_camera.get(k) != current_camera.get(k) for k in ("calibration_id", "color_intrinsics")):
            camera.close()
            raise ValueError("续采相机或内参与原会话不一致")
    CalibrationStore(root).save("camera", camera_model.calibration_id, camera_model.to_dict())

    rc = cfg["robot"]
    reader = RobotReader(
        can_interface=rc["can_interface"],
        dh_is_offset=int(rc["dh_is_offset"]),
        poll_hz=float(rc.get("poll_hz", 200.0)),
        queries_on_connect=True,
        feedback_timeout_s=float(rc.get("feedback_timeout_s", 1.0)),
        tool_offset_m=rc.get("tool_offset_m", [0.0, 0.0, 0.0]),
        ee_frame=rc.get("ee_frame", "link6"),
        base_frame=rc.get("base_frame", "piper_base_link"),
        sdk_joint_limit=bool(rc.get("sdk_joint_limit", False)),
        sdk_gripper_limit=bool(rc.get("sdk_gripper_limit", False)),
    )
    if not reader.open(timeout_s=5.0):
        camera.close()
        return {"status": "failed", "reason": reader.open_error, "session_id": session_id}

    session_meta = {
        "schema_version": SCHEMA_VERSION,
        "kind": "handeye_session",
        "session_id": session_id,
        "mode": cfg["handeye"]["mode"],
        "started_at": _utc_now(),
        "board": dict(board_cfg),
        "board_record": {
            "type": board_cfg.get("type"),
            "marker_id": board_cfg.get("marker_id"),
            "dictionary": board_cfg.get("dictionary"),
            "marker_size_m": board_cfg.get("marker_size_m"),
            "inner_corners": board_cfg.get("inner_corners"),
            "square_size_m": board_cfg.get("square_size_m"),
            "length_unit": board_cfg.get("length_unit", "m"),
            "source_of_size": board_cfg.get("size_source", "official_example_unmeasured"),
        },
        "camera": camera_model.to_dict(),
        "kinematics": reader.fk.describe(),
        "ee_frame_for_solve": cfg["handeye"].get("ee_frame_for_solve", "link6"),
        "camera_optical_frame": cfg["handeye"].get("camera_optical_frame", "camera_color_optical_frame"),
        "motion_commands_sent": False,
        "notes": notes,
        "samples": [],
    }

    if existing_meta:
        session_meta = existing_meta
        session_meta.setdefault("resumed_at", []).append(_utc_now())
        session_meta.pop("ended_at", None)
    samples: List[Dict[str, Any]] = existing_samples
    index = max((s["sample_index"] for s in samples), default=-1) + 1
    write_json(sdir / "session.json", session_meta)
    try:
        if preview:
            from .handeye_preview import PreviewCamera
            camera = PreviewCamera(camera)
            prompt_fn = camera.prompt
            print_fn = camera.report
            print_fn(f"预览地址：{camera.url}；会话：{session_id}；已有 {len(samples)} 帧")
        while True:
            if auto_seconds is None:
                key = str(prompt_fn(
                    f"[{len(samples)} 个有效样本] 调整姿态并静止后回车=记录；d=删除上一个；q=保存退出 > "
                )).strip().lower()
            else:
                key = ""
                time.sleep(auto_seconds)
            if key in ("q", "quit", "exit"):
                break
            if key == "d":
                if samples:
                    removed = samples.pop()
                    p = sdir / removed["image"]
                    if p.is_file():
                        p.unlink()
                    (sdir / f"sample_{removed['sample_index']:04d}.json").unlink(missing_ok=True)
                    print_fn(f"已删除样本 {removed['sample_index']}，剩余 {len(samples)}")
                continue
            if key:
                print_fn("请输入回车、d 或 q")
                continue

            # Discard buffered images from before the operator pressed Enter.
            # Keep reading for one static-check window, then pair with the
            # nearest received joint feedback rather than a pre-prompt state.
            deadline = time.monotonic() + 0.45
            frame = None
            while time.monotonic() < deadline:
                frame = camera.read()
            if frame is None:
                print_fn("取帧超时，跳过本次")
                continue
            states = reader.snapshot()
            st = min(states, key=lambda s: abs(s.joints_host_recv_ns - frame.frameset_host_recv_ns)) if states else None

            det = detect_board(frame.color_bgr, board_cfg, camera_model)
            static = _static_check(reader)
            if st is None:
                print_fn("没有机械臂反馈，丢弃本次")
                continue
            dt_ms = abs(st.joints_host_recv_ns - frame.frameset_host_recv_ns) / 1e6
            if dt_ms > float(cfg["capture"]["sync"]["robot_tolerance_ms"]):
                print_fn(f"图像与关节反馈时间差 {dt_ms:.1f} ms 超限，丢弃本次")
                continue
            if not det.get("found"):
                print_fn(f"标定板未检出（{det.get('reason')}），丢弃本次")
                continue
            if not det.get("quality_ok"):
                print_fn(
                    f"检测质量不足：reproj={det['corner_reproj_error_px']:.2f}px "
                    f"margin={det['border_margin_px']:.1f}px area={det['corner_area_px2']:.0f}px²，丢弃本次"
                )
                continue
            if static.get("static") is not True:
                print_fn(f"未获得足够的静止反馈：{static}，丢弃本次")
                continue

            T_base_ee = reader.fk.fk_base_link6(st.joints_rad)
            sample = {
                "schema_version": SCHEMA_VERSION,
                "session_id": session_id,
                "sample_index": index,
                "captured_at": _utc_now(),
                "image": f"images/color_{index:04d}.png",
                "image_size": det["image_size"],
                "detection": det,
                "robot": {
                    "robot_state_id": st.state_id,
                    "joint_names": list(JOINT_NAMES),
                    "joint_positions_rad": [float(v) for v in st.joints_rad],
                    "joints_host_recv_ns": int(st.joints_host_recv_ns),
                    "joints_source_timestamp_ns": st.joints_source_timestamp_ns,
                    "joints_clock_source": st.joints_clock_source,
                    "ee_pose": list(st.ee_pose) if st.ee_pose else None,
                    "ee_pose_frame": st.ee_pose_frame,
                    "ee_pose_source": st.ee_pose_source,
                    "T_base_ee": matrix_to_list(T_base_ee),
                },
                "timing": {
                    "image_device_ts_ms": frame.color_device_ts_ms,
                    "image_ts_domain": frame.color_ts_domain,
                    "image_host_recv_ns": frame.frameset_host_recv_ns,
                    "robot_host_recv_ns": int(st.joints_host_recv_ns),
                    "image_vs_robot_dt_ms": (int(st.joints_host_recv_ns) - frame.frameset_host_recv_ns) / 1e6,
                    "clock_note": "图像为设备时间戳+主机接收时间；关节只有主机接收时间，属软件时间匹配",
                    "static_check": static,
                },
            }
            import cv2

            if not cv2.imwrite(str(sdir / sample["image"]), frame.color_bgr):
                raise OSError("保存标定图像失败")
            write_json(sdir / f"sample_{index:04d}.json", sample)
            samples.append(sample)
            index += 1
            print_fn(
                f"已记录样本 {sample['sample_index']}："
                f"T_cam_target t={np.round(det['tvec_m'], 4).tolist()} m，"
                f"距板 {np.linalg.norm(det['tvec_m']):.3f} m"
            )
    except (KeyboardInterrupt, EOFError):
        print_fn("采样结束，保存已完成的样本")
    finally:
        camera.close()
        reader.close()

    session_meta["samples"] = [
        {
            "sample_index": s["sample_index"],
            "image": s["image"],
            "detection_found": s["detection"]["found"],
            "robot_state_id": s["robot"]["robot_state_id"],
        }
        for s in samples
    ]
    session_meta["ended_at"] = _utc_now()
    session_meta["sample_count"] = len(samples)
    write_json(sdir / "session.json", session_meta)
    return {
        "status": "ok",
        "session_id": session_id,
        "session_dir": str(sdir),
        "samples": len(samples),
        "session_meta": session_meta,
    }


# --------------------------------------------------------------------------- 样本读取与检查


def load_session(root: Path, session_id: str, redetect: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import cv2

    sdir = session_dir(root, session_id)
    meta_path = sdir / "session.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"会话不存在: {meta_path}")
    from .jsonio import read_json

    meta = read_json(meta_path)
    samples: List[Dict[str, Any]] = []
    for p in sorted(sdir.glob("sample_*.json")):
        samples.append(read_json(p))

    if redetect:
        cam = meta["camera"]
        board_cfg = meta["board"]

        class _Cam:
            color_intrinsics = cam["color_intrinsics"]

        det = build_detector(board_cfg)
        for s in samples:
            img = cv2.imread(str(sdir / s["image"]), cv2.IMREAD_COLOR)
            if img is None:
                s["detection"] = {"found": False, "reason": "离线重检测：图像缺失"}
                continue
            s["detection"] = detect_board(img, board_cfg, _Cam(), det)
            s["detection"]["redetected_offline"] = True
    return meta, samples


def _valid_transform(value: Any) -> bool:
    try:
        T = np.asarray(value, dtype=float)
        return bool(
            T.shape == (4, 4) and np.isfinite(T).all()
            and np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
            and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
            and abs(np.linalg.det(T[:3, :3]) - 1) < 1e-5
        )
    except (TypeError, ValueError):
        return False


def _sample_error(sample: Dict[str, Any]) -> Optional[str]:
    det = sample.get("detection") or {}
    if not det.get("found") or not det.get("quality_ok"):
        return det.get("reason") or "检测质量不足"
    if not _valid_transform((sample.get("robot") or {}).get("T_base_ee")):
        return "缺少有效 robot.T_base_ee"
    if not _valid_transform(det.get("T_cam_target")):
        return "缺少有效 detection.T_cam_target"
    return None


def _usable_samples(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in samples if _sample_error(s) is None]


def check_samples(samples: Sequence[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """检查求解实际使用的矩阵，拒绝静止及仅绕一个轴旋转的退化数据。"""
    import cv2

    he = cfg["handeye"]
    usable = _usable_samples(samples)
    minimum = int(he.get("min_samples", 12))
    checks = {
        "total_samples": len(samples), "usable_samples": len(usable),
        "min_samples_required": minimum, "count_ok": len(usable) >= minimum,
        "rejected": [{"sample_index": s.get("sample_index"), "reason": _sample_error(s)}
                     for s in samples if _sample_error(s) is not None],
    }
    Ts = [np.asarray(s["robot"]["T_base_ee"], dtype=float) for s in usable]
    max_ang = 0.0
    rotvecs = []
    for i in range(len(Ts)):
        for j in range(i + 1, len(Ts)):
            relative = Ts[i][:3, :3].T @ Ts[j][:3, :3]
            v = cv2.Rodrigues(relative)[0].reshape(3)
            max_ang = max(max_ang, float(np.degrees(np.linalg.norm(v))))
            rotvecs.append(v)
    singular = np.linalg.svd(np.asarray(rotvecs), compute_uv=False) if rotvecs else np.zeros(3)
    axis_ratio = float(singular[1] / singular[0]) if len(singular) > 1 and singular[0] > 1e-8 else 0.0
    checks["ee_rotation_span_deg"] = {
        "max_pairwise_deg": max_ang, "required_min_deg": float(he.get("min_rotation_span_deg", 30)),
        "ok": max_ang >= float(he.get("min_rotation_span_deg", 30)),
    }
    checks["rotation_axis_diversity"] = {
        "second_to_first_singular_ratio": axis_ratio,
        "required_min_ratio": 0.05, "ok": axis_ratio >= 0.05,
    }
    if Ts:
        span = np.ptp(np.asarray(Ts)[:, :3, 3], axis=0)
        checks["ee_pose_span_m"] = dict(zip(("x", "y", "z"), map(float, span)))
        checks["ee_pose_span_m"]["norm"] = float(np.linalg.norm(span))
        distances = [np.linalg.norm(np.asarray(s["detection"]["T_cam_target"])[:3, 3]) for s in usable]
        checks["target_distance_m"] = {"min": float(min(distances)), "max": float(max(distances))}
    checks["ready_to_solve"] = bool(checks["count_ok"] and checks["ee_rotation_span_deg"]["ok"]
                                    and checks["rotation_axis_diversity"]["ok"])
    return checks


# --------------------------------------------------------------------------- 求解


def _avg_transform(Ts: np.ndarray) -> np.ndarray:
    """一组 4x4 的平均：平移取均值，旋转取四元数的主特征向量。"""
    t = Ts[:, :3, 3].mean(axis=0)
    A = np.zeros((4, 4))
    for T in Ts:
        q = np.asarray(matrix_to_quat(T[:3, :3]), dtype=float)
        A += np.outer(q, q)
    w, v = np.linalg.eigh(A)
    q = v[:, int(np.argmax(w))]
    if q[0] < 0:
        q = -q
    return Transform.from_rt(quat_to_matrix(normalize_quat(q)), t)


def _consistency(T_robot: Sequence[np.ndarray], T_cam_target: Sequence[np.ndarray], X: np.ndarray) -> Dict[str, Any]:
    """给定 OpenCV 输入侧位姿与 X，计算固定目标变换的一致性误差。"""
    if not T_robot:
        return {"n": 0}
    if len(T_robot) != len(T_cam_target):
        raise ValueError(f"一致性检查输入长度不一致: robot={len(T_robot)}, target={len(T_cam_target)}")
    Ts = np.asarray([A @ X @ B for A, B in zip(T_robot, T_cam_target)])
    ref = _avg_transform(Ts)
    pos = np.linalg.norm(Ts[:, :3, 3] - ref[:3, 3], axis=1) * 1000.0  # mm
    ang = np.asarray([Transform.rotation_angle_deg(ref[:3, :3], T[:3, :3]) for T in Ts])
    return {
        "n": int(len(Ts)),
        "position_error_rms_mm": float(np.sqrt(np.mean(pos ** 2))),
        "position_error_max_mm": float(np.max(pos)),
        "rotation_error_rms_deg": float(np.sqrt(np.mean(ang ** 2))),
        "rotation_error_max_deg": float(np.max(ang)),
        "T_base_target_mean": matrix_to_list(ref),
        "position_error_per_sample_mm": [float(v) for v in pos],
        "rotation_error_per_sample_deg": [float(v) for v in ang],
    }


def _split_holdout(indices: List[int], ratio: float) -> Tuple[List[int], List[int]]:
    n = len(indices)
    n_hold = max(1, int(round(n * ratio))) if n >= 4 else 0
    if n_hold == 0:
        return indices, []
    step = n / float(n_hold)
    hold = sorted({indices[min(n - 1, int(round((k + 0.5) * step)))] for k in range(n_hold)})
    hold = sorted(set(hold))
    train = [i for i in indices if i not in set(hold)]
    if not train:
        return indices, []
    return train, hold


def solve(
    cfg: Dict[str, Any],
    root: Path,
    *,
    session_id: str,
    calibration_id: Optional[str] = None,
    verify_session_id: Optional[str] = None,
    solve_method: Optional[str] = None,
    redetect: bool = False,
    save: bool = True,
) -> Dict[str, Any]:
    import cv2

    root = Path(root)
    he = cfg["handeye"]
    meta, samples = load_session(root, session_id, redetect=redetect)
    checks = check_samples(samples, cfg)
    if not checks["count_ok"]:
        return {
            "status": "pending",
            "reason": f"有效样本 {checks['usable_samples']} 少于要求 {checks['min_samples_required']}",
            "session_id": session_id,
            "sample_checks": checks,
        }
    if not checks["ready_to_solve"]:
        return {
            "status": "pending",
            "reason": (
                "末端旋转跨度或旋转轴多样性不足，请绕不同轴改变姿态："
                f"{checks['ee_rotation_span_deg']['max_pairwise_deg']:.2f}° < "
                f"{checks['ee_rotation_span_deg']['required_min_deg']:.2f}°"
            ),
            "session_id": session_id,
            "sample_checks": checks,
        }

    usable = _usable_samples(samples)
    indices = list(range(len(usable)))
    verify_meta = None
    holdout_samples: List[Dict[str, Any]] = []
    if verify_session_id:
        verify_meta, verify_all = load_session(root, verify_session_id, redetect=redetect)
        board_a = meta.get("board_record", meta.get("board", {}))
        board_b = verify_meta.get("board_record", verify_meta.get("board", {}))
        board_keys = (("type", "inner_corners", "square_size_m")
                      if board_a.get("type") == "checkerboard"
                      else ("type", "dictionary", "marker_id", "marker_size_m"))
        if any(board_a.get(k) != board_b.get(k) for k in board_keys):
            return {
                "status": "invalid",
                "reason": "求解会话与验证会话的标定板配置不一致",
                "session_id": session_id,
                "verify_session_id": verify_session_id,
            }
        if meta.get("camera", {}).get("calibration_id") != verify_meta.get("camera", {}).get("calibration_id"):
            return {
                "status": "invalid",
                "reason": "求解会话与验证会话的相机内参版本不一致",
                "session_id": session_id,
                "verify_session_id": verify_session_id,
            }
        holdout_samples = _usable_samples(verify_all)
        train_idx, hold_idx = indices, []
    else:
        train_idx, hold_idx = _split_holdout(indices, float(he.get("verify_holdout_ratio", 0.25)))
        holdout_samples = [usable[i] for i in hold_idx]

    train = [usable[i] for i in train_idx]

    mode = str(he.get("mode", "eye_in_hand"))
    if mode not in ("eye_in_hand", "eye_to_hand"):
        raise ValueError(f"不支持的手眼模式: {mode}")

    def _prep(ss: Sequence[Dict[str, Any]]):
        Rg, tg, Rt, tt = [], [], [], []
        for s in ss:
            Tbe = np.asarray(s["robot"]["T_base_ee"], dtype=float)
            Tct = np.asarray(s["detection"]["T_cam_target"], dtype=float)
            # The official ROS package inverts T_base_ee for eye-to-hand
            # before passing it to cv2.calibrateHandEye.  OpenCV then returns
            # T_base_camera; for eye-in-hand it returns T_ee_camera.
            if mode == "eye_to_hand":
                Tbe = Transform.invert(Tbe)
            Rg.append(Tbe[:3, :3])
            tg.append(Tbe[:3, 3])
            Rt.append(Tct[:3, :3])
            tt.append(Tct[:3, 3])
        return Rg, tg, Rt, tt

    Rg, tg, Rt, tt = _prep(train)
    methods = [solve_method] if solve_method else list(he.get("compare_methods", ["TSAI"]))
    results: Dict[str, Any] = {}
    best = None
    for name in methods:
        flag_name = SOLVE_METHODS.get(name.upper())
        if flag_name is None:
            results[name] = {"error": f"未知方法 {name}，可用: {sorted(SOLVE_METHODS)}"}
            continue
        try:
            R, t = cv2.calibrateHandEye(
                Rg, tg, Rt, tt, method=getattr(cv2, flag_name)
            )
        except Exception as exc:  # pragma: no cover - 依赖 OpenCV 版本
            results[name] = {"error": str(exc)}
            continue
        X = Transform.from_rt(R, t.reshape(3))
        if not _valid_transform(X):
            results[name] = {"error": "OpenCV 返回非有限数值，样本可能退化"}
            continue
        consistency_robot = [
            Transform.invert(np.asarray(s["robot"]["T_base_ee"], dtype=float))
            if mode == "eye_to_hand" else np.asarray(s["robot"]["T_base_ee"], dtype=float)
            for s in train
        ]
        hold_consistency_robot = [
            Transform.invert(np.asarray(s["robot"]["T_base_ee"], dtype=float))
            if mode == "eye_to_hand" else np.asarray(s["robot"]["T_base_ee"], dtype=float)
            for s in holdout_samples
        ]
        train_cons = _consistency(consistency_robot,
                                  [np.asarray(s["detection"]["T_cam_target"]) for s in train], X)
        hold_cons = (
            _consistency(hold_consistency_robot,
                         [np.asarray(s["detection"]["T_cam_target"]) for s in holdout_samples], X)
            if holdout_samples
            else {"n": 0}
        )
        result_transform = matrix_to_list(X)
        results[name] = {
            "transform_name": "T_ee_camera" if mode == "eye_in_hand" else "T_base_camera",
            "transform": result_transform,
            # Keep the original key for consumers of the eye-in-hand API.
            # New code should use transform_name + transform so eye-to-hand
            # results cannot be mistaken for T_ee_camera.
            "T_ee_camera": result_transform if mode == "eye_in_hand" else None,
            "T_base_camera": result_transform if mode == "eye_to_hand" else None,
            "train_consistency": train_cons,
            "holdout_consistency": hold_cons,
        }
        score_cons = train_cons  # 方法只按训练误差选择，留出样本仅用于验收
        score = (
            (score_cons.get("position_error_rms_mm", float("inf")) / max(float(he.get("verify_max_position_rms_mm", 10.0)), 1e-9)) ** 2
            + (score_cons.get("rotation_error_rms_deg", float("inf")) / max(float(he.get("verify_max_rotation_rms_deg", 2.0)), 1e-9)) ** 2
        )
        if not np.isfinite(score):
            results[name]["error"] = "一致性误差不是有限数值"
            continue
        if best is None or score < best[1]:
            best = (name, score, X, train_cons, hold_cons)

    if best is None:
        return {"status": "invalid", "reason": "所有求解方法都失败", "session_id": session_id, "methods": results}

    method_name, _, X_solved, train_cons, hold_cons = best

    # ---- 与数据集 EE 坐标系对齐 ----
    ee_frame_for_solve = he.get("ee_frame_for_solve", "link6")
    ds_ee_frame = cfg["robot"].get("ee_frame", "link6")
    tool_offset = [float(v) for v in cfg["robot"].get("tool_offset_m", [0.0, 0.0, 0.0])]
    frame_conversion: Dict[str, Any] = {
        "solved_frame": ee_frame_for_solve,
        "dataset_ee_frame": ds_ee_frame,
        "converted": False,
        "note": "T_ee_camera 的下标必须与数据集 ee_pose 的 EE 坐标系一致，否则不能混用",
    }
    T_ee_cam = X_solved
    if mode == "eye_in_hand" and ee_frame_for_solve == "link6" and ds_ee_frame != "link6":
        # T_dsEE_camera = T_dsEE_link6 @ T_link6_camera = inv(T_link6_dsEE) @ X
        T_link6_dsEE = Transform.from_rt(np.eye(3), tool_offset)
        T_ee_cam = Transform.invert(T_link6_dsEE) @ X_solved
        frame_conversion.update(
            {
                "converted": True,
                "conversion": "T_dsEE_camera = inv(T_link6_dsEE) @ T_link6_camera",
                "T_link6_dsEE": matrix_to_list(T_link6_dsEE),
                "T_link6_camera": matrix_to_list(X_solved),
            }
        )

    # ---- 状态判定 ----
    max_pos_rms = float(he.get("verify_max_position_rms_mm", 10.0))
    max_rot_rms = float(he.get("verify_max_rotation_rms_deg", 2.0))
    if hold_cons.get("n", 0) < 2:
        status = "pending"
        status_reason = "留出验证样本不足（<2），无法给出可量化验证误差"
    elif hold_cons["position_error_rms_mm"] <= max_pos_rms and hold_cons["rotation_error_rms_deg"] <= max_rot_rms:
        status = "valid"
        status_reason = None
    else:
        status = "invalid"
        status_reason = (
            f"留出验证误差超阈值：位置 RMS {hold_cons['position_error_rms_mm']:.2f} mm (>{max_pos_rms} mm) "
            f"或旋转 RMS {hold_cons['rotation_error_rms_deg']:.2f}° (>{max_rot_rms}°)"
        )

    calibration_id = calibration_id or f"handeye-{session_id}"
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "handeye",
        "calibration_id": calibration_id,
        "status": status,
        "valid": status == "valid",
        "status_reason": status_reason,
        "mode": mode,
        "algorithm": {
            "solver": "cv2.calibrateHandEye",
            "method_used": method_name,
            "methods_compared": results,
            "opencv_version": getattr(cv2, "__version__", None),
            "numpy_version": np.__version__,
        },
        "transform": {
            "name": "T_ee_camera" if mode == "eye_in_hand" else "T_base_camera",
            "semantics": (
                "p_ee = T_ee_camera @ p_camera" if mode == "eye_in_hand"
                else "p_base = T_base_camera @ p_camera"
            ),
            "parent_frame": (
                (ee_frame_for_solve if not frame_conversion["converted"] else ds_ee_frame)
                if mode == "eye_in_hand" else cfg["robot"].get("base_frame", "piper_base_link")
            ),
            "child_frame": he.get("camera_optical_frame", "camera_color_optical_frame"),
            "matrix_row_major": matrix_to_list(T_ee_cam),
            "translation_m": [float(v) for v in T_ee_cam[:3, 3]],
            "translation_unit": "m",
            "matrix_direction": "p_parent = T_parent_child @ p_child",
            "camera_optical_frame_convention": (
                "camera_color_optical_frame：+x 右、+y 下、+z 前（RGB 光学系）"
            ),
            "camera_stream_used": "color (RGB)",
            "frame_conversion": frame_conversion,
        },
        "board": meta.get("board_record", meta.get("board", {})),
        "camera_calibration_id": meta["camera"]["calibration_id"],
        "session_id": session_id,
        "verify_session_id": verify_session_id,
        "sample_checks": checks,
        "samples_used": {
            "train_count": len(train),
            "holdout_count": len(holdout_samples),
            "train_sample_index": [s["sample_index"] for s in train],
            "holdout_sample_index": [s["sample_index"] for s in holdout_samples],
            "holdout_source": verify_session_id or "in_session_split",
        },
        "train_consistency": train_cons,
        "verification": hold_cons,
        "thresholds": {
            "verify_max_position_rms_mm": max_pos_rms,
            "verify_max_rotation_rms_deg": max_rot_rms,
        },
        "sampling_times": {
            "session_started_at": meta.get("started_at"),
            "session_ended_at": meta.get("ended_at"),
        },
        "solve_config": {
            "min_samples": he.get("min_samples"),
            "verify_holdout_ratio": he.get("verify_holdout_ratio"),
            "ee_frame_for_solve": ee_frame_for_solve,
            "note": (
                "T_cam_target 由标定板检测得到，T_base_ee 由反馈关节角正向运动学得到；"
                "eye_to_hand 按官方 ROS 实现先使用 T_ee_base"
            ),
        },
        "dependencies": {
            "opencv": getattr(cv2, "__version__", None),
            "numpy": np.__version__,
            "note": "官方参考 agilexrobotics/handeye_calibration_ros（cv2.calibrateHandEye）；本项目为其非 ROS 适配实现",
        },
        "notes": [
            "标定板尺寸采用会话记录值；示例/标签尺寸未测量时来源明确标记为未测量",
            "验证误差来自留出姿态上 T_base_target 的离散程度，不依赖标定板真值",
            "求解使用的是反馈关节角 FK 得到的 T_base_ee，不是目标关节角",
        ],
    }
    if status != "valid":
        payload["dataset_use_policy"] = (
            "status != valid 时数据集不得引用该标定作为可用变换；"
            "字段保留矩阵仅为追溯，不代表已通过验证"
        )
    if save:
        path = CalibrationStore(root).save("handeye", calibration_id, payload)
        payload["saved_path"] = str(path)
    return payload


def verify(
    cfg: Dict[str, Any],
    root: Path,
    *,
    calibration_id: str,
    session_id: str,
    report_id: Optional[str] = None,
) -> Dict[str, Any]:
    """用指定会话的样本验证已有标定，输出独立验证报告。"""
    root = Path(root)
    store = CalibrationStore(root)
    cal = store.load("handeye", calibration_id)
    X = np.asarray(cal["transform"]["matrix_row_major"], dtype=float)
    _, samples = load_session(root, session_id)
    usable = _usable_samples(samples)
    mode = cal.get("mode", cfg["handeye"].get("mode", "eye_in_hand"))
    robot_transforms = [
        Transform.invert(np.asarray(s["robot"]["T_base_ee"], dtype=float))
        if mode == "eye_to_hand" else np.asarray(s["robot"]["T_base_ee"], dtype=float)
        for s in usable
    ]
    cons = _consistency(
        robot_transforms,
        [np.asarray(s["detection"]["T_cam_target"]) for s in usable],
        X,
    )
    max_pos_rms = float(cfg["handeye"].get("verify_max_position_rms_mm", 10.0))
    max_rot_rms = float(cfg["handeye"].get("verify_max_rotation_rms_deg", 2.0))
    n = cons.get("n", 0)
    if n < 2:
        status = "pending"
    elif cons["position_error_rms_mm"] <= max_pos_rms and cons["rotation_error_rms_deg"] <= max_rot_rms:
        status = "valid"
    else:
        status = "invalid"
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "handeye_verification",
        "report_id": report_id or f"verify-{_utc_stamp()}",
        "generated_at": _utc_now(),
        "calibration_id": calibration_id,
        "mode": mode,
        "verified_with_session": session_id,
        "samples_used": n,
        "consistency": cons,
        "thresholds": {"verify_max_position_rms_mm": max_pos_rms, "verify_max_rotation_rms_deg": max_rot_rms},
        "status": status,
        "note": "验证样本应为未参与求解的姿态；本报告独立于求解过程，可复现",
    }
    write_json(handeye_dir(root) / f"{report['report_id']}.verification.json", report)
    return report


def list_calibrations(root: Path) -> Dict[str, Any]:
    root = Path(root)
    store = CalibrationStore(root)
    out: Dict[str, Any] = {"handeye": [], "sessions": []}
    for cid in store.index()["handeye"]:
        try:
            c = store.load("handeye", cid)
            out["handeye"].append(
                {
                    "calibration_id": cid,
                    "status": c.get("status"),
                    "valid": c.get("valid"),
                    "method": (c.get("algorithm") or {}).get("method_used"),
                    "holdout_rms_mm": (c.get("verification") or {}).get("position_error_rms_mm"),
                    "saved_at": c.get("saved_at"),
                }
            )
        except Exception as exc:
            out["handeye"].append({"calibration_id": cid, "error": str(exc)})
    sdir = handeye_dir(root) / "sessions"
    if sdir.is_dir():
        for p in sorted(sdir.glob("*/session.json")):
            from .jsonio import read_json

            m = read_json(p)
            out["sessions"].append(
                {"session_id": m.get("session_id"), "samples": m.get("sample_count"), "started_at": m.get("started_at")}
            )
    return out
