"""命令行入口。

    python3 -m piper_capture.cli doctor
    python3 -m piper_capture.cli camera-probe
    python3 -m piper_capture.cli capture --scene scene-tabletop --duration 20
    python3 -m piper_capture.cli handeye sample --session he-001
    python3 -m piper_capture.cli handeye solve --session he-001
    python3 -m piper_capture.cli gripper diagnose --seconds 10
    python3 -m piper_capture.cli video add --role side_task --episode <id> --file a.mp4
    python3 -m piper_capture.cli quality check --episode <id>
    python3 -m piper_capture.cli verify-fk

所有命令默认只读；只有 `gripper probe --allow-motion` 会真实驱动夹爪。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import SCHEMA_VERSION
from .config import ConfigError, dataset_root as resolve_root, load_config
from .jsonio import write_json


def _can_status(iface: str) -> Dict[str, Any]:
    """只读检查 CAN 接口状态。不激活、不修改配置。

    优先用 `ip -details -statistics link show`（在受限网络命名空间下仍可用），
    退回 sysfs。
    """
    import re
    import subprocess

    evidence: Dict[str, Any] = {"interface": iface}
    text = ""
    try:
        p = subprocess.run(
            ["ip", "-details", "-statistics", "link", "show", iface],
            capture_output=True,
            text=True,
            timeout=10,
        )
        text = (p.stdout or "") + (p.stderr or "")
    except Exception as exc:
        evidence["ip_error"] = str(exc)

    if not text.strip():
        sysfs = Path("/sys/class/net") / iface
        if sysfs.is_dir():
            return {
                "status": "unknown",
                "detail": f"{iface} 在 sysfs 中存在，但无法读取详情（可能在受限命名空间内）",
                "evidence": evidence,
            }
        return {"status": "fail", "detail": f"{iface} 不存在", "evidence": evidence}

    if "does not exist" in text or "Cannot find device" in text:
        return {"status": "fail", "detail": f"{iface} 不存在", "evidence": evidence}

    evidence["ip_details"] = text.strip()
    m = re.search(r"can state (\S+)", text)
    state = m.group(1) if m else None
    m = re.search(r"bitrate (\d+)", text)
    bitrate = int(m.group(1)) if m else None
    m = re.search(r"RX:.*?\n\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", text, re.S)
    rx = {"bytes": int(m.group(1)), "packets": int(m.group(2)), "errors": int(m.group(3)), "dropped": int(m.group(4))} if m else None
    m = re.search(r"TX:.*?\n\s*(\d+)\s+(\d+)", text, re.S)
    tx = {"packets": int(m.group(2))} if m else None
    up = "state UP" in text.split("\n")[0] or ",UP," in text.split("\n")[0]
    evidence.update({"can_state": state, "bitrate": bitrate, "up": up, "rx": rx, "tx": tx})

    healthy = bool(up and state and state.upper() in ("ERROR-ACTIVE", "ERROR-WARN"))
    if healthy:
        # 接口健康 ≠ 总线上有节点在发帧。只读地比较 1.5 s 前后的 rx_packets。
        # 实测遇到过：接口 UP / ERROR-ACTIVE、错误计数全 0，但总线上一个反馈帧都没有
        # （机械臂未上电或 CAN 线未接），此时只看接口状态会误报 pass。
        rx0 = rx["packets"] if rx else None
        rx1 = None
        if rx0 is not None:
            import time

            time.sleep(1.5)
            try:
                p2 = subprocess.run(
                    ["ip", "-details", "-statistics", "link", "show", iface],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                m2 = re.search(
                    r"RX:.*?\n\s*(\d+)\s+(\d+)", (p2.stdout or "") + (p2.stderr or ""), re.S
                )
                rx1 = int(m2.group(2)) if m2 else None
            except Exception as exc:
                evidence["traffic_probe_error"] = str(exc)
        delta = (rx1 - rx0) if (rx0 is not None and rx1 is not None) else None
        evidence["rx_packets_delta_1_5s"] = delta

        if delta == 0:
            return {
                "status": "fail",
                "detail": f"{iface} 接口本身正常（UP / {state} / bitrate={bitrate}），但 1.5 秒内 "
                f"rx_packets 没有增加（{rx0} → {rx1}），即总线上没有节点在发帧。"
                "接口健康不等于机械臂在通信：请检查机械臂是否上电、CAN 线是否接到适配器、"
                "接线/终端电阻是否正常。本项目不会自行激活或修改 CAN 配置。",
                "evidence": evidence,
            }
        if delta is None:
            return {
                "status": "pass",
                "detail": f"UP / {state} / bitrate={bitrate} / rx_packets={rx0}（总线流量未探测）",
                "evidence": evidence,
            }
        return {
            "status": "pass",
            "detail": f"UP / {state} / bitrate={bitrate} / rx_packets={rx0}"
            f" / 1.5s 内 +{delta} 帧（≈{delta / 1.5:.1f} 帧/s）",
            "evidence": evidence,
        }
    reasons = []
    if not up:
        reasons.append("接口 DOWN")
    if state:
        reasons.append(f"can state {state}")
    if bitrate is None:
        reasons.append("未配置 bitrate")
    return {
        "status": "fail",
        "detail": f"{iface} 当前不可用：{'、'.join(reasons)}。"
        "不能用它读取机械臂反馈；本项目不会自行激活或修改 CAN 配置。"
        "注意：若在受限沙箱/容器内运行，CAN 接口可能不可见，请在宿主环境重试",
        "evidence": evidence,
    }


def _print(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False))


def _root(cfg: Dict[str, Any], args: argparse.Namespace) -> Path:
    if getattr(args, "dataset_root", None):
        cfg["dataset_root"] = args.dataset_root
    return resolve_root(cfg, Path.cwd())


def _cfg(args: argparse.Namespace) -> Dict[str, Any]:
    return load_config(getattr(args, "config", None))


# --------------------------------------------------------------------------- doctor


def cmd_doctor(args: argparse.Namespace) -> int:
    from .camera import RealsenseCamera
    from .kinematics import DH_TABLES, parse_urdf_chain, urdf_fk_link6, dh_fk_link6, sdk_cal_fk
    from .episode import DatasetManifest
    from .schema import JOINT_NAMES

    cfg = _cfg(args)
    root = _root(cfg, args)
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "doctor",
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "motion_commanded": False,
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "hostname": platform.node(),
        },
        "dataset_root": str(root),
        "dataset_exists": root.is_dir(),
        "checks": [],
    }

    def check(name: str, status: str, detail: Any, evidence: Any = None) -> None:
        report["checks"].append({"name": name, "status": status, "detail": detail, "evidence": evidence})

    # ROS
    import os

    ros_distro = os.environ.get("ROS_DISTRO")
    has_ros = Path("/opt/ros").is_dir() and any(Path("/opt/ros").iterdir())
    check(
        "ROS 环境",
        "info",
        f"ROS_DISTRO={ros_distro or '(空)'}，/opt/ros 存在={has_ros}",
        "本项目不依赖 ROS，使用 piper_sdk 直连 CAN + pyrealsense2；官方 ROS 示例仅作参考",
    )

    # SDK
    try:
        import piper_sdk

        from .robot import RobotReader  # noqa: F401

        check("piper_sdk", "pass", getattr(piper_sdk, "__version__", "unknown"), str(Path(piper_sdk.__file__).parent))
    except Exception as exc:
        check("piper_sdk", "fail", f"导入失败: {exc}")

    # 相机
    try:
        devs = RealsenseCamera.list_devices()
        check("RealSense 设备", "pass" if devs else "fail", f"{len(devs)} 台", devs)
        profiles = RealsenseCamera.supported_profiles(cfg["camera"].get("serial"))
        want_c = cfg["camera"]["color"]
        want_d = cfg["camera"]["depth"]
        ok_c = any(
            p["width"] == want_c["width"] and p["height"] == want_c["height"] and p["fps"] == want_c["fps"] and p["format"] == want_c["format"]
            for p in profiles["color"]
        )
        ok_d = any(
            p["width"] == want_d["width"] and p["height"] == want_d["height"] and p["fps"] == want_d["fps"] and p["format"] == want_d["format"]
            for p in profiles["depth"]
        )
        check(
            f"目标流配置 {want_c['width']}x{want_c['height']}@{want_c['fps']}",
            "pass" if (ok_c and ok_d) else "fail",
            f"color 支持={ok_c}, depth 支持={ok_d}",
            {"color_options_1280x720": [p for p in profiles["color"] if p["width"] == 1280 and p["height"] == 720],
             "depth_options_1280x720": [p for p in profiles["depth"] if p["width"] == 1280 and p["height"] == 720]},
        )
    except Exception as exc:
        check("RealSense 设备", "fail", f"探测失败: {exc}")

    # CAN
    can_info = _can_status(cfg["robot"]["can_interface"])
    check(
        f"CAN 接口 {cfg['robot']['can_interface']}",
        can_info["status"],
        can_info["detail"],
        can_info["evidence"],
    )
    if can_info["status"] != "pass":
        check(
            "CAN 激活提示",
            "info",
            "本项目不激活/不修改 CAN 配置；沿用现有方式激活后再采集",
            "官方方式：bash can_activate.sh can0 1000000（需要 sudo）。"
            "本项目不在代码里执行该命令，也不会改写接口配置。",
        )

    # 运动学三方交叉校验
    try:
        joints = [0.0] * 6
        dh01 = dh_fk_link6(joints, 0x01)
        dh00 = dh_fk_link6(joints, 0x00)
        sdk = sdk_cal_fk(joints, 0x01)
        import numpy as np
        from .schema import Transform

        sdk_T = Transform.from_rpy_xyz([v * 3.141592653589793 / 180.0 for v in sdk[5][3:]], [v / 1000.0 for v in sdk[5][:3]])
        err01 = float(np.linalg.norm(dh01[:3, 3] - sdk_T[:3, 3]) * 1000.0)
        err00 = float(np.linalg.norm(dh00[:3, 3] - sdk_T[:3, 3]) * 1000.0)
        urdf_path = Path(cfg["robot"]["urdf_path"])
        urdf_err = None
        if urdf_path.is_file():
            chain = parse_urdf_chain(urdf_path)
            urdf_err = float(np.linalg.norm(urdf_fk_link6(joints, chain)[:3, 3] - dh01[:3, 3]) * 1000.0)
        check(
            "正运动学交叉校验（零位）",
            "pass" if err01 < 0.5 else "fail",
            f"本项目 DH(0x01) vs piper_sdk = {err01:.4f} mm；DH(0x00) vs SDK = {err00:.4f} mm；"
            f"URDF vs DH(0x01) = {('%.4f mm' % urdf_err) if urdf_err is not None else '未找到官方 URDF，跳过'}",
            {"urdf_path": str(urdf_path), "urdf_found": urdf_path.is_file()},
        )
    except Exception as exc:
        check("正运动学交叉校验", "fail", str(exc))

    # 数据集
    check("数据集目录", "pass" if root.is_dir() else "info", str(root), DatasetManifest(root).load() if root.is_dir() else "尚未创建，首次采集会自动创建")

    out = root / "reports" / "doctor.json"
    write_json(out, report)
    report["report_path"] = str(out)
    _print(report)
    return 0 if not any(c["status"] == "fail" for c in report["checks"]) else 1


def cmd_camera_probe(args: argparse.Namespace) -> int:
    from .camera import RealsenseCamera

    cfg = _cfg(args)
    out: Dict[str, Any] = {
        "devices": RealsenseCamera.list_devices(),
        "supported_profiles": RealsenseCamera.supported_profiles(cfg["camera"].get("serial")),
        "requested": {"color": cfg["camera"]["color"], "depth": cfg["camera"]["depth"]},
    }
    if args.open:
        cam = RealsenseCamera(
            serial=cfg["camera"].get("serial"),
            color_width=cfg["camera"]["color"]["width"],
            color_height=cfg["camera"]["color"]["height"],
            color_fps=cfg["camera"]["color"]["fps"],
            color_format=cfg["camera"]["color"]["format"],
            depth_width=cfg["camera"]["depth"]["width"],
            depth_height=cfg["camera"]["depth"]["height"],
            depth_fps=cfg["camera"]["depth"]["fps"],
            depth_format=cfg["camera"]["depth"]["format"],
        )
        model = cam.open()
        out["opened_model"] = model.to_dict()
        import time

        t0 = time.time()
        n = 0
        while time.time() - t0 < args.seconds:
            if cam.read() is not None:
                n += 1
        out["measured_fps"] = n / max(1e-9, time.time() - t0)
        out["diagnostics"] = cam.diagnostics()
        cam.close()
    _print(out)
    return 0


# --------------------------------------------------------------------------- capture


def cmd_capture(args: argparse.Namespace) -> int:
    from .capture import CaptureSession

    cfg = _cfg(args)
    root = _root(cfg, args)
    if args.camera_only:
        cfg["robot"]["enabled"] = False
    session = CaptureSession(
        cfg,
        base_dir=Path.cwd(),
        scene_id=args.scene,
        episode_id=args.episode,
        duration_s=args.duration,
        camera_only=args.camera_only,
        notes=args.notes,
        progress=(lambda d: print(f"  ... {d['seq']} 样本, {d['rate']:.1f} Hz")) if args.verbose else None,
    )
    summary = session.run()
    _print(summary)
    return 0 if summary.get("status") in ("closed", "aborted") else 1


def cmd_capture_lerobot(args: argparse.Namespace) -> int:
    from .lerobot_writer import LeRobotCaptureSession

    cfg = _cfg(args)
    session = LeRobotCaptureSession(cfg, base_dir=Path.cwd(), duration_s=args.duration, task=args.task)
    summary = session.run()
    _print(summary)
    return 0 if summary.get("status") in ("closed", "aborted") else 1


def cmd_capture_lerobot_interactive(args: argparse.Namespace) -> int:
    from .interactive_capture import run_interactive

    return run_interactive(
        config=args.config,
        dataset_root=args.dataset_root,
        task=args.task,
        episodes=args.episodes,
    )


def cmd_camera_ui(args: argparse.Namespace) -> int:
    from .camera_tuner import serve

    source = args.config or "configs/lerobot_v3_two_d435i.json"
    output = args.output or str(Path(source).with_name(Path(source).stem + "_tuned.json"))
    return serve(source, output, args.port, not args.no_browser)


def cmd_verify_lerobot(args: argparse.Namespace) -> int:
    from .lerobot_writer import verify_lerobot_dataset

    result = verify_lerobot_dataset(Path(args.root), args.repo_id)
    _print(result)
    return 0


# --------------------------------------------------------------------------- handeye


def cmd_handeye(args: argparse.Namespace) -> int:
    from . import handeye as he

    cfg = _cfg(args)
    root = _root(cfg, args)
    if args.sub == "sample":
        res = he.sample(
            cfg,
            root,
            session_id=args.session,
            notes=args.notes,
            color_format=args.color_format,
            resume=args.resume,
            preview=args.preview,
        )
        _print({k: v for k, v in res.items() if k != "session_meta"})
        return 0 if res.get("status") == "ok" else 1
    if args.sub == "solve":
        res = he.solve(
            cfg,
            root,
            session_id=args.session,
            calibration_id=args.calibration_id,
            verify_session_id=args.verify_session,
            solve_method=args.method,
            redetect=args.redetect,
        )
        _print(res)
        return 0 if res.get("status") == "valid" else 2
    if args.sub == "verify":
        res = he.verify(cfg, root, calibration_id=args.calibration_id, session_id=args.session)
        _print(res)
        return 0 if res.get("status") == "valid" else 2
    if args.sub == "check":
        meta, samples = he.load_session(root, args.session, redetect=args.redetect)
        _print(he.check_samples(samples, cfg))
        return 0
    if args.sub == "list":
        _print(he.list_calibrations(root))
        return 0
    raise SystemExit("未知子命令")


# --------------------------------------------------------------------------- gripper


def cmd_gripper(args: argparse.Namespace) -> int:
    from . import gripper as gp

    cfg = _cfg(args)
    root = _root(cfg, args)
    if args.sub == "diagnose":
        res = gp.diagnose(cfg, root, duration_s=args.seconds)
        _print(res)
        return 0 if res.get("status") == "ok" else 1
    if args.sub == "probe":
        res = gp.probe(
            cfg,
            root,
            allow_motion=args.allow_motion,
            targets_mm=args.targets,
            dwell_s=args.dwell,
            effort_raw=args.effort,
            send_hz=args.send_hz,
        )
        _print(res)
        return 0
    if args.sub == "calibrate":
        points = []
        for spec in args.point:
            raw_s, mm_s = spec.split(":")
            points.append((float(raw_s), float(mm_s)))
        res = gp.calibrate(cfg, root, points=points, calibration_id=args.calibration_id, notes=args.notes)
        _print(res)
        return 0
    if args.sub == "pending":
        res = gp.pending_calibration(args.calibration_id, args.reason or "尚未完成实物多点测量")
        _print(res)
        return 0
    raise SystemExit("未知子命令")


# --------------------------------------------------------------------------- video


def cmd_video(args: argparse.Namespace) -> int:
    from . import external as ex

    cfg = _cfg(args)
    root = _root(cfg, args)
    if args.sub == "add":
        res = ex.register(
            root,
            Path(args.file),
            args.role,
            scene_id=args.scene,
            episode_id=args.episode,
            copy=not args.reference_only,
            recorded_at=args.recorded_at,
            time_offset_s=args.time_offset,
            offset_basis=args.offset_basis,
            operator=args.operator,
            notes=args.notes,
        )
        _print(res)
        return 0
    if args.sub == "list":
        _print(ex.list_registered(root, role=args.role, scene_id=args.scene, episode_id=args.episode))
        return 0
    if args.sub == "verify":
        res = ex.verify_registered(root, check_content=args.check_content)
        _print(res)
        return 0 if res["ok"] else 1
    if args.sub == "probe":
        _print(ex.probe_video(Path(args.file)))
        return 0
    raise SystemExit("未知子命令")


# --------------------------------------------------------------------------- quality


def cmd_quality(args: argparse.Namespace) -> int:
    from . import quality as q

    cfg = _cfg(args)
    root = _root(cfg, args)
    if args.sub == "check":
        rep = q.check_episode(root, args.episode, cfg=cfg, image_checks=args.image_checks)
        _print({k: v for k, v in rep.items() if k != "checks"})
        print("\n逐项检查：")
        for c in rep["checks"]:
            print(f"  [{c['status']:>15}] {c['name']}: {c['detail']}")
        return 0 if rep["overall"] == "pass" else 2
    if args.sub == "dataset":
        _print(q.check_dataset(root, cfg=cfg, image_checks=args.image_checks))
        return 0
    raise SystemExit("未知子命令")


# --------------------------------------------------------------------------- verify-fk


def cmd_verify_fk(args: argparse.Namespace) -> int:
    import numpy as np

    from .kinematics import (
        DH_TABLES,
        dh_fk_link6,
        parse_urdf_chain,
        sdk_cal_fk,
        urdf_fk_link6,
    )
    from .schema import Transform

    cfg = _cfg(args)
    hs = [0.0, 0.3, -0.4, 0.5, -0.6, 0.7]
    hs2 = [0.5, -0.8, 0.9, -0.2, 0.4, -1.1]
    urdf_path = Path(args.urdf or cfg["robot"]["urdf_path"])
    chain = parse_urdf_chain(urdf_path) if urdf_path.is_file() else None

    rows: List[Dict[str, Any]] = []
    for label, q in (("zero", [0.0] * 6), ("pose_a", hs), ("pose_b", hs2)):
        sdk = sdk_cal_fk(q, 0x01)
        sdk_T = Transform.from_rpy_xyz(
            [np.radians(v) for v in sdk[5][3:]], [v / 1000.0 for v in sdk[5][:3]]
        )
        row: Dict[str, Any] = {"pose": label, "joints_rad": q}
        for off in (0x00, 0x01):
            T = dh_fk_link6(q, off)
            row[f"dh_0x{off:02x}_vs_sdk_mm"] = float(np.linalg.norm(T[:3, 3] - sdk_T[:3, 3]) * 1000.0)
            row[f"dh_0x{off:02x}_vs_sdk_deg"] = Transform.rotation_angle_deg(T[:3, :3], sdk_T[:3, :3])
            row[f"dh_0x{off:02x}_xyz_m"] = [float(v) for v in T[:3, 3]]
        if chain is not None:
            U = urdf_fk_link6(q, chain)
            row["urdf_vs_dh_0x01_mm"] = float(np.linalg.norm(U[:3, 3] - dh_fk_link6(q, 0x01)[:3, 3]) * 1000.0)
            row["urdf_vs_dh_0x01_deg"] = Transform.rotation_angle_deg(U[:3, :3], dh_fk_link6(q, 0x01)[:3, :3])
            row["urdf_xyz_m"] = [float(v) for v in U[:3, 3]]
        row["sdk_xyz_m"] = [float(v) for v in sdk_T[:3, 3]]
        rows.append(row)

    out = {
        "schema_version": SCHEMA_VERSION,
        "kind": "fk_cross_check",
        "sources": {
            "project_dh_0x00": DH_TABLES[0x00],
            "project_dh_0x01": DH_TABLES[0x01],
            "sdk": "piper_sdk.C_PiperForwardKinematics.CalFK (XYZ: mm, RPY: deg)",
            "urdf": str(urdf_path) if chain is not None else None,
        },
        "rows": rows,
        "interpretation": (
            "dh_is_offset=0x01 与 piper_sdk 及官方 URDF 的差值应在 0.1 mm 量级；"
            "dh_is_offset=0x00 会因 j2/j3 的 2° 零位偏置产生约 10 mm 恒定差，"
            "用于确认当前型号应使用哪一套参数"
        ),
    }
    _print(out)
    return 0


# --------------------------------------------------------------------------- argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="piper_capture", description="PiPER + D435i 数据采集/标定/质检工具")
    p.add_argument("--config", help="JSON 配置文件（与默认配置深度合并）")
    p.add_argument("--dataset-root", help="数据集根目录（覆盖配置）")

    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="环境检查（只读）")
    d.add_argument("--write-report", action="store_true", help="数据集目录不存在时也写报告")
    d.set_defaults(func=cmd_doctor)

    cp = sub.add_parser("camera-probe", help="列出 D435i 支持的流配置，可选实测帧率")
    cp.add_argument("--open", action="store_true", help="实际打开相机并实测帧率")
    cp.add_argument("--seconds", type=float, default=8.0)
    cp.set_defaults(func=cmd_camera_probe)

    ui = sub.add_parser("camera-ui", help="双 D435i 实时预览、调参并保存采集配置")
    ui.add_argument("--output", help="保存配置路径，默认源文件名加 _tuned.json")
    ui.add_argument("--port", type=int, default=8766)
    ui.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    ui.set_defaults(func=cmd_camera_ui)

    c = sub.add_parser("capture", help="采集一个 episode（默认只读机械臂）")
    c.add_argument("--scene", default="scene-tabletop")
    c.add_argument("--episode", default=None)
    c.add_argument("--duration", type=float, default=None, help="秒；不指定则持续到 Ctrl-C")
    c.add_argument("--camera-only", action="store_true", help="不连接机械臂，只采 D435i")
    c.add_argument("--notes", default=None)
    c.add_argument("--verbose", action="store_true")
    c.set_defaults(func=cmd_capture)

    lc = sub.add_parser("capture-lerobot", help="两台 D435i + PiPER 直接写 LeRobotDataset v3")
    lc.add_argument("--duration", type=float, default=None, help="秒；不指定则持续到 Ctrl-C")
    lc.add_argument("--task", default="piper observation")
    lc.set_defaults(func=cmd_capture_lerobot)

    li = sub.add_parser(
        "capture-lerobot-interactive",
        help="交互采集：按空格开始 episode，按 q 结束并保存",
    )
    li.add_argument("--episodes", type=int, default=None, help="完成指定数量后退出；默认持续到 Ctrl-C")
    li.add_argument("--task", default="piper observation")
    li.set_defaults(func=cmd_capture_lerobot_interactive)

    lv = sub.add_parser("verify-lerobot", help="官方 LeRobotDataset 重新加载并检查字段")
    lv.add_argument("--root", required=True)
    lv.add_argument("--repo-id", default="piper_two_d435i")
    lv.set_defaults(func=cmd_verify_lerobot)

    h = sub.add_parser("handeye", help="手眼标定")
    hsub = h.add_subparsers(dest="sub", required=True)
    hs = hsub.add_parser("sample", help="交互采样（人工调整姿态，回车记录）")
    hs.add_argument("--session", default=None)
    hs.add_argument("--notes", default=None)
    hs.add_argument("--color-format", default=None)
    hs.add_argument("--resume", action="store_true", help="续接同一标定板、相机与模式的已有会话")
    hs.add_argument("--preview", action="store_true", help="本机网页实时预览和记录按钮（127.0.0.1:8765）")
    hs.set_defaults(func=cmd_handeye)
    hso = hsub.add_parser("solve", help="求解并留出验证")
    hso.add_argument("--session", required=True)
    hso.add_argument("--calibration-id", default=None)
    hso.add_argument("--verify-session", default=None, help="用另一个会话的全部样本做验证")
    hso.add_argument("--method", default=None, choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"])
    hso.add_argument("--redetect", action="store_true", help="从保存的图像离线重新检测")
    hso.set_defaults(func=cmd_handeye)
    hv = hsub.add_parser("verify", help="用指定会话验证已有标定")
    hv.add_argument("--calibration-id", required=True)
    hv.add_argument("--session", required=True)
    hv.set_defaults(func=cmd_handeye)
    hc = hsub.add_parser("check", help="只检查样本数量与姿态变化")
    hc.add_argument("--session", required=True)
    hc.add_argument("--redetect", action="store_true")
    hc.set_defaults(func=cmd_handeye)
    hl = hsub.add_parser("list", help="列出标定与采样会话")
    hl.set_defaults(func=cmd_handeye)

    g = sub.add_parser("gripper", help="夹爪诊断与开度校准")
    gsub = g.add_subparsers(dest="sub", required=True)
    gd = gsub.add_parser("diagnose", help="只读被动诊断")
    gd.add_argument("--seconds", type=float, default=10.0)
    gd.set_defaults(func=cmd_gripper)
    gpr = gsub.add_parser("probe", help="主动探测（会真实驱动夹爪）")
    gpr.add_argument("--allow-motion", action="store_true", help="必须显式确认才会发送命令")
    gpr.add_argument("--targets", type=float, nargs="+", default=[0.0, 20.0, 40.0, 60.0, 0.0])
    gpr.add_argument("--dwell", type=float, default=1.2)
    gpr.add_argument("--effort", type=int, default=1000)
    gpr.add_argument("--send-hz", type=float, default=200.0)
    gpr.set_defaults(func=cmd_gripper)
    gc = gsub.add_parser("calibrate", help="保存多点实测开度校准")
    gc.add_argument("--calibration-id", required=True)
    gc.add_argument("--point", action="append", required=True, help="raw:measured_mm，可重复")
    gc.add_argument("--notes", default=None)
    gc.set_defaults(func=cmd_gripper)
    gpx = gsub.add_parser("pending", help="输出未校准占位记录与所需测量步骤")
    gpx.add_argument("--calibration-id", default=None)
    gpx.add_argument("--reason", default=None)
    gpx.set_defaults(func=cmd_gripper)

    v = sub.add_parser("video", help="第三人称视频导入/登记")
    vsub = v.add_subparsers(dest="sub", required=True)
    va = vsub.add_parser("add", help="登记或导入视频")
    va.add_argument("--role", required=True, choices=["side_task", "environment_overview"])
    va.add_argument("--file", required=True)
    va.add_argument("--scene", default=None)
    va.add_argument("--episode", default=None)
    va.add_argument("--reference-only", action="store_true", help="只登记路径，不复制进数据集")
    va.add_argument("--recorded-at", default=None)
    va.add_argument("--time-offset", type=float, default=None)
    va.add_argument("--offset-basis", default=None)
    va.add_argument("--operator", default=None)
    va.add_argument("--notes", default=None)
    va.set_defaults(func=cmd_video)
    vl = vsub.add_parser("list", help="列出已登记视频")
    vl.add_argument("--role", default=None, choices=["side_task", "environment_overview"])
    vl.add_argument("--scene", default=None)
    vl.add_argument("--episode", default=None)
    vl.set_defaults(func=cmd_video)
    vv = vsub.add_parser("verify", help="核对文件存在与校验值")
    vv.add_argument("--check-content", action="store_true")
    vv.set_defaults(func=cmd_video)
    vp = vsub.add_parser("probe", help="探测视频基本信息")
    vp.add_argument("--file", required=True)
    vp.set_defaults(func=cmd_video)

    q = sub.add_parser("quality", help="数据质量检查")
    qsub = q.add_subparsers(dest="sub", required=True)
    qc = qsub.add_parser("check", help="检查单个 episode")
    qc.add_argument("--episode", required=True)
    qc.add_argument("--image-checks", type=int, default=12)
    qc.set_defaults(func=cmd_quality)
    qd = qsub.add_parser("dataset", help="检查整个数据集")
    qd.add_argument("--image-checks", type=int, default=6)
    qd.set_defaults(func=cmd_quality)

    f = sub.add_parser("verify-fk", help="正运动学三方交叉校验")
    f.add_argument("--urdf", default=None)
    f.set_defaults(func=cmd_verify_fk)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断（若在采集中，已完成的数据已落盘）", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, PermissionError) as exc:
        # 输入/环境不满足前置条件（如点数不足、文件不存在、未加 --allow-motion）
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"RuntimeError: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
