"""Direct LeRobotDataset v3 writer for the two D435i + PiPER capture.

The module deliberately keeps the device code separate from the legacy stage-one
PNG/JSONL writer.  A frame is submitted to the official writer only after both
cameras have produced a frame pair within the configured host-time tolerance.
Missing or unmatched frames are counted and dropped as a complete sample; no
last-frame or black-image substitution is performed.
"""
from __future__ import annotations

import json
import signal
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import numpy as np

from .camera import RealsenseCamera
from .clock import DeviceClockMapper, RobotStateMatcher
from .config import dataset_root as resolve_root
from .kinematics import ForwardKinematics
from .robot import GripperCalibration, RobotReader, RobotState


VIDEO_KEYS = (
    "observation.images.wrist",
    "observation.images.wrist_depth",
    "observation.images.third_person",
    "observation.images.third_person_depth",
)
DEPTH_KEYS = {VIDEO_KEYS[1], VIDEO_KEYS[3]}


def _describe_existing(out: Path) -> str:
    """已有输出目录时给出可执行的处置建议。只读检查，不自动删除任何文件。"""
    try:
        info = out / "meta" / "info.json"
        data_files = [p for p in (out / "data").rglob("*") if p.is_file()] if (out / "data").is_dir() else []
        video_files = (
            [p for p in (out / "videos").rglob("*") if p.is_file()] if (out / "videos").is_dir() else []
        )
    except Exception as exc:
        return f"\n（读取目录内容失败：{exc}）"

    if not info.is_file():
        return (
            f"\n目录非空但没有 meta/info.json（data 文件 {len(data_files)} 个、"
            f"视频文件 {len(video_files)} 个），不是完整的 LeRobot 数据集，请人工确认后再处理。"
        )
    try:
        meta = json.loads(info.read_text())
    except Exception as exc:
        return f"\n（meta/info.json 无法解析：{exc}）"

    n_ep, n_fr = meta.get("total_episodes"), meta.get("total_frames")
    if n_ep == 0 and n_fr == 0 and not data_files and not video_files:
        return (
            f"\n该目录是**没有任何数据**的残留空壳（total_episodes=0、total_frames=0，"
            f"data/ 与 videos/ 下无文件），通常是上次采集在 create 成功、写入任何帧之前就失败留下的。"
            f"确认无需保留后可删除再重试：rm -rf {out}"
        )
    return (
        f"\n该目录**已包含真实数据**（total_episodes={n_ep}、total_frames={n_fr}、"
        f"data 文件 {len(data_files)} 个、视频文件 {len(video_files)} 个）。"
        f"本项目不会覆盖它：请改用新的 output_root，或先自行备份再删除。"
    )


def _require_lerobot():
    try:
        import lerobot
        from lerobot.configs import DepthEncoderConfig, RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        version = getattr(lerobot, "__version__", None)
        if version != "0.6.1":
            raise RuntimeError(f"需要锁定 LeRobot 0.6.1，实际版本为 {version!r}")
    except Exception as exc:  # pragma: no cover - exercised on an unprovisioned host
        raise RuntimeError(
            "本采集器需要锁定的 LeRobot v0.6.1。请按 requirements-lerobot.txt 安装，"
            f"当前导入失败: {exc}"
        ) from exc
    return LeRobotDataset, RGBEncoderConfig, DepthEncoderConfig


def _feature_info(*, depth: bool, depth_scale_m: Optional[float] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "is_depth_map": bool(depth),
        "source": "Intel RealSense D435i",
        "storage_policy": "official_lerobot_video_writer",
    }
    if depth:
        # DepthEncoderConfig parameters are metres.  Frames are converted from
        # each camera's uint16 device units using its measured depth_scale before
        # add_frame, so the writer's input unit is explicitly metres.
        info.update(
            {
                "depth_unit": "m",
                "input_unit": "m",
                "raw_input_dtype": "uint16",
                "raw_input_unit": "camera_depth_units",
                "depth_scale_m": depth_scale_m,
                "conversion": "depth_m = raw_uint16 * depth_scale_m",
                "invalid_value_raw": 0,
            }
        )
    else:
        info.update({"input_unit": "uint8", "encoding": "RGB"})
    return info


def make_features(height: int, width: int, wrist_depth_scale_m: float, third_depth_scale_m: float,
                  *, third_height: Optional[int] = None, third_width: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """Return the complete v3 feature schema used by this capture."""
    return {
        "observation.images.wrist": {
            "dtype": "video", "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
            "info": _feature_info(depth=False),
        },
        "observation.images.wrist_depth": {
            "dtype": "video", "shape": (height, width, 1),
            "names": ["height", "width", "channel"],
            "info": _feature_info(depth=True, depth_scale_m=wrist_depth_scale_m),
        },
        "observation.images.third_person": {
            "dtype": "video", "shape": (third_height or height, third_width or width, 3),
            "names": ["height", "width", "channel"],
            "info": _feature_info(depth=False),
        },
        "observation.images.third_person_depth": {
            "dtype": "video", "shape": (third_height or height, third_width or width, 1),
            "names": ["height", "width", "channel"],
            "info": _feature_info(depth=True, depth_scale_m=third_depth_scale_m),
        },
        # Six joint angles are in rad.  The seventh value is the uncalibrated
        # driver feedback unit, never mislabeled as measured millimetres.
        "observation.state": {
            "dtype": "float32", "shape": (7,),
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_feedback_raw"],
            "info": {
                "units": ["rad", "rad", "rad", "rad", "rad", "rad", "0.001 mm (driver raw)"],
                "gripper_calibrated": False,
                "gripper_policy": "raw feedback retained; no measured mm asserted without calibration",
            },
        },
        "observation.ee_pose": {
            "dtype": "float32", "shape": (7,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
            "info": {"units": ["m", "m", "m", "wxyz"], "source": "feedback_joint_fk"},
        },
        "timestamp_source_s": {"dtype": "float64", "shape": (1,), "names": ["seconds"]},
        "sync_error_ms": {"dtype": "float32", "shape": (1,), "names": ["milliseconds"]},
    }


class LeRobotCaptureSession:
    """Capture two serial-bound D435i streams and PiPER feedback into v3."""

    def __init__(self, cfg: Dict[str, Any], *, base_dir: Path, duration_s: Optional[float] = None, task: str = "piper observation"):
        self.cfg = cfg
        self.base_dir = Path(base_dir).resolve()
        self.duration_s = duration_s
        self.task = task
        self.stop_event = __import__("threading").Event()
        self._recent: Deque[RobotState] = deque(maxlen=256)
        self.wrist: Optional[RealsenseCamera] = None
        self.third: Optional[RealsenseCamera] = None
        self.reader: Optional[RobotReader] = None
        self.dataset = None
        self.matcher: Optional[RobotStateMatcher] = None
        self.summary: Dict[str, Any] = {
            "status": "failed", "frames_written": 0, "unmatched_camera_pairs": 0,
            "missing_wrist": 0, "missing_third_person": 0, "robot_invalid": 0,
            "match_errors_ms": [], "encoder_drops": {},
        }

    def request_stop(self, *_: Any) -> None:
        self.stop_event.set()

    def _install_signals(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.request_stop)
            except ValueError:
                pass

    def _on_robot_state(self, state: RobotState) -> None:
        self._recent.append(state)

    def _camera(self, spec: Dict[str, Any]) -> RealsenseCamera:
        s = spec["streams"]
        return RealsenseCamera(
            serial=spec["serial"],
            sensor_options=spec.get("sensor_options"),
            color_width=s["color"]["width"], color_height=s["color"]["height"], color_fps=s["color"]["fps"], color_format=s["color"]["format"],
            depth_width=s["depth"]["width"], depth_height=s["depth"]["height"], depth_fps=s["depth"]["fps"], depth_format=s["depth"]["format"],
            allow_spec_downgrade=False,
            warmup_frames=int(self.cfg.get("camera", {}).get("warmup_frames", 30)),
            frame_timeout_ms=int(self.cfg.get("camera", {}).get("frame_timeout_ms", 5000)),
        )

    def _new_dataset(self, wrist_model: Any, third_model: Any):
        LeRobotDataset, RGBEncoderConfig, DepthEncoderConfig = _require_lerobot()
        out = Path(self.cfg["lerobot"]["output_root"])
        if not out.is_absolute():
            out = (self.base_dir / out).resolve()
        wc = self.cfg["camera"]["wrist"]["streams"]["color"]
        tc = self.cfg["camera"]["third_person"]["streams"]["color"]
        features = make_features(int(wc["height"]), int(wc["width"]), wrist_model.depth_scale_m,
                                 third_model.depth_scale_m, third_height=int(tc["height"]), third_width=int(tc["width"]))
        rgb_cfg = RGBEncoderConfig(**self.cfg["lerobot"].get("rgb_encoder", {}))
        depth_cfg = DepthEncoderConfig(**self.cfg["lerobot"].get("depth_encoder", {}))
        if out.exists() and any(out.iterdir()):
            info = out / "meta" / "info.json"
            if not info.is_file():
                raise FileExistsError(
                    f"LeRobot 输出目录已有内容，但不是可继续写入的数据集: {out}"
                    + _describe_existing(out)
                )
            # LeRobot 0.6.1 provides resume(), which keeps the existing
            # metadata and appends the next saved episode instead of creating
            # a new dataset or overwriting the previous episodes.
            ds = LeRobotDataset.resume(
                repo_id=self.cfg["lerobot"].get("repo_id", "piper_two_d435i"),
                root=out,
                tolerance_s=float(self.cfg["capture"]["sync"].get("tolerance_ms", 33.0)) / 1000.0,
                streaming_encoding=True,
                encoder_queue_maxsize=int(self.cfg["lerobot"].get("encoder_queue_maxsize", 30)),
                encoder_threads=self.cfg["lerobot"].get("encoder_threads"),
                rgb_encoder=rgb_cfg,
                depth_encoder=depth_cfg,
            )
            expected_fps = int(self.cfg["capture"]["target_sample_rate"])
            if int(ds.meta.fps) != expected_fps:
                raise RuntimeError(
                    f"已有数据集 fps={ds.meta.fps}，当前配置 fps={expected_fps}，拒绝混合写入: {out}"
                )
            for key, spec in features.items():
                old = ds.meta.features.get(key)
                if old is None or old.get("dtype") != spec.get("dtype") or tuple(old.get("shape", ())) != tuple(spec.get("shape", ())):
                    raise RuntimeError(
                        f"已有数据集字段 {key} 与当前相机规格不一致，拒绝混合写入: {out}"
                    )
            for key, serial in (
                ("observation.images.wrist", self.cfg["camera"]["wrist"]["serial"]),
                ("observation.images.third_person", self.cfg["camera"]["third_person"]["serial"]),
            ):
                old_serial = ds.meta.features[key].get("info", {}).get("camera_serial")
                if old_serial and old_serial != serial:
                    raise RuntimeError(
                        f"已有数据集 {key} 来自相机 {old_serial}，当前为 {serial}，拒绝混合写入: {out}"
                    )
        else:
            ds = None
        # These values are metres in v0.6.x; retain the actual configuration in
        # feature info so decoding is physically reversible.
        for key in DEPTH_KEYS:
            features[key]["info"].update({
                "quantization": {
                    "depth_min_m": depth_cfg.depth_min,
                    "depth_max_m": depth_cfg.depth_max,
                    "shift_m": depth_cfg.shift,
                    "use_log": depth_cfg.use_log,
                    "quantization_bits": 12,
                    "codec": depth_cfg.vcodec,
                    "pix_fmt": depth_cfg.pix_fmt,
                }
            })
        if ds is None:
            ds = LeRobotDataset.create(
                repo_id=self.cfg["lerobot"].get("repo_id", "piper_two_d435i"),
                fps=int(self.cfg["capture"]["target_sample_rate"]),
                features=features,
                root=out,
                robot_type="piper",
                use_videos=True,
                tolerance_s=float(self.cfg["capture"]["sync"].get("tolerance_ms", 33.0)) / 1000.0,
                streaming_encoding=True,
                encoder_queue_maxsize=int(self.cfg["lerobot"].get("encoder_queue_maxsize", 30)),
                encoder_threads=self.cfg["lerobot"].get("encoder_threads"),
                rgb_encoder=rgb_cfg,
                depth_encoder=depth_cfg,
            )
        # Official metadata is the source of directory/video/parquet/index
        # layout.  Only supported feature info is augmented with device facts.
        capture_metadata = {
            "camera_serials": {"wrist": self.cfg["camera"]["wrist"]["serial"], "third_person": self.cfg["camera"]["third_person"]["serial"]},
            "sensor_options": {"wrist": wrist_model.sensor_options, "third_person": third_model.sensor_options},
            "software_sync": True,
            "matching_basis": "nearest host frameset receive time",
            "tolerance_ms": self.cfg["capture"]["sync"]["tolerance_ms"],
            "hardware_sync": False,
            "action": {"status": "unavailable", "reason": "read-only capture has no command channel; feedback is never copied as action"},
            "depth_inputs": "float32 metres after per-camera raw_uint16 * measured depth_scale_m",
            "depth_encoder": {"depth_min_m": depth_cfg.depth_min, "depth_max_m": depth_cfg.depth_max, "shift_m": depth_cfg.shift, "use_log": depth_cfg.use_log, "vcodec": depth_cfg.vcodec, "pix_fmt": depth_cfg.pix_fmt},
        }
        # ``features[*].info`` is the official extensible metadata structure;
        # keep device identity, synchronization and the explicit unavailable
        # action policy there instead of creating a private sidecar file.
        for key in ds.meta.features:
            ds.meta.features[key].setdefault("info", {})["capture_metadata"] = capture_metadata
        ds.meta.features["observation.images.wrist"]["info"]["camera_serial"] = self.cfg["camera"]["wrist"]["serial"]
        ds.meta.features["observation.images.third_person"]["info"]["camera_serial"] = self.cfg["camera"]["third_person"]["serial"]
        return ds

    @staticmethod
    def _depth_m(pair: Any, scale_m: float) -> np.ndarray:
        raw = np.asarray(pair.depth_aligned_u16, dtype=np.uint16)
        # Keep a single channel explicitly; zero remains the sensor's invalid
        # measurement and is never replaced by a prior frame.
        return (raw.astype(np.float32) * float(scale_m))[..., None]

    def run(self) -> Dict[str, Any]:
        self._install_signals()
        cams = self.cfg["camera"]
        sync = self.cfg["capture"]["sync"]
        try:
            self.wrist = self._camera(cams["wrist"])
            self.third = self._camera(cams["third_person"])
            wrist_model = self.wrist.open()
            third_model = self.third.open()
            self.dataset = self._new_dataset(wrist_model, third_model)
            robot_cfg = self.cfg["robot"]
            if robot_cfg.get("enabled", True):
                gripper = GripperCalibration()
                cid = self.cfg.get("gripper", {}).get("calibration_id")
                if cid:
                    from .episode import CalibrationStore
                    store = CalibrationStore(resolve_root(self.cfg, self.base_dir))
                    if store.exists("gripper", cid):
                        gripper = GripperCalibration(store.load("gripper", cid))
                self.reader = RobotReader(
                    can_interface=robot_cfg["can_interface"], dh_is_offset=int(robot_cfg["dh_is_offset"]),
                    poll_hz=float(robot_cfg.get("poll_hz", 200.0)), queries_on_connect=True,
                    feedback_timeout_s=float(robot_cfg.get("feedback_timeout_s", 1.0)),
                    tool_offset_m=robot_cfg.get("tool_offset_m", [0.0, 0.0, 0.0]), ee_frame=robot_cfg.get("ee_frame", "link6"),
                    base_frame=robot_cfg.get("base_frame", "piper_base_link"), on_state=self._on_robot_state,
                    gripper_calibration=gripper,
                )
                if not self.reader.open(timeout_s=5.0):
                    raise RuntimeError(f"PiPER feedback unavailable: {self.reader.open_error}")
                fk = self.reader.fk
            else:
                raise RuntimeError("正式 LeRobot 采集要求 robot.enabled=true；camera-only 不会伪造机械臂字段")
            self.matcher = RobotStateMatcher(
                fk, tolerance_ms=float(sync["robot_tolerance_ms"]),
                max_state_age_ms=float(sync["max_robot_state_age_ms"]), mode=str(sync["robot_match_mode"]),
            )
            started = time.monotonic()
            progress_last_s = 0.0
            print(
                f"开始采集：腕部 {self.cfg['camera']['wrist']['serial']} + "
                f"第三人称 {self.cfg['camera']['third_person']['serial']}，"
                "Ctrl-C 结束（会先把已采帧编码完再退出）",
                file=sys.stderr,
                flush=True,
            )
            # Read both USB devices concurrently.  Sequential blocking reads
            # would manufacture an approximately one-frame timestamp offset.
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="d435i-read") as pool:
              while not self.stop_event.is_set() and (self.duration_s is None or time.monotonic() - started < self.duration_s):
                wf = pool.submit(self.wrist.read)
                tf = pool.submit(self.third.read)
                wp, tp = wf.result(), tf.result()
                if wp is None:
                    self.summary["missing_wrist"] += 1; continue
                if tp is None:
                    self.summary["missing_third_person"] += 1; continue
                camera_dt_ms = (float(wp.frameset_host_recv_ns) - float(tp.frameset_host_recv_ns)) / 1e6
                if abs(camera_dt_ms) > float(sync["tolerance_ms"]):
                    self.summary["unmatched_camera_pairs"] += 1
                    self.summary["match_errors_ms"].append(camera_dt_ms)
                    continue
                target_ns = int(round((wp.frameset_host_recv_ns + tp.frameset_host_recv_ns) / 2))
                matched = self.matcher.match(list(self._recent), target_ns)
                if not matched.valid or matched.joints_rad is None or matched.ee_pose is None:
                    self.summary["robot_invalid"] += 1
                    continue
                gripper_raw = matched.gripper_feedback_raw
                if gripper_raw is None:
                    self.summary["robot_invalid"] += 1
                    continue
                state = np.asarray([*matched.joints_rad, float(gripper_raw)], dtype=np.float32)
                ee = np.asarray(matched.ee_pose, dtype=np.float32)
                frame = {
                    "observation.images.wrist": np.asarray(wp.color_bgr[..., ::-1], dtype=np.uint8).copy(),
                    "observation.images.wrist_depth": self._depth_m(wp, wrist_model.depth_scale_m),
                    "observation.images.third_person": np.asarray(tp.color_bgr[..., ::-1], dtype=np.uint8).copy(),
                    "observation.images.third_person_depth": self._depth_m(tp, third_model.depth_scale_m),
                    "observation.state": state,
                    "observation.ee_pose": ee,
                    "timestamp_source_s": np.asarray([target_ns / 1e9], dtype=np.float64),
                    "sync_error_ms": np.asarray([camera_dt_ms], dtype=np.float32),
                    "task": self.task,
                }
                self.dataset.add_frame(frame)
                self.summary["frames_written"] += 1
                now = time.monotonic()
                if now - progress_last_s >= 5.0 or self.summary["frames_written"] == 1:
                    progress_last_s = now
                    elapsed = max(1e-9, now - started)
                    print(
                        f"  已写入 {self.summary['frames_written']} 样本"
                        f"（{self.summary['frames_written'] / elapsed:.1f} Hz）"
                        f" | 丢弃: 相机不匹配 {self.summary['unmatched_camera_pairs']},"
                        f" 缺腕部帧 {self.summary['missing_wrist']},"
                        f" 缺三视帧 {self.summary['missing_third_person']},"
                        f" 机械臂无效 {self.summary['robot_invalid']}",
                        file=sys.stderr,
                        flush=True,
                    )
            # Runtime matching evidence belongs in the official feature metadata
            # so it travels with info.json; no sidecar report is created.
            errors = self.summary["match_errors_ms"]
            self.dataset.meta.features["sync_error_ms"].setdefault("info", {})["runtime"] = {
                "frames_written": self.summary["frames_written"],
                "unmatched_camera_pairs": self.summary["unmatched_camera_pairs"],
                "missing_wrist": self.summary["missing_wrist"],
                "missing_third_person": self.summary["missing_third_person"],
                "robot_invalid": self.summary["robot_invalid"],
                "camera_dt_error_ms_min": min(errors) if errors else None,
                "camera_dt_error_ms_max": max(errors) if errors else None,
                "robot_match_counters": self.matcher.describe()["counters"] if self.matcher else {},
                "policy": "unmatched/invalid samples are dropped; no stale-frame or black-frame fill",
            }
            stream = getattr(getattr(self.dataset, "writer", None), "_streaming_encoder", None)
            pending_drops = dict(getattr(stream, "_dropped_frames", {}) or {})
            if pending_drops:
                self.summary["encoder_drops"] = pending_drops
                self.dataset.clear_episode_buffer(delete_images=True)
                self.dataset.finalize()
                self.summary.update({"status": "failed_encoder_backpressure", "error": "官方实时编码队列溢出，已丢弃并清理未完成 episode"})
                return self.summary
            if self.summary["frames_written"]:
                self.dataset.save_episode(parallel_encoding=False)
            elif self.summary["status"] != "failed_encoder_backpressure":
                self.dataset.finalize()
                self.summary.update({"status": "failed_no_valid_frames", "error": "episode 没有同时满足相机和机械臂时间容差的有效帧"})
                return self.summary
            self.dataset.finalize()
            drops = getattr(getattr(self.dataset, "writer", None), "_streaming_encoder", None)
            self.summary["encoder_drops"] = dict(getattr(drops, "_dropped_frames", {}) or {})
            if self.summary["encoder_drops"]:
                self.summary["status"] = "failed_encoder_backpressure"
            else:
                self.summary["status"] = "aborted" if self.stop_event.is_set() else "closed"
            self.summary["output_root"] = str(self.dataset.root)
            self.summary["lerobot_version"] = "0.6.1"
            return self.summary
        except BaseException as exc:
            self.summary.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            if self.dataset is not None:
                try:
                    if getattr(self.dataset, "has_pending_frames", lambda: False)():
                        self.dataset.clear_episode_buffer(delete_images=True)
                    self.dataset.finalize()
                except Exception:
                    pass
            return self.summary
        finally:
            if self.reader is not None:
                self.reader.close()
            if self.wrist is not None:
                self.wrist.close()
            if self.third is not None:
                self.third.close()


def verify_lerobot_dataset(root: Path, repo_id: str = "piper_two_d435i") -> Dict[str, Any]:
    """Reload a finished dataset through the official reader and inspect one row.

    ``LeRobotDataset.__init__`` resolves the revision against the Hub whenever the
    local reader cache is not usable (empty/incomplete dataset).  For a local-only
    dataset that is never wanted: it either fails with a network/proxy error or
    reports a missing Hub repo.  So the on-disk dataset is sanity-checked first and
    an empty one is reported locally, without touching the network.
    """
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise RuntimeError(
            f"不是 LeRobot 数据集：缺少 {info_path}。若采集从未成功写过 episode，请检查 capture-lerobot 的输出。"
        )
    meta = json.loads(info_path.read_text())
    n_ep, n_fr = meta.get("total_episodes"), meta.get("total_frames")
    if not n_ep or not n_fr:
        raise RuntimeError(
            f"数据集为空，没有可校验的帧（total_episodes={n_ep}、total_frames={n_fr}）：{root}。"
            "这通常是上次采集在写入任何帧之前就失败留下的空壳；确认无需保留后删除该目录再重新采集。"
        )

    LeRobotDataset, _, _ = _require_lerobot()
    ds = LeRobotDataset(repo_id, root=root, depth_output_unit="m", force_cache_sync=False)
    if len(ds) == 0:
        raise RuntimeError("LeRobotDataset 已加载但没有帧")
    row = ds[0]
    missing = [k for k in (*VIDEO_KEYS, "observation.state", "observation.ee_pose") if k not in row]
    if missing:
        raise RuntimeError(f"LeRobotDataset 首帧缺少字段: {missing}")
    return {
        "ok": True,
        "root": str(Path(root).resolve()),
        "lerobot_version": "0.6.1",
        "frames": len(ds),
        "video_keys": list(ds.meta.video_keys),
        "depth_keys": list(ds.meta.depth_keys),
        "first_row_keys": sorted(row.keys()),
        "state_shape": list(np.asarray(row["observation.state"]).shape),
        "ee_pose_shape": list(np.asarray(row["observation.ee_pose"]).shape),
    }
