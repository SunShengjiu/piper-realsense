"""采集会话：把 D435i 的 RGB-D 帧与只读机械臂反馈组织成 episode 样本。

组织基准是 **RGB-D 帧对**（目标约 30 样本/秒）；机械臂反馈以原生频率单独
记录到 robot_states.jsonl，供事后复核。

本会话不发任何机械臂或夹爪运动指令。
"""
from __future__ import annotations

import signal
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from .camera import CameraSpecError, RealsenseCamera
from .clock import DeviceClockMapper, RobotStateMatcher
from .config import dataset_root as resolve_root
from .episode import (
    CalibrationStore,
    DatasetManifest,
    EpisodeWriter,
    build_sample_record,
    ensure_scene,
    new_episode_id,
)
from .jsonio import write_json
from .kinematics import ForwardKinematics
from .robot import GripperCalibration, RobotReader, RobotState
from .schema import SCHEMA_VERSION


class CaptureAborted(RuntimeError):
    pass


class _ImageWriter:
    """把三张 PNG 的编码/落盘从采集主循环里解耦。

    实测 1280x720 三张无损 PNG：compression=0 串行约 31ms、并行约 16ms；
    而 `wait_for_frames` 本身就要阻塞约 33ms（一帧周期）。若在循环里同步落盘，
    每样本耗时变成 33+31=64ms，实测只有约 20 样本/秒。

    这里主循环只把帧交给后台线程池，pending 有上限（信号量），
    超出上限时主循环才阻塞——保证不会无界占用内存，也不会静默丢样本。
    """

    def __init__(self, *, workers: int = 3, max_pending: int = 64, png_compression: int = 0) -> None:
        self.png_compression = int(png_compression)
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="imgwrite")
        self._slots = threading.Semaphore(max(1, int(max_pending)))
        self._lock = threading.Lock()
        self._futures: Deque[Any] = deque()
        self.queued = 0
        self.completed = 0
        self.failed = 0
        self.max_pending_seen = 0
        self.errors: List[str] = []

    def submit(
        self,
        pair: Any,
        rgb_dir: Optional[Path],
        raw_dir: Path,
        aligned_dir: Optional[Path],
        stem: str,
    ) -> None:
        self._slots.acquire()
        with self._lock:
            self.queued += 1
            pending = self.queued - self.completed - self.failed
            if pending > self.max_pending_seen:
                self.max_pending_seen = pending
        self._futures.append(
            self._pool.submit(self._save, pair, rgb_dir, raw_dir, aligned_dir, stem)
        )

    def _save(self, pair: Any, rgb_dir: Any, raw_dir: Any, aligned_dir: Any, stem: str) -> None:
        try:
            RealsenseCamera.save_frames(
                pair, rgb_dir, raw_dir, aligned_dir, stem, png_compression=self.png_compression
            )
            with self._lock:
                self.completed += 1
        except Exception as exc:  # 落盘失败必须留痕，不能让样本指向不存在的文件
            with self._lock:
                self.failed += 1
                if len(self.errors) < 20:
                    self.errors.append(f"{stem}: {exc}")
        finally:
            self._slots.release()

    def drain(self, timeout_s: float = 120.0) -> None:
        """等待所有已入队的落盘任务结束（中断/收尾时调用）。"""
        deadline = time.monotonic() + float(timeout_s)
        while True:
            with self._lock:
                if not self._futures:
                    return
                fut = self._futures.popleft()
            try:
                fut.result(timeout=max(0.0, deadline - time.monotonic()))
            except Exception:
                pass

    def close(self) -> None:
        self.drain()
        self._pool.shutdown(wait=True)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            pending = self.queued - self.completed - self.failed
            return {
                "png_compression": self.png_compression,
                "queued": self.queued,
                "completed": self.completed,
                "failed": self.failed,
                "pending_at_close": pending,
                "max_pending_seen": self.max_pending_seen,
                "errors": list(self.errors),
                "mode": "async_background_writer",
                "note": "落盘与取帧解耦；收尾时全部 drain，完成的文件与样本一一对应",
            }


class CaptureSession:
    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        base_dir: Path,
        scene_id: str = "scene-tabletop",
        episode_id: Optional[str] = None,
        duration_s: Optional[float] = None,
        camera_only: bool = False,
        notes: Optional[str] = None,
        progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.base_dir = Path(base_dir).resolve()
        self.root = resolve_root(cfg, self.base_dir)
        self.scene_id = scene_id
        self.episode_id = episode_id or new_episode_id()
        self.duration_s = duration_s
        self.camera_only = camera_only
        self.notes = notes
        self.progress = progress

        self.stop_event = threading.Event()
        self._recent: Deque[RobotState] = deque(maxlen=128)
        self.reader: Optional[RobotReader] = None
        self.camera: Optional[RealsenseCamera] = None
        self.writer: Optional[EpisodeWriter] = None
        self.image_writer: Optional[_ImageWriter] = None
        self.rate_limited_skips = 0
        self.mappers: Dict[str, DeviceClockMapper] = {
            "color": DeviceClockMapper("color", "realsense_device_clock"),
            "depth": DeviceClockMapper("depth", "realsense_device_clock"),
        }
        self.summary: Dict[str, Any] = {}

    # ------------------------------------------------------------------ 生命周期
    def request_stop(self, *_: Any) -> None:
        self.stop_event.set()

    def _install_signal_handlers(self) -> None:
        try:
            signal.signal(signal.SIGINT, self.request_stop)
            signal.signal(signal.SIGTERM, self.request_stop)
        except ValueError:  # 非主线程
            pass

    def _load_gripper_calibration(self) -> GripperCalibration:
        cid = self.cfg["gripper"].get("calibration_id")
        if not cid:
            return GripperCalibration(
                {
                    "valid": False,
                    "calibration_id": None,
                    "status_reason": "配置未指定夹爪校准文件，gripper_width_mm 置空",
                }
            )
        store = CalibrationStore(self.root)
        if not store.exists("gripper", cid):
            return GripperCalibration(
                {
                    "valid": False,
                    "calibration_id": cid,
                    "status_reason": f"配置引用的夹爪校准文件不存在: {cid}",
                }
            )
        payload = store.load("gripper", cid)
        return GripperCalibration(payload)

    def run(self) -> Dict[str, Any]:
        self._install_signal_handlers()
        gripper_cal = self._load_gripper_calibration()

        # ---- 相机 ----
        cam_cfg = self.cfg["camera"]
        self.camera = RealsenseCamera(
            serial=cam_cfg.get("serial"),
            color_width=cam_cfg["color"]["width"],
            color_height=cam_cfg["color"]["height"],
            color_fps=cam_cfg["color"]["fps"],
            color_format=cam_cfg["color"]["format"],
            depth_width=cam_cfg["depth"]["width"],
            depth_height=cam_cfg["depth"]["height"],
            depth_fps=cam_cfg["depth"]["fps"],
            depth_format=cam_cfg["depth"]["format"],
            allow_spec_downgrade=bool(cam_cfg.get("allow_spec_downgrade")),
            warmup_frames=int(cam_cfg.get("warmup_frames", 30)),
            frame_timeout_ms=int(cam_cfg.get("frame_timeout_ms", 5000)),
            calibration_id=cam_cfg.get("calibration_id"),
        )
        camera_model = self.camera.open()
        store = CalibrationStore(self.root)
        camera_cal_path = store.save("camera", camera_model.calibration_id, camera_model.to_dict())

        # ---- 机械臂（只读）----
        robot_cfg = self.cfg["robot"]
        if not self.camera_only and robot_cfg.get("enabled", True):
            self.reader = RobotReader(
                can_interface=robot_cfg["can_interface"],
                dh_is_offset=int(robot_cfg["dh_is_offset"]),
                poll_hz=float(robot_cfg.get("poll_hz", 200.0)),
                queries_on_connect=True,
                feedback_timeout_s=float(robot_cfg.get("feedback_timeout_s", 1.0)),
                tool_offset_m=robot_cfg.get("tool_offset_m", [0.0, 0.0, 0.0]),
                ee_frame=robot_cfg.get("ee_frame", "link6"),
                base_frame=robot_cfg.get("base_frame", "piper_base_link"),
                sdk_joint_limit=bool(robot_cfg.get("sdk_joint_limit", False)),
                sdk_gripper_limit=bool(robot_cfg.get("sdk_gripper_limit", False)),
                on_state=self._on_robot_state,
                gripper_calibration=gripper_cal,
            )
            ok = self.reader.open(timeout_s=5.0)
            if not ok:
                self.camera.close()
                self.summary = {
                    "status": "failed",
                    "reason": self.reader.open_error,
                    "camera_calibration_id": camera_model.calibration_id,
                }
                return self.summary
        elif not self.camera_only:
            self.camera_only = True

        handeye_cid = self._resolve_handeye_id()
        scene_json = ensure_scene(
            self.root,
            self.scene_id,
            camera_calibration_id=camera_model.calibration_id,
            handeye_calibration_id=handeye_cid,
            gripper_calibration_id=gripper_cal.calibration_id if gripper_cal.valid else None,
            mount=self.cfg["camera"].get("mount"),
        )
        DatasetManifest(self.root).update()

        self.writer = EpisodeWriter(
            self.root,
            self.scene_id,
            self.episode_id,
            robot_meta=self._robot_meta(gripper_cal),
            camera_meta={
                **camera_model.to_dict(),
                "camera_calibration_path": self.writer_rel(camera_cal_path),
                "scene_json": self.writer_rel(scene_json),
            },
            sync_config={
                **self.cfg["capture"]["sync"],
                "clock_mapping": {
                    "color": self.mappers["color"].describe(),
                    "depth": self.mappers["depth"].describe(),
                },
            },
            capture_config={
                "target_sample_rate": self.cfg["capture"]["target_sample_rate"],
                "camera_only": self.camera_only,
                "motion_commands_sent": False,
                "configured_start_pose": self.cfg["robot"].get("start_pose"),
            },
            calibration_ids={
                "camera_calibration_id": camera_model.calibration_id,
                "handeye_calibration_id": handeye_cid,
                "gripper_calibration_id": gripper_cal.calibration_id if gripper_cal.valid else None,
            },
            notes=self.notes,
            fsync_every=int(self.cfg["capture"].get("fsync_every", 0)),
        )

        sync_cfg = self.cfg["capture"]["sync"]
        self.image_writer = _ImageWriter(
            workers=int(cam_cfg.get("save_workers", 3)),
            max_pending=int(cam_cfg.get("max_pending_saves", 64)),
            png_compression=int(cam_cfg.get("png_compression", 0)),
        )
        fk = self.reader.fk if self.reader else ForwardKinematics(
            dh_is_offset=int(robot_cfg["dh_is_offset"]),
            tool_offset_m=robot_cfg.get("tool_offset_m", [0.0, 0.0, 0.0]),
            ee_frame=robot_cfg.get("ee_frame", "link6"),
            base_frame=robot_cfg.get("base_frame", "piper_base_link"),
        )
        matcher = RobotStateMatcher(
            fk,
            tolerance_ms=float(sync_cfg["robot_tolerance_ms"]),
            max_state_age_ms=float(sync_cfg["max_robot_state_age_ms"]),
            mode=str(sync_cfg["robot_match_mode"]),
        )

        status = "aborted"
        loop_error: Optional[str] = None
        try:
            status = self._capture_loop(matcher, camera_model, sync_cfg)
        except KeyboardInterrupt:  # 信号已由 handler 转为 stop_event，这里是兜底
            status = "aborted"
        except Exception as exc:  # 异常也必须收尾：已入队的图像要落盘、metadata 要写
            loop_error = f"{type(exc).__name__}: {exc}"
            self.writer.log(f"采集循环异常终止: {loop_error}")
        return self._finalize(status, matcher, camera_model, gripper_cal, loop_error=loop_error)

    def writer_rel(self, path: Path) -> str:
        return str(Path(path).resolve().relative_to(self.root))

    def _resolve_handeye_id(self) -> Optional[str]:
        cid = self.cfg["handeye"].get("calibration_id")
        if cid and CalibrationStore(self.root).exists("handeye", cid):
            return cid
        return None

    def _robot_meta(self, gripper_cal: GripperCalibration) -> Dict[str, Any]:
        if self.reader is None:
            return {"enabled": False, "reason": "camera_only 模式，未连接机械臂"}
        meta = self.reader.export_driver_record()
        meta["gripper_calibration"] = {
            "calibration_id": gripper_cal.calibration_id,
            "calibrated": gripper_cal.valid,
            "status_reason": gripper_cal.reason,
        }
        return meta

    def _on_robot_state(self, state: RobotState) -> None:
        self._recent.append(state)
        if self.writer is not None and self.cfg["capture"].get("log_robot_states", True):
            self.writer.write_robot_state(state.to_dict())

    # ------------------------------------------------------------------ 主循环
    def _capture_loop(self, matcher: RobotStateMatcher, camera_model: Any, sync_cfg: Dict[str, Any]) -> str:
        assert self.writer is not None and self.camera is not None
        target_rate = float(self.cfg["capture"]["target_sample_rate"])
        rate_tolerance = float(self.cfg["capture"].get("rate_tolerance", 0.05))
        tolerance = float(sync_cfg["tolerance_ms"])
        started = time.monotonic()
        seq = 0
        rate_limited_skips = 0
        consecutive_timeouts = 0
        last_report = started
        while not self.stop_event.is_set():
            if self.duration_s is not None and time.monotonic() - started >= self.duration_s:
                break
            frame = self.camera.read()
            if frame is None:
                consecutive_timeouts += 1
                if consecutive_timeouts >= 30:
                    self.writer.log("连续 30 次取帧超时，停止采集")
                    return "aborted"
                continue
            consecutive_timeouts = 0
            now_mono = time.monotonic()
            # 只在“平均样本率已明显超过目标”时才跳过，并计数。
            # 不能用“距上一帧的间隔 < 1/target”判据：相机恰好跑在目标帧率时，
            # 帧间隔抖动会让 33.30ms < 33.33ms 成立，从而系统性丢帧
            # （实测样本率被压到 18Hz，而相机本身是 30Hz）。
            if target_rate > 0:
                elapsed = now_mono - started
                if elapsed > 1.0 and (seq + 1) / elapsed > target_rate * (1.0 + rate_tolerance):
                    rate_limited_skips += 1
                    continue
            seq += 1

            self.mappers["color"].observe(frame.color_device_ts_ms, frame.color_host_recv_ns)
            self.mappers["depth"].observe(frame.depth_device_ts_ms, frame.depth_host_recv_ns)

            target_ns = frame.frameset_host_recv_ns
            matched = matcher.match(list(self._recent), target_ns)
            sample_id = f"{self.episode_id}-{seq:06d}"
            stem = f"{seq:06d}"
            paths = {
                "rgb": f"{stem}.png" if self.cfg["camera"]["save_rgb"] else None,
                "depth_raw": f"{stem}.png",
                "depth_aligned": f"{stem}.png" if frame.depth_aligned_u16.size else None,
            }
            assert self.image_writer is not None
            self.image_writer.submit(
                frame,
                self.writer.dir / "rgb" if self.cfg["camera"]["save_rgb"] else None,
                self.writer.dir / "depth_raw",
                self.writer.dir / "depth_aligned",
                stem,
            )
            rel_paths = {
                "rgb": f"rgb/{paths['rgb']}" if paths["rgb"] else None,
                "depth_raw": f"depth_raw/{paths['depth_raw']}" if paths["depth_raw"] else None,
                "depth_aligned": f"depth_aligned/{paths['depth_aligned']}" if paths["depth_aligned"] else None,
            }
            record = build_sample_record(
                sample_id=sample_id,
                episode_id=self.episode_id,
                seq=seq,
                frame=frame,
                matched=matched,
                camera_model=camera_model,
                paths=rel_paths,
                relative_to=self.root,
                handeye_calibration_id=self._resolve_handeye_id(),
                clock_mappers=self.mappers,
                sync_tolerance_ms=tolerance,
            )
            self.writer.write_sample(record)
            if self.progress and seq % 15 == 0:
                self.progress({"seq": seq, "rate": seq / max(1e-6, time.monotonic() - started), "sample_id": sample_id})
            if time.monotonic() - last_report > 5.0:
                last_report = time.monotonic()
                self.writer.log(
                    f"已写 {seq} 个样本，机械臂反馈 {self.reader.counters['joint_frames'] if self.reader else 0} 帧，"
                    f"无效样本 {self.writer.invalid_sample_count}"
                )
            self._recent_prune(target_ns)
        self.rate_limited_skips = rate_limited_skips
        # stop_event 由 SIGINT/SIGTERM 的 handler 置位：这种情况是"被中断"，
        # 不能记成正常收尾，否则 metadata.status 会与实际不符（episode.py 约定
        # 正常结束写 closed，Ctrl-C 写 aborted）。
        return "aborted" if self.stop_event.is_set() else "closed"

    def _recent_prune(self, target_ns: int) -> None:
        cutoff = target_ns - 2_000_000_000
        while self._recent and self._recent[0].joints_host_recv_ns < cutoff:
            self._recent.popleft()

    # ------------------------------------------------------------------ 收尾
    def _finalize(
        self,
        status: str,
        matcher: RobotStateMatcher,
        camera_model: Any,
        gripper_cal: GripperCalibration,
        loop_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        assert self.writer is not None and self.camera is not None
        # 先停相机与机械臂，再 drain 落盘任务：避免后台线程还在写而句柄被关
        self.camera.close()
        if self.reader is not None:
            self.reader.close()
        image_stats = {"enabled": False}
        if self.image_writer is not None:
            self.image_writer.close()
            image_stats = self.image_writer.stats()
            if image_stats["failed"]:
                self.writer.log(f"图像落盘失败 {image_stats['failed']} 个样本: {image_stats['errors'][:5]}")

        camera_diag = self.camera.diagnostics()
        camera_diag["image_writer"] = image_stats
        robot_diag = self.reader.diagnostics() if self.reader else {"enabled": False}
        driver_record = self.reader.export_driver_record() if self.reader else {"enabled": False}
        diagnostics = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "camera": camera_diag,
            "robot": robot_diag,
            "sync": matcher.describe(),
            "clock_mapping": {
                "color": self.mappers["color"].describe(),
                "depth": self.mappers["depth"].describe(),
            },
            "gripper_calibration": {
                "calibration_id": gripper_cal.calibration_id,
                "calibrated": gripper_cal.valid,
                "status_reason": gripper_cal.reason,
            },
            "motion_commands_sent": False,
            "loop_error": loop_error,
            "image_writer": image_stats,
            "rate_limited_skips": self.rate_limited_skips,
        }
        write_json(self.writer.dir / "logs" / "diagnostics.json", diagnostics)
        write_json(self.writer.dir / "logs" / "driver_record.json", driver_record)

        self.writer.finalize(status, extra={"diagnostics_summary": {"camera": camera_diag, "robot_present": self.reader is not None}})
        dur = self.writer.metadata.get("duration_s")
        self.summary = {
            "status": status,
            "episode_id": self.episode_id,
            "episode_dir": str(self.writer.dir),
            "samples": self.writer.sample_count,
            "invalid_samples": self.writer.invalid_sample_count,
            "robot_states": self.writer.state_count,
            "duration_s": dur,
            "sample_span_s": self.writer.metadata.get("sample_span_s"),
            "sample_rate_hz": self.writer.metadata.get("measured_sample_rate_hz"),
            "camera_only": self.camera_only,
            "camera_calibration_id": camera_model.calibration_id,
            "handeye_calibration_id": self.writer.calibration_ids.get("handeye_calibration_id"),
            "gripper_calibrated": gripper_cal.valid,
            "camera_diagnostics": camera_diag,
            "robot_diagnostics": robot_diag,
            "sync": matcher.describe(),
            "loop_error": loop_error,
        }
        return self.summary
