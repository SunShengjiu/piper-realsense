"""D435i RGB-D 采集（pyrealsense2 直连，不依赖 ROS）。

要点：
  - 目标规格 RGB 1280x720@30、Depth 1280x720@30；启动后核对**实际生效**的
    profile，Requested 与 Actual 不一致时默认报错，不静默降级。
  - 原始深度与几何对齐深度分别保存为无损 uint16 PNG；
    aligned_depth_to_color 由 RealSense 的 rs.align 生成，与 RGB 同尺寸，
    不使用任何普通缩放替代。
  - 保存内参、畸变、Depth->Color 外参、depth_scale 与实际流配置。
  - 深度 0 表示“无有效测量”，保留该语义，不把彩色深度图当作深度数据。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

FORMAT_MAP: Dict[str, Any] = {}


def _rs():
    import pyrealsense2 as rs

    global FORMAT_MAP
    if not FORMAT_MAP:
        FORMAT_MAP = {
            "bgr8": rs.format.bgr8,
            "rgb8": rs.format.rgb8,
            "z16": rs.format.z16,
            "y8": rs.format.y8,
        }
    return rs


def _distortion_name(rs: Any, model: Any) -> str:
    try:
        return {
            rs.distortion.none: "none",
            rs.distortion.modified_brown_conrady: "modified_brown_conrady",
            rs.distortion.inverse_brown_conrady: "inverse_brown_conrady",
            rs.distortion.ftheta: "ftheta",
            rs.distortion.brown_conrady: "brown_conrady",
            rs.distortion.kannala_brandt4: "kannala_brandt4",
        }.get(model, str(model))
    except Exception:
        return str(model)


def _timestamp_domain_name(rs: Any, domain: Any) -> str:
    try:
        return {
            rs.timestamp_domain.hardware_clock: "hardware_clock",
            rs.timestamp_domain.system_time: "system_time",
            rs.timestamp_domain.global_time: "global_time",
        }.get(domain, str(domain))
    except Exception:
        return str(domain)


def intrinsics_to_dict(rs: Any, intr: Any) -> Dict[str, Any]:
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx": float(intr.ppx),
        "ppy": float(intr.ppy),
        "model": _distortion_name(rs, intr.model),
        "coeffs": [float(c) for c in intr.coeffs],
    }


def extrinsics_to_dict(extr: Any) -> Dict[str, Any]:
    return {
        "rotation_row_major": [float(v) for v in extr.rotation],
        "translation_m": [float(v) for v in extr.translation],
        "note": "RealSense 原始外参，未做任何手眼标定替换",
    }


@dataclass
class FramePair:
    """一个 RGB-D 样本对。"""

    seq: int
    color_bgr: np.ndarray
    depth_raw_u16: np.ndarray
    depth_aligned_u16: np.ndarray
    # 设备时间戳（属于各自时钟域）
    color_device_ts_ms: float
    depth_device_ts_ms: float
    color_ts_domain: str
    depth_ts_domain: str
    color_frame_number: int
    depth_frame_number: int
    aligned_frame_number: Optional[int]
    # 主机接收时间（time.time_ns 时钟域）
    color_host_recv_ns: int
    depth_host_recv_ns: int
    frameset_host_recv_ns: int
    # 有用统计
    invalid_depth_pixels: int = 0
    aligned_invalid_pixels: int = 0

    @property
    def pair_dt_ms(self) -> float:
        return (self.color_device_ts_ms - self.depth_device_ts_ms)


@dataclass
class CameraModel:
    """相机标定参数与实际流配置快照。"""

    serial: str
    firmware_version: str
    calibration_id: str
    color_intrinsics: Dict[str, Any]
    depth_intrinsics: Dict[str, Any]
    depth_to_color_extrinsics: Dict[str, Any]
    color_to_depth_extrinsics: Dict[str, Any]
    depth_scale_m: float
    actual_streams: Dict[str, Any]
    requested_streams: Dict[str, Any]
    spec_match: bool
    spec_mismatch_reason: Optional[str]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "device": {
                "name": "Intel RealSense D435i",
                "serial": self.serial,
                "firmware_version": self.firmware_version,
                "sdk": "pyrealsense2",
            },
            "color_intrinsics": self.color_intrinsics,
            "depth_intrinsics": self.depth_intrinsics,
            "depth_to_color_extrinsics": self.depth_to_color_extrinsics,
            "color_to_depth_extrinsics": self.color_to_depth_extrinsics,
            "depth_scale_m": self.depth_scale_m,
            "depth_formula": "depth_m = raw_depth_uint16 * depth_scale_m ; raw==0 表示无有效测量",
            "actual_streams": self.actual_streams,
            "requested_streams": self.requested_streams,
            "spec_match": self.spec_match,
            "spec_mismatch_reason": self.spec_mismatch_reason,
            "warnings": self.warnings,
            "optical_frames": {
                "color": "camera_color_optical_frame (RGB 光学系, +z 前, +x 右, +y 下)",
                "depth": "camera_depth_optical_frame (Depth 光学系)",
                "aligned_depth_frame": "camera_color_optical_frame (对齐到 RGB)",
            },
        }


class CameraSpecError(RuntimeError):
    """请求的流规格无法满足。"""


class RealsenseCamera:
    def __init__(
        self,
        *,
        serial: Optional[str] = None,
        color_width: int = 1280,
        color_height: int = 720,
        color_fps: int = 30,
        color_format: str = "bgr8",
        depth_width: int = 1280,
        depth_height: int = 720,
        depth_fps: int = 30,
        depth_format: str = "z16",
        allow_spec_downgrade: bool = False,
        warmup_frames: int = 30,
        frame_timeout_ms: int = 5000,
        calibration_id: Optional[str] = None,
    ) -> None:
        self.serial = serial
        self.requested = {
            "color": {"width": color_width, "height": color_height, "fps": color_fps, "format": color_format},
            "depth": {"width": depth_width, "height": depth_height, "fps": depth_fps, "format": depth_format},
        }
        self.allow_spec_downgrade = bool(allow_spec_downgrade)
        self.warmup_frames = int(warmup_frames)
        self.frame_timeout_ms = int(frame_timeout_ms)
        self.requested_calibration_id = calibration_id
        self.model: Optional[CameraModel] = None
        self._pipeline = None
        self._align = None
        self._profile = None
        self._seq = 0
        self._prime_discarded = 0
        self.counters: Dict[str, Any] = {
            "framesets": 0,
            "color_frames": 0,
            "depth_frames": 0,
            "missing_color": 0,
            "missing_depth": 0,
            "frameset_timeouts": 0,
            "invalid_depth_pixels_total": 0,
            "first_host_ns": None,
            "last_host_ns": None,
            "dropped_hint": 0,
            "frame_number_gaps": 0,
            "last_color_frame_number": None,
            "last_depth_frame_number": None,
            "max_frameset_gap_ms": 0.0,
        }

    # -------------------------------------------------------------- 设备与规格
    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        rs = _rs()
        out = []
        for dev in rs.context().query_devices():
            info = {"name": dev.get_info(rs.camera_info.name)}
            for key in ("serial_number", "firmware_version", "product_line", "usb_type_descriptor"):
                try:
                    info[key] = dev.get_info(getattr(rs.camera_info, key))
                except Exception:
                    info[key] = None
            out.append(info)
        return out

    @staticmethod
    def supported_profiles(serial: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        rs = _rs()
        result: Dict[str, List[Dict[str, Any]]] = {"color": [], "depth": []}
        for dev in rs.context().query_devices():
            if serial and dev.get_info(rs.camera_info.serial_number) != serial:
                continue
            for sensor in dev.sensors:
                name = sensor.get_info(rs.camera_info.name)
                key = "color" if "RGB" in name else ("depth" if "Stereo" in name else None)
                if key is None:
                    continue
                for prof in sensor.get_stream_profiles():
                    try:
                        video = prof.as_video_stream_profile()
                    except Exception:
                        continue
                    result[key].append(
                        {
                            "width": video.width(),
                            "height": video.height(),
                            "fps": prof.fps(),
                            "format": str(prof.format()).replace("format.", ""),
                        }
                    )
        for key in result:
            result[key] = sorted(
                {tuple(sorted(d.items())): d for d in result[key]}.values(),
                key=lambda d: (d["width"], d["height"], d["fps"], d["format"]),
            )
        return result

    def _check_spec_available(self) -> None:
        supported = self.supported_profiles(self.serial)
        for key in ("color", "depth"):
            want = self.requested[key]
            match = [
                p
                for p in supported[key]
                if p["width"] == want["width"]
                and p["height"] == want["height"]
                and p["fps"] == want["fps"]
                and p["format"] == want["format"]
            ]
            if not match:
                raise CameraSpecError(
                    f"{key} 请求规格 {want} 不在设备支持列表中；"
                    f"同分辨率可用选项: "
                    f"{[p for p in supported[key] if p['width'] == want['width'] and p['height'] == want['height']]}"
                )

    # -------------------------------------------------------------- 生命周期
    def open(self) -> CameraModel:
        rs = _rs()
        self._check_spec_available()
        ctx = rs.context()
        devices = list(ctx.query_devices())
        if not devices:
            raise RuntimeError("未发现 RealSense 设备")
        if self.serial:
            devices = [d for d in devices if d.get_info(rs.camera_info.serial_number) == self.serial]
            if not devices:
                raise RuntimeError(f"未找到序列号为 {self.serial} 的设备")
        dev = devices[0]
        serial = dev.get_info(rs.camera_info.serial_number)
        try:
            firmware = dev.get_info(rs.camera_info.firmware_version)
        except Exception:
            firmware = "unknown"

        cfg = rs.config()
        if self.serial:
            # 必须把 pipeline 绑定到指定序列号；否则多相机时第二个 pipeline
            # 可能再次抢占第一台设备，表现为 VIDIOC_S_FMT/Device busy。
            cfg.enable_device(self.serial)
        c = self.requested["color"]
        d = self.requested["depth"]
        cfg.enable_stream(rs.stream.color, c["width"], c["height"], FORMAT_MAP[c["format"]], c["fps"])
        cfg.enable_stream(rs.stream.depth, d["width"], d["height"], FORMAT_MAP[d["format"]], d["fps"])

        self._pipeline = rs.pipeline()
        self._profile = self._pipeline.start(cfg)
        self._align = rs.align(rs.stream.color)

        actual = self._collect_actual_streams()
        mismatch = self._spec_mismatch(actual)
        if mismatch and not self.allow_spec_downgrade:
            self.close()
            raise CameraSpecError(
                "实际生效的流配置与请求不一致，且 allow_spec_downgrade=false："
                + "; ".join(mismatch)
            )

        depth_sensor = dev.first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())

        color_prof = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_prof = self._profile.get_stream(rs.stream.depth).as_video_stream_profile()
        warnings: List[str] = []
        if mismatch:
            warnings.append("规格不符但已按配置允许降级: " + "; ".join(mismatch))

        self.model = CameraModel(
            serial=serial,
            firmware_version=firmware,
            calibration_id=self.requested_calibration_id or f"d435i-{serial}",
            color_intrinsics=intrinsics_to_dict(rs, color_prof.get_intrinsics()),
            depth_intrinsics=intrinsics_to_dict(rs, depth_prof.get_intrinsics()),
            depth_to_color_extrinsics=extrinsics_to_dict(depth_prof.get_extrinsics_to(color_prof)),
            color_to_depth_extrinsics=extrinsics_to_dict(color_prof.get_extrinsics_to(depth_prof)),
            depth_scale_m=depth_scale,
            actual_streams=actual,
            requested_streams=self.requested,
            spec_match=not mismatch,
            spec_mismatch_reason="; ".join(mismatch) if mismatch else None,
            warnings=warnings,
        )
        for _ in range(max(0, self.warmup_frames)):
            self._pipeline.wait_for_frames(self.frame_timeout_ms)
        self._prime_aligned_pair()
        return self.model

    def _prime_aligned_pair(self, tolerance_ms: float = 10.0, max_frames: int = 45) -> int:
        """丢弃帧集直到 color/depth 设备时间戳对齐。

        实测 D435i 刚启动时，预热后的第一个帧集里 depth 时间戳可能比 color 落后
        1.2s（depth 帧号还从 4 回落到 1），若直接当样本会对齐失败并被标无效。
        这里先把这类启动期帧丢掉，返回丢弃的帧数，让首个样本就是干净配对。
        """
        discarded = 0
        for _ in range(max(0, int(max_frames))):
            try:
                frames = self._pipeline.wait_for_frames(self.frame_timeout_ms)
            except Exception:
                break
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                discarded += 1
                continue
            if abs(float(color.get_timestamp()) - float(depth.get_timestamp())) <= float(tolerance_ms):
                self._prime_discarded = discarded
                return discarded
            discarded += 1
        self._prime_discarded = discarded
        return discarded

    def _collect_actual_streams(self) -> Dict[str, Any]:
        rs = _rs()
        cp = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        dp = self._profile.get_stream(rs.stream.depth).as_video_stream_profile()
        return {
            "color": {
                "width": cp.width(),
                "height": cp.height(),
                "fps": cp.fps(),
                "format": str(cp.format()).replace("format.", ""),
                "stream_index": int(cp.stream_index()),
                "unique_id": int(getattr(cp, "unique_id", lambda: -1)()),
            },
            "depth": {
                "width": dp.width(),
                "height": dp.height(),
                "fps": dp.fps(),
                "format": str(dp.format()).replace("format.", ""),
                "stream_index": int(dp.stream_index()),
                "unique_id": int(getattr(dp, "unique_id", lambda: -1)()),
            },
        }

    def _spec_mismatch(self, actual: Dict[str, Any]) -> List[str]:
        out = []
        for key in ("color", "depth"):
            want = self.requested[key]
            got = actual[key]
            for field in ("width", "height", "fps", "format"):
                if want[field] != got[field]:
                    out.append(f"{key}.{field}: 请求 {want[field]} 实际 {got[field]}")
        return out

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._align = None
        self._profile = None

    # -------------------------------------------------------------- 取帧
    def read(self, timeout_ms: Optional[int] = None) -> Optional[FramePair]:
        rs = _rs()
        if self._pipeline is None:
            raise RuntimeError("相机未打开")
        try:
            frames = self._pipeline.wait_for_frames(self.frame_timeout_ms if timeout_ms is None else timeout_ms)
        except Exception:
            self.counters["frameset_timeouts"] += 1
            return None
        host_after = time.time_ns()
        self.counters["framesets"] += 1
        self._note_gap(host_after)

        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color:
            self.counters["missing_color"] += 1
            return None
        if not depth:
            self.counters["missing_depth"] += 1
            return None
        self.counters["color_frames"] += 1
        self.counters["depth_frames"] += 1

        aligned_frames = self._align.process(frames)
        aligned_depth = aligned_frames.get_depth_frame()

        color_bgr = np.asanyarray(color.get_data())
        depth_raw = np.asanyarray(depth.get_data())
        if aligned_depth:
            depth_aligned = np.asanyarray(aligned_depth.get_data())
        else:
            depth_aligned = np.zeros((0, 0), dtype=np.uint16)

        cnum = int(color.get_frame_number())
        dnum = int(depth.get_frame_number())
        if self.counters["last_color_frame_number"] is not None and cnum != self.counters["last_color_frame_number"] + 1:
            self.counters["frame_number_gaps"] += 1
        self.counters["last_color_frame_number"] = cnum
        self.counters["last_depth_frame_number"] = dnum

        self._seq += 1
        pair = FramePair(
            seq=self._seq,
            color_bgr=color_bgr,
            depth_raw_u16=depth_raw,
            depth_aligned_u16=depth_aligned,
            color_device_ts_ms=float(color.get_timestamp()),
            depth_device_ts_ms=float(depth.get_timestamp()),
            color_ts_domain=_timestamp_domain_name(rs, color.get_frame_timestamp_domain()),
            depth_ts_domain=_timestamp_domain_name(rs, depth.get_frame_timestamp_domain()),
            color_frame_number=cnum,
            depth_frame_number=dnum,
            aligned_frame_number=int(aligned_depth.get_frame_number()) if aligned_depth else None,
            # 主机接收时间：RealSense 以 frameset 为单位交付，无法分别观测
            # color/depth 各自的到达时刻，因此两路都用 frameset 交付时刻
            # (host_after)，这是保守且诚实的口径，不伪造逐流到达时间。
            color_host_recv_ns=host_after,
            depth_host_recv_ns=host_after,
            frameset_host_recv_ns=host_after,
            invalid_depth_pixels=int(np.count_nonzero(depth_raw == 0)),
            aligned_invalid_pixels=int(np.count_nonzero(depth_aligned == 0)) if depth_aligned.size else 0,
        )
        self.counters["invalid_depth_pixels_total"] += pair.invalid_depth_pixels
        return pair

    def _note_gap(self, host_ns: int) -> None:
        last = self.counters["last_host_ns"]
        if last is not None:
            gap_ms = (host_ns - int(last)) / 1e6
            if gap_ms > self.counters["max_frameset_gap_ms"]:
                self.counters["max_frameset_gap_ms"] = gap_ms
            expected = 1000.0 / max(1, self.requested["color"]["fps"])
            if gap_ms > 2.5 * expected:
                self.counters["dropped_hint"] += 1
        else:
            self.counters["first_host_ns"] = host_ns
        self.counters["last_host_ns"] = host_ns

    # -------------------------------------------------------------- 落盘
    @staticmethod
    def save_frames(
        pair: FramePair,
        rgb_dir: Optional[Path],
        raw_dir: Path,
        aligned_dir: Optional[Path],
        stem: str,
        *,
        png_compression: int = 1,
    ) -> Dict[str, Optional[str]]:
        """保存 RGB / 原始深度 / 对齐深度为无损 PNG（uint16 深度不做任何缩放）。

        png_compression 只影响编码耗时（0-9 都是无损），默认 1 以保证 30 fps 落盘。
        """
        import cv2

        params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]
        out: Dict[str, Optional[str]] = {"rgb": None, "depth_raw": None, "depth_aligned": None}
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{stem}.png"
        if not cv2.imwrite(str(raw_path), pair.depth_raw_u16, params):
            raise RuntimeError(f"原始深度写入失败: {raw_path}")
        out["depth_raw"] = raw_path.name
        if pair.depth_aligned_u16.size:
            if aligned_dir is None:
                raise RuntimeError("对齐深度非空但未提供输出目录")
            aligned_dir = Path(aligned_dir)
            aligned_dir.mkdir(parents=True, exist_ok=True)
            a_path = aligned_dir / f"{stem}.png"
            if not cv2.imwrite(str(a_path), pair.depth_aligned_u16, params):
                raise RuntimeError(f"对齐深度写入失败: {a_path}")
            out["depth_aligned"] = a_path.name
        if rgb_dir is not None:
            rgb_dir = Path(rgb_dir)
            rgb_dir.mkdir(parents=True, exist_ok=True)
            c_path = rgb_dir / f"{stem}.png"
            if not cv2.imwrite(str(c_path), pair.color_bgr, params):
                raise RuntimeError(f"RGB 写入失败: {c_path}")
            out["rgb"] = c_path.name
        return out

    def diagnostics(self) -> Dict[str, Any]:
        c = self.counters
        first = c["first_host_ns"]
        last = c["last_host_ns"]
        span = (int(last) - int(first)) / 1e9 if first and last else None
        return {
            "framesets": c["framesets"],
            "color_frames": c["color_frames"],
            "depth_frames": c["depth_frames"],
            "measured_color_rate_hz": (c["color_frames"] / span) if span else None,
            "measured_depth_rate_hz": (c["depth_frames"] / span) if span else None,
            "missing_color": c["missing_color"],
            "missing_depth": c["missing_depth"],
            "frameset_timeouts": c["frameset_timeouts"],
            "frame_number_gaps": c["frame_number_gaps"],
            "max_frameset_gap_ms": c["max_frameset_gap_ms"],
            "dropped_frames_hint": c["dropped_hint"],
            "invalid_depth_pixels_total": c["invalid_depth_pixels_total"],
            "span_s": span,
            "spec_match": self.model.spec_match if self.model else None,
            "warmup_frames": self.warmup_frames,
            "prime_discarded_frames": self._prime_discarded,
        }