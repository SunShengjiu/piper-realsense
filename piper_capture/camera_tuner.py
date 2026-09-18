"""Local two-camera tuning UI. Start with `piper_capture.cli camera-ui`."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import queue
import secrets
import shlex
import tempfile
import threading
import time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

from .camera import RealsenseCamera, _rs
from .camera_options import describe_options, saved_options, set_option
from .config import load_config

ROLES = ('wrist', 'third_person')


def infrared_stats(frame):
    """Measure sensor Y8 values before JPEG; never normalize away clipping."""
    if frame is None or not frame.size:
        return None
    return dict(mean=round(float(frame.mean()), 1),
                dark_percent=round(float(np.mean(frame <= 5)) * 100, 1),
                saturated_percent=round(float(np.mean(frame >= 250)) * 100, 1))


def validate_specs(specs):
    if set(specs) != set(ROLES):
        raise ValueError('必须同时配置腕部和第三人称两台相机')
    if not all(specs[r].get('serial') for r in ROLES):
        raise ValueError('请选择两台相机')
    if specs['wrist']['serial'] == specs['third_person']['serial']:
        raise ValueError('两种角色不能使用同一个序列号')
    for role in ROLES:
        for kind, fmt in [('color', 'bgr8'), ('depth', 'z16')]:
            p = specs[role]['streams'][kind]
            if p['format'] != fmt or p['fps'] != 30:
                raise ValueError('当前采集流程使用 BGR8 / Z16、30 fps；请选择支持的规格')
            if p['width'] <= 0 or p['height'] <= 0:
                raise ValueError('分辨率必须为正数')


def save_configuration(base, specs, output):
    """Atomic save preserves unrelated config; validation happens before replacement."""
    validate_specs(specs)
    cfg = copy.deepcopy(base)
    for role in ROLES:
        cfg.setdefault('camera', {}).setdefault(role, {}).update(copy.deepcopy(specs[role]))
    load_config(overrides=cfg)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent, delete=False) as f:
            temp = f.name
            json.dump(cfg, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, output)
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)
    return cfg


class CameraWorker:
    """One owner thread per SDK pipeline; parameter writes run between reads."""
    def __init__(self, spec):
        self.spec = copy.deepcopy(spec)
        self.commands = queue.Queue()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.status = {'phase': 'starting', 'error': '', 'fps': 0, 'options': {}, 'frames': 0}
        self.images = {}
        self.thread = threading.Thread(target=self.run, daemon=True)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.status)

    def call(self, action, *args):
        if self.snapshot()['phase'] != 'running':
            raise RuntimeError('相机尚未就绪，请等待预览或检查连接错误')
        future = Future()
        self.commands.put((future, action, args))
        try:
            return future.result(timeout=15)
        except TimeoutError:
            future.cancel()
            raise RuntimeError('相机操作超时，请停止后重新连接')

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=12)
        if self.thread.is_alive():
            raise RuntimeError('相机仍在关闭，请稍后重试')

    def run(self):
        import cv2
        spec = self.spec
        c, d = spec['streams']['color'], spec['streams']['depth']
        camera = RealsenseCamera(
            serial=spec['serial'], sensor_options=spec.get('sensor_options'),
            color_width=c['width'], color_height=c['height'], color_fps=c['fps'], color_format=c['format'],
            depth_width=d['width'], depth_height=d['height'], depth_fps=d['fps'], depth_format=d['format'],
            warmup_frames=5, frame_timeout_ms=1000,
            enable_infrared=True,
        )
        try:
            model = camera.open()
            rs, dev = _rs(), camera._profile.get_device()
            with self.lock:
                self.status.update(phase='running', options=describe_options(dev, rs), streams=model.actual_streams)
            started, last_preview, last_frame = time.monotonic(), 0.0, time.monotonic()
            frames = 0
            while not self.stop_event.is_set():
                while not self.commands.empty():
                    future, action, args = self.commands.get_nowait()
                    if not future.set_running_or_notify_cancel():
                        continue
                    try:
                        if action == 'set':
                            set_option(dev, rs, *args)
                            result = describe_options(dev, rs)
                        elif action == 'save':
                            result = saved_options(dev, rs)
                        else:
                            raise ValueError('未知相机操作')
                        with self.lock:
                            self.status['options'] = describe_options(dev, rs)
                        future.set_result(result)
                    except Exception as exc:
                        future.set_exception(exc)
                pair = camera.read(timeout_ms=1000)
                now = time.monotonic()
                if pair is None:
                    if now - last_frame > 5:
                        raise RuntimeError('超过 5 秒没有图像，请检查 USB 连接或重新连接相机')
                    continue
                last_frame = now
                frames += 1
                if now - last_preview < .10:
                    continue
                last_preview = now
                depth = pair.depth_aligned_u16
                if not depth.size:
                    raise RuntimeError('深度对齐没有输出，请重新连接相机')
                metres = depth.astype(np.float32) * model.depth_scale_m
                # Fixed display range makes exposure/depth comparisons consistent.
                previews = {'color': pair.color_bgr, 'infrared_left': pair.infrared_left_u8,
                            'infrared_right': pair.infrared_right_u8}
                for kind, values in [('depth', depth), ('depth_raw', pair.depth_raw_u16)]:
                    scaled = values.astype(np.float32) * model.depth_scale_m / 3.0
                    colored = cv2.applyColorMap(np.uint8(np.clip(scaled, 0, 1) * 255), cv2.COLORMAP_TURBO)
                    colored[values == 0] = 0
                    previews[kind] = colored
                encoded = {}
                for kind, img in previews.items():
                    if img is None:
                        continue
                    scale = min(1.0, 720 / img.shape[1])
                    img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)))
                    ok, data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if ok:
                        encoded[kind] = data.tobytes()
                valid = depth > 0
                center = metres[metres.shape[0]//2, metres.shape[1]//2]
                with self.lock:
                    self.images = encoded
                    self.status.update(fps=round(frames / (now-started), 1), frames=frames,
                                       valid_percent=round(float(valid.mean()) * 100, 1),
                                       raw_valid_percent=round(float(np.mean(pair.depth_raw_u16 > 0)) * 100, 1),
                                       infrared={side: infrared_stats(previews['infrared_' + side])
                                                 for side in ('left', 'right')},
                                       center_m=round(float(center), 3) if center > 0 else None,
                                       last_frame=time.time())
        except Exception as exc:
            with self.lock:
                self.status.update(phase='error', error=str(exc))
        finally:
            camera.close()
            with self.lock:
                if self.status['phase'] != 'error':
                    self.status['phase'] = 'stopped'
            while not self.commands.empty():
                future, _, _ = self.commands.get_nowait()
                if not future.done():
                    future.set_exception(RuntimeError('相机已停止'))


class Tuner:
    def __init__(self, config_path, output_path):
        self.config_path = Path(config_path).resolve()
        self.output_path = Path(output_path).resolve()
        self.base = json.loads(self.config_path.read_text(encoding='utf-8'))
        cfg = load_config(self.config_path)
        if not all(r in cfg['camera'] for r in ROLES):
            raise ValueError('调参界面需要含 camera.wrist 和 camera.third_person 的双相机配置')
        self.specs = {r: copy.deepcopy(cfg['camera'][r]) for r in ROLES}
        validate_specs(self.specs)
        self.workers = {}
        self.operation_lock = threading.RLock()
        self.saved_at = None
        self.dirty = False

    def state(self):
        return dict(specs=copy.deepcopy(self.specs), cameras={r: w.snapshot() for r, w in list(self.workers.items())},
                    source=str(self.config_path), output=str(self.output_path), saved_at=self.saved_at, dirty=self.dirty,
                    command=f'.venv-lerobot/bin/python -m piper_capture.cli --config {shlex.quote(str(self.output_path))} capture-lerobot --duration 5')

    def discover(self):
        return [dict(**d, profiles={k: [p for p in v if p['fps'] == 30 and p['format'] == ('bgr8' if k == 'color' else 'z16')]
                                    for k, v in RealsenseCamera.supported_profiles(d['serial_number']).items()})
                for d in RealsenseCamera.list_devices()]

    def stop(self):
        with self.operation_lock:
            for role, worker in list(self.workers.items()):
                if worker.snapshot()['phase'] == 'running':
                    try:
                        self.specs[role]['sensor_options'] = worker.call('save')
                    except RuntimeError:
                        pass
                worker.close()

    def start(self, specs):
        with self.operation_lock:
            validate_specs(specs)
            # Reopening the same devices retains the actual last successful tuning.
            for role, worker in self.workers.items():
                if worker.snapshot()['phase'] == 'running' and specs[role]['serial'] == worker.spec['serial']:
                    specs[role]['sensor_options'] = worker.call('save')
            self.stop()
            self.specs = copy.deepcopy(specs)
            self.workers = {r: CameraWorker(s) for r, s in specs.items()}
            self.dirty = True
            for worker in self.workers.values():
                worker.thread.start()

    def set(self, role, sensor, name, value):
        with self.operation_lock:
            if role not in self.workers:
                raise ValueError('请先连接相机')
            self.dirty = True
            return self.workers[role].call('set', sensor, name, value)

    def save(self):
        with self.operation_lock:
            if any(r not in self.workers or self.workers[r].snapshot()['phase'] != 'running'
                   or not self.workers[r].snapshot()['frames'] for r in ROLES):
                raise RuntimeError('两台相机都成功预览后才能保存实际参数')
            specs = copy.deepcopy(self.specs)
            for role in ROLES:
                specs[role]['sensor_options'] = self.workers[role].call('save')
            self.base = save_configuration(self.base, specs, self.output_path)
            self.specs = specs
            self.saved_at = time.strftime('%H:%M:%S')
            self.dirty = False
            return self.state()


def serve(config_path, output_path, port=8766, open_browser=True):
    tuner = Tuner(config_path, output_path)
    token = secrets.token_urlsafe(32)
    html = Path(__file__).with_name('camera_tuner.html').read_text(encoding='utf-8').replace('__TOKEN__', token).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, body, content_type='application/json; charset=utf-8'):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == '/':
                    self.reply(200, html, 'text/html; charset=utf-8')
                elif path == '/api/state':
                    self.reply(200, tuner.state())
                elif path == '/api/devices':
                    self.reply(200, tuner.discover())
                elif path.startswith('/frame/'):
                    _, _, role, kind = path.split('/')
                    worker = tuner.workers.get(role)
                    data = None
                    if worker:
                        with worker.lock:
                            if worker.status['phase'] == 'running':
                                data = worker.images.get(kind)
                    self.reply(200 if data else 404, data or b'', 'image/jpeg')
                else:
                    self.reply(404, {'error': '地址不存在'})
            except Exception as exc:
                self.reply(400, {'error': str(exc)})

        def do_POST(self):
            if self.headers.get('X-Tuner-Token') != token:
                self.reply(403, {'error': '请刷新本机调参页面'})
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 <= length <= 100_000:
                    raise ValueError('请求过大')
                payload = json.loads(self.rfile.read(length) or b'{}')
                path = urlparse(self.path).path
                if path == '/api/start':
                    tuner.start(payload['specs'])
                elif path == '/api/option':
                    options = tuner.set(payload['role'], payload['sensor'], payload['name'], payload['value'])
                    self.reply(200, {'options': options})
                    return
                elif path == '/api/save':
                    tuner.save()
                elif path == '/api/stop':
                    tuner.stop()
                else:
                    self.reply(404, {'error': '地址不存在'})
                    return
                self.reply(200, tuner.state())
            except Exception as exc:
                self.reply(400, {'error': str(exc)})

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'双相机调参界面：{url}\n保存位置：{tuner.output_path}\nCtrl-C 退出并释放相机。', flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        tuner.stop()
    return 0
