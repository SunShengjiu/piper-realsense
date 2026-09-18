"""Loopback-only preview and manual sampling controls, sharing one camera."""
from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><meta charset="utf-8"><title>手眼标定预览</title>
<style>body{background:#19202a;color:white;font:20px sans-serif;margin:24px}
img{width:min(100%,1100px);display:block}button{font-size:20px;margin:16px 12px 0 0;padding:12px}
#status{white-space:pre-wrap}</style>
<h2>眼在手上 · 棋盘格标定</h2><img id="preview" alt="等待相机画面">
<p>标定板固定不动；调整机械臂并停稳后记录。保持完整棋盘入镜。</p>
<button onclick="command('record')">记录当前姿态</button>
<button onclick="command('undo')">删除上一帧</button>
<button onclick="command('finish')">结束采样并求解</button><p id="status">正在连接…</p>
<script>
const img=document.getElementById('preview'),status=document.getElementById('status');
img.onload=()=>setTimeout(()=>img.src='/frame.jpg?t='+Date.now(),100);
img.onerror=()=>setTimeout(()=>img.src='/frame.jpg?t='+Date.now(),1000);
img.src='/frame.jpg';
async function command(name){let r=await fetch('/'+name,{method:'POST',headers:{'X-Handeye-Local':'1'}});if(!r.ok)status.textContent=await r.text();}
setInterval(async()=>{try{const r=await fetch('/status');const d=await r.json();status.textContent=d.message;}catch(e){status.textContent='采样服务已停止，请查看终端求解结果。';}},500);
</script>"""


class PreviewCamera:
    def __init__(self, camera, port=8765):
        self.camera = camera
        self.commands = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.condition = threading.Condition()
        self.frame = None
        self.serial = 0
        self.jpeg = None
        self.error = None
        self.message = '等待相机画面'
        self.busy = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, data, kind='text/plain; charset=utf-8'):
                if isinstance(data, str):
                    data = data.encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                path = self.path.split('?')[0]
                if path == '/':
                    self.reply(200, PAGE, 'text/html; charset=utf-8')
                elif path == '/frame.jpg':
                    with owner.condition:
                        jpeg = owner.jpeg
                    self.reply(200 if jpeg else 503, jpeg or b'Waiting', 'image/jpeg')
                elif path == '/status':
                    self.reply(200, json.dumps({'message': owner.message, 'busy': owner.busy}, ensure_ascii=False), 'application/json')
                else:
                    self.reply(404, 'Not found')

            def do_POST(self):
                # Custom header prevents cross-origin form requests; no CORS is enabled.
                if self.headers.get('X-Handeye-Local') != '1':
                    self.reply(403, 'Missing X-Handeye-Local: 1')
                    return
                action = {'/record': '', '/undo': 'd', '/finish': 'q'}.get(self.path)
                if action is None:
                    self.reply(404, 'Not found')
                    return
                with owner.condition:
                    if owner.busy or not owner.commands.empty():
                        self.reply(409, '上一条操作尚未完成')
                        return
                    owner.busy = True
                    owner.commands.put_nowait(action)
                self.reply(202, '已接收，结果请看预览页面\n')

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        self.reader_thread = threading.Thread(target=self._capture, daemon=True)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.reader_thread.start()
        self.server_thread.start()

    def _capture(self):
        import cv2
        last_jpeg = 0.
        try:
            while not self.stop.is_set():
                frame = self.camera.read()
                if frame is None:
                    continue
                jpeg = None
                if time.monotonic() - last_jpeg >= .1:
                    ok, encoded = cv2.imencode('.jpg', frame.color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        jpeg = encoded.tobytes()
                    last_jpeg = time.monotonic()
                with self.condition:
                    self.frame = frame
                    self.serial += 1
                    if jpeg is not None:
                        self.jpeg = jpeg
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.message = f'相机错误：{exc}'
                self.condition.notify_all()

    def read(self):
        with self.condition:
            previous = self.serial
            self.condition.wait_for(lambda: self.serial != previous or self.error is not None, timeout=5.)
            if self.error is not None:
                raise RuntimeError(f'预览相机读取失败: {self.error}') from self.error
            return self.frame if self.serial != previous else None

    def prompt(self, text):
        self.busy = False
        self.message += '\n' + text
        print(text, flush=True)
        while True:
            if self.error is not None:
                raise RuntimeError(f'预览相机读取失败: {self.error}') from self.error
            try:
                action = self.commands.get(timeout=.2)
                with self.condition:
                    self.busy = True
                return action
            except queue.Empty:
                pass

    def report(self, text):
        self.message = str(text)
        print(text, flush=True)

    def close(self):
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.reader_thread.join(timeout=6.)
        self.camera.close()
