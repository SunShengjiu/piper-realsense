"""Keyboard-driven LeRobot capture loop.

The normal capture command uses SIGINT to finish one episode.  This wrapper
keeps one terminal session open: press Space to start an episode and ``q`` to
finish it.  The child capture process receives SIGINT so its normal cleanup and
video finalization still run.
"""
from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import termios
import tty
from typing import Optional


def _read_key(fd: int, timeout: float = 0.2) -> Optional[str]:
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return None
    data = os.read(fd, 1)
    return data.decode(errors="ignore") if data else None


def run_interactive(
    *,
    config: Optional[str],
    dataset_root: Optional[str],
    task: str,
    episodes: Optional[int],
) -> int:
    if not sys.stdin.isatty():
        raise RuntimeError("交互采集需要在终端中运行，不能从管道输入")

    command = [sys.executable, "-m", "piper_capture.cli"]
    if config:
        command.extend(["--config", config])
    if dataset_root:
        command.extend(["--dataset-root", dataset_root])
    command.extend(["capture-lerobot", "--task", task])

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    completed = 0
    try:
        tty.setcbreak(fd)
        print("\n交互采集已就绪：按空格开始一个 episode，按 q 结束当前 episode。", flush=True)
        print("在等待下一次开始时按 Ctrl-C 退出。", flush=True)
        while episodes is None or completed < episodes:
            print(f"\n等待开始（已完成 {completed} 个）...", flush=True)
            while True:
                key = _read_key(fd)
                if key == " ":
                    break
                if key in ("q", "\x03"):
                    print("已退出交互采集。", flush=True)
                    return 0

            print("已开始当前 episode；场景完成后按 q。", flush=True)
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
            q_sent = False
            try:
                while child.poll() is None:
                    key = _read_key(fd)
                    if key == "q":
                        print("正在结束当前 episode，请等待视频写入完成...", flush=True)
                        child.send_signal(signal.SIGINT)
                        q_sent = True
                        break
                    if key == "\x03":
                        child.send_signal(signal.SIGINT)
                        q_sent = True
                        break
                return_code = child.wait()
            except KeyboardInterrupt:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                return_code = child.wait()
                print("已退出交互采集。", flush=True)
                return 130

            if q_sent and return_code == 0:
                completed += 1
                print(f"当前 episode 已保存（累计 {completed} 个）。", flush=True)
            elif return_code == 0:
                # A child can finish on its own only if a duration was added to
                # the command in the future; still count a successful episode.
                completed += 1
                print(f"当前 episode 已完成（累计 {completed} 个）。", flush=True)
            else:
                print(f"当前 episode 失败，退出码 {return_code}；可按空格重试。", flush=True)
    except KeyboardInterrupt:
        print("\n已退出交互采集。", flush=True)
        return 130
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return 0
