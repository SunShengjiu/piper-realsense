"""JSON / JSONL 读写工具。

写入策略：
  - 单个 JSON 文件先写临时文件再 os.replace，避免中断留下半个文件。
  - JSONL 逐行写入并 flush，中断后已完成的行仍然可解析（每行独立 JSON 对象）。
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence


def write_json(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=indent, sort_keys=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class JsonlWriter:
    """带 flush 的 JSONL 追加写入器，支持 with 语句与显式 close。"""

    def __init__(self, path: Path | str, *, fsync_every: int = 0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._fsync_every = max(0, int(fsync_every))
        self._since_sync = 0
        self.lines_written = 0
        self.closed = False

    def write(self, record: Dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("JsonlWriter 已关闭")
        self._fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._fh.flush()
        self.lines_written += 1
        self._since_sync += 1
        if self._fsync_every and self._since_sync >= self._fsync_every:
            os.fsync(self._fh.fileno())
            self._since_sync = 0

    def flush(self) -> None:
        if not self.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        finally:
            self._fh.close()
            self.closed = True

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_jsonl(path: Path | str) -> Iterator[Dict[str, Any]]:
    """逐行读取 JSONL；容忍中断留下的最后一行不完整内容并报告。"""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} 损坏的 JSONL 行: {exc}") from exc


def load_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def count_jsonl_lines(path: Path | str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def relpath(target: Path | str, base: Path | str) -> str:
    """返回相对 base 的 POSIX 风格路径；若不在 base 下则返回绝对路径。"""
    target = Path(target).resolve()
    base = Path(base).resolve()
    try:
        return target.relative_to(base).as_posix()
    except ValueError:
        return target.as_posix()


def resolve_path(value: str, base: Path | str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (Path(base) / p).resolve()


def matrix_to_list(T: Any) -> List[List[float]]:
    return [[float(v) for v in row] for row in T]


def list_to_matrix(rows: Sequence[Sequence[float]]) -> Any:
    import numpy as np

    return np.asarray(rows, dtype=float)