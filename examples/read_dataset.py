#!/usr/bin/env python3
"""数据集读取示例：不依赖 piper_capture 包，只按数据字典读取。

用法:
    python3 examples/read_dataset.py <dataset_root> [episode_id] [--images N]

示例:
    python3 examples/read_dataset.py /home/robot/shucai1/dataset
    python3 examples/read_dataset.py /home/robot/shucai1/dataset ep-001 --images 3

要点（对应数据字典 docs/data_dictionary.md）：
  - 所有路径都相对 dataset_root 解析，所以整个数据集搬移后仍可读取；
  - JSONL 逐行读取，容忍中断留下的不完整尾行；
  - 深度是 uint16 无损 PNG，米 = raw * depth_scale_m，raw==0 表示无有效测量；
  - ee_pose 由反馈关节角 FK 得到（ee_pose_source == "feedback_joint_fk"），
    四元数顺序是 wxyz；缺少机械臂状态时字段为 null，不做任何填充。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


# --------------------------------------------------------------------- 基础读取


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """逐行读取 JSONL。末尾若有不完整行（进程被 kill），打印警告并停止。"""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [警告] {path.name}:{lineno} 最后一行不完整，已跳过（{exc.msg}）", file=sys.stderr)
                return


def resolve(root: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else (root / p)


# --------------------------------------------------------------------- 检查项


def check_images(ep_dir: Path, samples: List[Dict[str, Any]], limit: int) -> List[str]:
    """读取前 limit 个样本的图像，核对尺寸/类型/深度尺度。

    注意基准：`cameras.*.path` 相对 **episode 目录**（字段 `cameras.path_base`
    显式写明），metadata 里的 calibration/scene 路径才是相对 dataset root。
    """
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # 没有 cv2 时只检查文件存在性
        print(f"  未安装 opencv，跳过像素级检查（{exc}）")
        return []
    out: List[str] = []
    for s in samples[:limit]:
        if not s.get("valid") and not s.get("cameras"):
            continue
        cams = s.get("cameras") or {}
        color = cams.get("color") or {}
        raw = cams.get("depth_raw") or {}
        aligned = cams.get("depth_aligned") or {}
        p_rgb = resolve(ep_dir, color.get("path"))
        p_raw = resolve(ep_dir, raw.get("path"))
        p_al = resolve(ep_dir, aligned.get("path"))
        img = cv2.imread(str(p_rgb), cv2.IMREAD_COLOR) if p_rgb else None
        d_raw = cv2.imread(str(p_raw), cv2.IMREAD_UNCHANGED) if p_raw else None
        d_al = cv2.imread(str(p_al), cv2.IMREAD_UNCHANGED) if p_al else None
        if img is None or d_raw is None or d_al is None:
            out.append(f"sample {s['sample_id']}: 图像缺失或不可读")
            continue
        ok_shape = img.shape[:2] == d_raw.shape[:2] == d_al.shape[:2]
        ok_dtype = d_raw.dtype == np.uint16 and d_al.dtype == np.uint16
        scale = raw.get("depth_scale_m")
        valid_px = int((d_raw > 0).sum())
        out.append(
            f"sample {s['sample_id']}: rgb={img.shape[1]}x{img.shape[0]}, "
            f"depth_raw={d_raw.shape[1]}x{d_raw.shape[0]}/{d_raw.dtype}, "
            f"aligned={d_al.shape[1]}x{d_al.shape[0]}/{d_al.dtype}, 同尺寸={ok_shape}, "
            f"uint16={ok_dtype}, depth_scale_m={scale}, 有效深度像素={valid_px}"
        )
    return out


def summarize_episode(root: Path, ep_id: str, images: int) -> Dict[str, Any]:
    ep_dir = root / "episodes" / ep_id
    meta_path = ep_dir / "metadata.json"
    print(f"\n=== episode {ep_id} ===")
    if not meta_path.is_file():
        print(f"  缺少 {meta_path}")
        return {}
    meta = read_json(meta_path)
    counters = meta.get("counters") or {}
    print(f"  status={meta.get('status')}  scene={meta.get('scene_id')}")
    print(f"  samples={counters.get('samples')}  invalid={counters.get('invalid_samples')}  "
          f"robot_states={counters.get('robot_states')}")
    print(f"  sample_span_s={meta.get('sample_span_s')}  "
          f"measured_sample_rate_hz={meta.get('measured_sample_rate_hz')}  "
          f"duration_s={meta.get('duration_s')}")
    print(f"  joint_names={meta.get('joint_names')}")
    print(f"  ee_pose_layout={meta.get('ee_pose_layout')}")
    print(f"  calibration_ids={meta.get('calibration_ids')}")

    samples = list(iter_jsonl(ep_dir / "samples.jsonl"))
    print(f"  实际读取到 {len(samples)} 条样本")
    if samples:
        s0 = samples[0]
        print("  首个样本：")
        print(f"    rgb.source_timestamp_ms={((s0.get('timestamps') or {}).get('rgb') or {}).get('source_timestamp_ms')}"
              f"  clock_domain={((s0.get('timestamps') or {}).get('rgb') or {}).get('source_clock_domain')}")
        print(f"    robot_joints.source_timestamp_ns={((s0.get('timestamps') or {}).get('robot_joints') or {}).get('source_timestamp_ns')}"
              f"  clock_source={((s0.get('timestamps') or {}).get('robot_joints') or {}).get('clock_source')}")
        print(f"    sync={json.dumps(s0.get('sync'), ensure_ascii=False)}")
        print(f"    joint_positions_rad={s0.get('joint_positions_rad')}")
        print(f"    ee_pose={s0.get('ee_pose')}  frame={s0.get('ee_pose_frame')}  source={s0.get('ee_pose_source')}")
        print(f"    gripper_feedback_raw={s0.get('gripper_feedback_raw')}  "
              f"gripper_width_mm={s0.get('gripper_width_mm')}  gripper_valid={s0.get('gripper_valid')}")
        print(f"    valid={s0.get('valid')}  invalid_reasons={s0.get('invalid_reasons')}")

        # 实测样本率（用主机接收时间跨度，而不是含开关相机的墙钟时长）
        ts = [s["timestamps"]["reference_host_ns"] for s in samples if (s.get("timestamps") or {}).get("reference_host_ns")]
        if len(ts) > 1:
            span = (max(ts) - min(ts)) / 1e9
            print(f"  实测样本率 = {(len(ts) - 1) / span:.3f} Hz（跨度 {span:.3f} s）")

        # 同步误差统计
        dts = [s.get("robot_joints_dt_ms") for s in samples if s.get("robot_joints_dt_ms") is not None]
        if dts:
            print(f"  关节同步误差 |dt|: max={max(abs(v) for v in dts):.3f} ms, n={len(dts)}")
        else:
            print("  关节同步误差：无可用关节状态（该 episode 未采集机械臂或采集时机械臂无反馈）")

        for line in check_images(ep_dir, samples, images):
            print(f"  图像检查 -> {line}")

    states_path = ep_dir / "robot_states.jsonl"
    n_states = sum(1 for _ in iter_jsonl(states_path)) if states_path.is_file() else 0
    print(f"  robot_states.jsonl 行数={n_states}")

    cmds_path = ep_dir / "commands.jsonl"
    if cmds_path.is_file():
        for rec in iter_jsonl(cmds_path):
            print(f"  commands.jsonl: command_recording_enabled={rec.get('command_recording_enabled')} "
                  f"({rec.get('reason')})")
            break

    ext = meta.get("external_videos") or []
    if ext:
        for v in ext:
            print(f"  第三人称视频: {v.get('role')} -> {v.get('stored_path_relative_to_dataset')} "
                  f"sync={v.get('synchronization_status')}")
    else:
        print("  第三人称视频: 未导入（不影响机械臂与相机数据读取）")
    return meta


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="数据集读取示例")
    ap.add_argument("dataset_root")
    ap.add_argument("episode_id", nargs="?")
    ap.add_argument("--images", type=int, default=2, help="每个 episode 做像素检查的样本数")
    args = ap.parse_args(argv)

    root = Path(args.dataset_root).expanduser().resolve()
    print(f"数据集根目录: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        print(f"缺少 {manifest_path}", file=sys.stderr)
        return 2
    manifest = read_json(manifest_path)
    print(f"schema_version={manifest.get('schema_version')}  dataset_id={manifest.get('dataset_id')}")
    print(f"joint_names={manifest.get('joint_names')}")
    print(f"units={json.dumps(manifest.get('units'), ensure_ascii=False)}")
    print(f"episodes: {[e['episode_id'] for e in manifest.get('episodes', [])]}")

    if args.episode_id:
        targets = [args.episode_id]
    else:
        targets = [e["episode_id"] for e in manifest.get("episodes", [])]
    if not targets:
        print("没有 episode 可读", file=sys.stderr)
        return 1
    for ep in targets:
        summarize_episode(root, ep, args.images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())