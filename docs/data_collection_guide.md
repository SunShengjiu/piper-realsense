# PiPER + D435i 数据采集操作步骤

适用目录：`/home/robot/shucai1`。使用零重力模式手动示教，程序自动记录 RGB、
原始深度、对齐深度、关节角、反馈关节角计算的末端位姿以及夹爪原始反馈。
采集程序不下发运动指令，也不自动切换零重力模式或移动到起始位。

本文用已创建的 `ep-data-20260918-105102` 演示监控、停止和质检命令。
是否仍在采集，以实际进程为准；新一段采集需将命令中的 episode 名换成新名称。

## 1. 确认是否已有采集进程

```bash
cd /home/robot/shucai1
pgrep -af '[p]ython3 -m piper_capture.cli.* capture'
```

有输出时，核对 PID 和 `--episode`，继续监控已有进程，不要重复启动。
没有输出时，可按第 7 节启动新一段采集。

## 2. 监控采集数量

在另一个终端执行：

```bash
cd /home/robot/shucai1
watch -n 2 'wc -l dataset/episodes/ep-data-20260918-105102/samples.jsonl dataset/episodes/ep-data-20260918-105102/robot_states.jsonl'
```

`samples.jsonl` 是图像与机械臂反馈匹配后的样本，目标约 30 条/秒；
`robot_states.jsonl` 保存独立机械臂反馈，数量通常更多。
运行时文件可能有缓冲，计数仅供监控，最终数量以收尾后的元数据为准。

## 3. 监控采集日志

可再打开一个终端执行：

```bash
cd /home/robot/shucai1
tail -f dataset/episodes/ep-data-20260918-105102/logs/session.log
```

关注样本数是否增长、无效样本数及相机/CAN 错误。
在 `watch` 或 `tail -f` 的终端按 `Ctrl-C` **只会退出监控，不会停止采集**。

## 4. 进行手动示教

通过机械臂示教按钮进入零重力模式，再手动完成任务动作。
数采起始姿态约定为关节 1～6 `[90°, 0°, 0°, 0°, 0°, 0°]`；采集命令只记录
这一配置和实际反馈，不自动回起始位。零位仍为六关节全零。

当前保存的是实际运动反馈，没有独立的下发动作指令记录。
夹爪未做实测开度校准时仍保存原始反馈，`gripper_width_mm` 为 `null`。

## 5. 结束采集并保存

在**启动采集的终端**按一次 `Ctrl-C`，等待程序完成图像落盘、关闭设备并输出摘要。

如果采集由助手或另一个不可见终端启动，可先找到对应进程：

```bash
pgrep -af '[p]ython3 -m piper_capture.cli.* capture .*--episode ep-data-20260918-105102'
```

核对输出中的 episode 后，把下面的 `12345` 替换为该采集进程的实际 PID：

```bash
kill -INT 12345
```

`SIGINT` 与采集终端里的 `Ctrl-C` 都会触发程序正常收尾。不要使用 `kill -9`，
以免尚未完成的图像写入与元数据收尾被中断。等待该 PID 退出后再启动下一段。
手动中断的 episode 状态可能记为 `aborted`，已完成数据仍会保留，应以质检判断可用性。

## 6. 生成质量报告

采集退出后执行：

```bash
cd /home/robot/shucai1
python3 -m piper_capture.cli quality check --episode ep-data-20260918-105102
python3 -m piper_capture.cli quality dataset
```

报告位置：

```text
dataset/episodes/ep-data-20260918-105102/quality_report.json
dataset/reports/dataset_quality.json
```

结论为 `pass`、`needs_attention` 或 `fail`。全数据集报告也包含历史试采记录，
查看本次情况时以本 episode 的报告为准。

当前配置引用的相机—末端变换是官方模型名义值，状态为
`nominal_unverified`、`valid=false`，尚未完成现场手眼标定；
质检会提示外参未验证。有效图像/关节样本不代表外参已达到精确抓取要求。

## 7. 开始下一段采集

确认旧采集进程已退出，相机未被其他程序占用，然后执行：

```bash
cd /home/robot/shucai1
python3 -m piper_capture.cli \
  --config configs/handeye_checkerboard_eye_in_hand.json \
  capture \
  --scene scene-tabletop \
  --episode ep-$(date +%Y%m%d-%H%M%S) \
  --verbose
```

每段使用新的 episode 名，`--verbose` 会在采集终端显示样本数和速率。
默认持续采集直到 `Ctrl-C`；需要定时结束时，可在命令末尾加 `--duration 60`
表示采集约 60 秒。配置文件虽然含有棋盘参数，执行 `capture` 不要求标定板在画面中。

采集期间不要同时运行 `camera-probe --open`、手眼预览或其他占用 D435i 的程序。
如手眼网页预览仍在运行，可在页面点击“结束采样并求解”，等待其退出后再启动采集。

## 8. 数据保存位置

本次 episode 的绝对路径：

```text
/home/robot/shucai1/dataset/episodes/ep-data-20260918-105102/
```

| 文件/目录 | 内容 |
| --- | --- |
| `samples.jsonl` | 每帧图像路径、关节状态、末端位姿、时间戳与匹配误差 |
| `robot_states.jsonl` | 独立机械臂反馈记录 |
| `rgb/` | 彩色图像 |
| `depth_raw/` | 原始深度图 |
| `depth_aligned/` | 对齐到 RGB 的深度图 |
| `metadata.json` | episode 元数据、配置、标定引用与收尾统计 |
| `logs/session.log` | 采集日志 |
| `quality_report.json` | 执行质检后生成的报告 |

标定文件位于 `dataset/calibrations/`，字段与单位详见
[数据字典](data_dictionary.md)。
