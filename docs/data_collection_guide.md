# 双 D435i + PiPER：逐场景采集 episode

本指南对应 `capture-lerobot-interactive`。一个场景的一次演示保存为 **1 个 episode**，
不是一帧。目标是采集约 150 个 episode，每个 episode 不限时间，由你按键结束。
程序读取机械臂反馈，不会自动移动机械臂或回起始位。

## 1. 采集前准备

- 打开机械臂电源，确认 `can0` 已配置为 1000000 bit/s 且有机械臂反馈。
- 连接两台 D435i，尽量使用主板直连 USB 3 接口；本机曾因线材/接口或设备状态异常而断流。
- 退出其他采集程序。在调参页面点击“停止预览 · 释放相机”，或在调参服务终端按 Ctrl-C。
  **只关闭浏览器页签不会释放相机。**
- 打开一个正常的交互终端，保持输入焦点在该终端；按键无需回车，`q` 使用英文小写。

本机已安装 `.venv-lerobot`。新环境安装步骤见 [README 的 LeRobot 环境说明](../README.md#10-lerobotdataset-v3第二阶段)。

检查设备（只枚举，不开始采集）：

```bash
cd /home/robot/shucai1
.venv-lerobot/bin/python - <<'PYTHON'
import pyrealsense2 as rs
for d in rs.context().query_devices():
    print(d.get_info(rs.camera_info.serial_number),
          "USB", d.get_info(rs.camera_info.usb_type_descriptor))
PYTHON
```

应同时出现以下两台；设备被识别不等于已验证开流和机械臂同步成功。

| 角色 | 配置序列号 |
| --- | --- |
| 腕部 | `241122071942` |
| 第三人称 | `042222071742` |

## 2. 只需运行一次的采集命令

```bash
cd /home/robot/shucai1
.venv-lerobot/bin/python -m piper_capture.cli \
  --config configs/lerobot_v3_two_d435i_150ep.json \
  capture-lerobot-interactive --episodes 150
```

启动后不会立刻录制，而是显示“等待开始”。不要为每个场景重复启动命令。

## 3. 每个场景怎么操作

1. **摆好当前场景**，准备好物体、机械臂和相机视角。
2. **按空格**启动当前 episode。相机、机械臂和写入器需要初始化；等子进程显示
   “开始采集”和“已写入 … 样本”后再开始演示。按空格后立即出现的“已开始当前 episode”
   只表示启动请求已发出。
3. 完成本次演示后，**先按 `q`**，结束当前 episode，没有固定时长。
4. **等待“当前 episode 已保存”及下一次“等待开始”**。按 `q` 后还要完成视频编码、
   数据写入和设备关闭，不要立即关闭终端或再次按键。
5. **这时再更换场景**，按空格开始下一个 episode，重复上述步骤。

**顺序：摆场景 → 空格 → 等待开始采集 → 演示 → q → 等待保存 → 换场景 → 空格。**
不要先换场景再按 `q`，否则换场景的过程也会被录入上一条。

| 当前阶段 | 空格 | `q` | Ctrl-C |
| --- | --- | --- | --- |
| 等待开始 | 启动下一条 | 退出整个交互程序 | 退出整个交互程序 |
| 正在采集 | 无操作 | 结束并保存当前条，然后继续等待 | 请求结束当前条并退出程序 |
| 正在保存 | 等待完成后再操作 | 等待完成后再操作 | 建议等待保存完再退出 |

正常按 `q` 保存时，子进程 JSON 可能显示 `status: "aborted"`，这是共用 Ctrl-C 收尾逻辑的标记，
不表示丢弃数据。应同时确认 `frames_written > 0` 和交互程序提示“当前 episode 已保存”。
`status: "failed"` 或 `failed_*` 表示失败，不计入本轮成功次数。

## 4. 150 条、退出和下次继续

`--episodes 150` 表示 **本次运行再成功采集 150 条**，不是数据集的总数上限。
界面“已完成/累计”也是本次运行的计数；程序重启后从 0 重新计数，但数据集继续追加。
完成指定数量后自动退出。也可不写 `--episodes`，持续等待按键，直到手动退出。

提前结束时，先按 `q` 保存当前条，等回到“等待开始”后再按 Ctrl-C。
下次运行同一命令会在同一数据集追加 episode，不覆盖旧条目。比如已经保存 40 条，
计划总共 150 条，下次把命令末尾改成 `--episodes 110`。

旧版本曾拒绝向非空目录写入；当前版本使用 LeRobot 的 `resume()` 追加，并核对帧率、
字段形状和已记录的相机序列号。不兼容的数据集或不完整的目录需要先检查，不能为消除报错直接删除。

## 5. 数据保存在哪里，怎么确认数量

当前命令使用 `configs/lerobot_v3_two_d435i_150ep.json`，其 `lerobot.output_root` 为
`dataset/lerobot_v3_150ep`。从项目根目录运行时，完整路径为：

```text
/home/robot/shucai1/dataset/lerobot_v3_150ep/
├── meta/       数据集配置、任务、统计与 episode 索引
├── data/       Parquet：关节反馈、末端位姿、时间戳与同步误差等
└── videos/     两台相机各自的 RGB 和深度视频
```

LeRobot v3 按文件块组织数据，**不是一个 episode 一个独立文件夹**；不要数 MP4 文件来统计 episode。
旧目录 `dataset/lerobot_v3` 是另一份数据集，不属于上面这条命令的输出。
输出位置由配置中的 `lerobot.output_root` 决定，不能用全局 `--dataset-root` 替代。

保存完成后，在项目根目录查询总数：

```bash
.venv-lerobot/bin/python - <<'PYTHON'
import json
from pathlib import Path
p = Path("dataset/lerobot_v3_150ep/meta/info.json")
if p.exists():
    info = json.loads(p.read_text())
    print("已保存 episode：", info["total_episodes"])
    print("已保存总帧数：", info["total_frames"])
else:
    print("尚未创建数据集")
PYTHON
```

结束本轮采集后可用官方加载器检查数据是否能读取：

```bash
.venv-lerobot/bin/python -m piper_capture.cli verify-lerobot \
  --root dataset/lerobot_v3_150ep --repo-id piper_two_d435i
```

这项检查不替代人工检查场景、图像质量或完整的数据质量评估。
采集数据和视频被 `.gitignore` 排除，提交代码不会自动备份数据。

## 6. 当前相机参数与调参

当前 150-episode 配置中，两台相机均请求 RGB/Depth `1280×720@30`，
**RGB 自动曝光关闭、增益 0、曝光设置值 1000**。曝光值按设备选项单位解释，不能直接当作 1000 μs。
深度参数目前未在该配置中强制写入，实际状态以设备回读为准。
采集会记录实际传感器参数；需要调参时见 [相机与红外调参指南](camera_tuning_guide.md)。

`*_tuned.json` 与 `*_150ep.json` 是独立文件。修改 tuned 文件不会自动同步到 150ep 文件。
要修改正式采集参数，可以用正式采集配置打开调参界面，并保存回同一文件：

```bash
python3 -m piper_capture.cli \
  --config configs/lerobot_v3_two_d435i_150ep.json \
  camera-ui --output configs/lerobot_v3_two_d435i_150ep.json
```

完成后先停止预览再采集。保存调参会写入设备支持的实际选项，可能重新带入深度预设；
如果应用参数失败，按下一节检查，不要反复删除数据或换输出目录。

## 7. 启动失败时怎么处理

| 报错或现象 | 检查与处理 |
| --- | --- |
| `CameraSpecError`、可用规格列表为空 | 先运行上面的设备枚举命令。目标相机未连接也会导致空列表；两台都在时再查 USB 速度和支持规格。不要直接认定分辨率不支持。 |
| `Device or resource busy` | 先停止其他预览/采集。若无占用且错误点在 `visual_preset` 等选项，可能是固件或开流状态下写参数失败，并不能仅凭此报错断定另有进程占用。 |
| `Frame didn't arrive within 5000` | 有一路 5 秒未出帧。检查 USB 线和主板接口、逐台测试；仅设备枚举成功不能证明能出帧。 |
| `hwmon ... unknown`、`Broken pipe` | 错误也可能发生在枚举阶段。退出相机程序，将故障相机 USB 拔掉约 10 秒再接好，重查两台序列号；持续失败时排查线材、接口和设备。 |
| `PiPER feedback unavailable`、机械臂无效计数增长 | 检查机械臂供电、CAN 接线、`can0` 状态及真实反馈。相机正常不代表机械臂反馈正常。 |
| `frames_written: 0`、`failed_no_valid_frames` | 当前条未成功保存有效样本；查相机配对与机械臂反馈。不要把失败计作完成一条。 |
| 目录已有内容或规格不一致 | 确认运行的是更新后的代码、配置指向正确数据集。不要删除需要保留的数据；更换采集规格时使用新的输出目录。 |
| `torchcodec` 无法加载，随后提示回退 `pyav` | 本机已知解码器回退提示；继续看后续采集状态，不能仅凭这段警告判定采集失败。 |

失败后交互程序会回到等待状态。排除原因后按空格重试，无需重启整个交互命令。
