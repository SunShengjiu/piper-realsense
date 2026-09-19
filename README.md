# PiPER + D435i 数据采集 / 手眼标定 / 数据质检

本目录实现 PiPER 机械臂 + RealSense D435i 的数据采集链路：只读读取机械臂关节反馈、
保存 RGB-D 帧对、记录时间戳与同步误差、导出驱动记录，并提供手眼标定、夹爪开度校准、
第三人称视频登记和数据质量检查工具。

**默认全程只读**：普通采集、诊断和标定命令都不下发机械臂或夹爪运动指令，不激活/不修改
CAN 配置，不升级固件，不改关节零位。只有显式加入 `--arm-button-return-zero`，或运行
`robot go-zero --allow-motion`，程序才会驱动机械臂；`gripper probe --allow-motion`
仍是唯一的夹爪主动探测入口。

```
piper_capture/      Python 包（CLI + 各功能模块）
docs/               数据字典
examples/           数据读取示例
dataset/            数据集根目录（默认；大体积数据不提交 Git）
```

## 日常采集入口：空格开始，q 结束

采集 150 个不同场景的 episode，使用下面这一条命令，**每个 episode 不限时**：

```bash
cd /home/robot/shucai1
.venv-lerobot/bin/python -m piper_capture.cli \
  --config configs/lerobot_v3_two_d435i_150ep.json \
  capture-lerobot-interactive --episodes 150 --return-on-q
```

1. 开始前确认两台相机和机械臂连接正常，并停止调参页面的预览。
2. 摆好场景，**按空格**。等待“开始采集”和“已写入 … 样本”后进行演示。
3. 演示结束，按 `q` 结束当前 episode。程序先完成视频/数据保存，再按“重置到待机 → CAN”
   流程，以限速把六个关节移动到配置的采集初始位姿 `[90, 0, 0, 0, 0, 0]°`。
4. 看到“当前 episode 已保存，机械臂已回到配置初始位姿”后再换场景，按空格采集下一条。
5. 提前结束本轮，在“等待开始”时按 Ctrl-C；下次运行可继续追加。

按键无需回车，终端需要保持输入焦点。启用自动回位后，`q` 是每条 episode 的结束键；
程序会在显式运动工作流中完成示教退出和 CAN 切换。
`--episodes 150` 指本次运行成功新增 150 条；若已有 40 条、目标总共 150 条，下次用 `--episodes 110`。

### 按机械臂按钮结束并自动回零

`--return-on-q` 监听交互终端的 `q` 结束事件；`--arm-button-return-zero` 还兼容 PiPER
示教按钮。按钮流程监听 PiPER 状态反馈中的示教记录停止（`teach_status=0x02` 或
`arm_status: 0x0B -> 正常`），也兼容上位机直接发出的示教→CAN切换。按钮触发后，程序
先发送官方要求的 `ResetPiper` 进入待机，再选择 CAN 控制模式，最后用
`MotionCtrl_2 + JointCtrl` 回位。默认速度 20%，到位误差阈值 ±1°，超时 20 秒；检测到
状态异常、模式离开或超时会停止发目标并报告失败。回零完成后仍保持 CAN 模式。

这里的“回零”是**移动到六关节角度 0° 的目标姿态**，不会调用 `JointConfig(set_zero=0xAE)`，
因此不会修改机械臂持久化的电机零点。自定义目标可以加到命令末尾，例如：

```bash
--target-deg 90 0 0 0 0 0
```

需要单独测试回零时，确认机械臂周围无人、急停可用，再运行：

```bash
.venv-lerobot/bin/python -m piper_capture.cli \
  --config configs/lerobot_v3_two_d435i_150ep.json \
  robot go-zero --allow-motion --wait-for-button
```

这个独立命令等待示教按钮结束记录，并会执行重置到待机和 CAN 切换；不加 `--allow-motion`
时只会拒绝执行。`ResetPiper` 可能短暂释放电机使能，执行前应确认机械臂有支撑且急停可用。

数据路径：`/home/robot/shucai1/dataset/lerobot_v3_150ep/`，包含 `meta/`、`data/`、`videos/`。
一个 episode 不是一帧，也不是一个独立目录；总条数见 `meta/info.json` 的 `total_episodes`。
当前配置请求双路 RGB-D 1280×720@30，RGB 增益 0、曝光设置值 1000、自动曝光关闭。

**完整步骤、数量查询、续采和故障处理：[数据采集操作指南](docs/data_collection_guide.md)。**
相机参数调整另见 [相机与红外调参指南](docs/camera_tuning_guide.md)。
下面的旧版 `capture` 命令与 LeRobot 输出格式不同，日常逐场景采集请使用上面的交互命令。

## 1. 运行环境（已实测）

| 项 | 实测值 |
| --- | --- |
| 主机 | Ubuntu 22.04 / Python 3.10.12 |
| ROS | 未安装（`/opt/ros` 不存在、`ROS_DISTRO` 为空）→ 本项目**不依赖 ROS**，用 `piper_sdk` 直连 socketcan |
| 机械臂驱动 | `piper_sdk`（`C_PiperInterface_V2`）+ socketcan `can0`，bitrate 1000000 |
| CAN 适配器 | gs_usb（OpenMoko Geschwister Schneider，`1d50:606f`，USB `3-4:1.0`） |
| 相机 | RealSense D435i，序列号 `241122071942`，固件 `5.13.0.55`，USB 3.2 |
| 相机库 | `pyrealsense2` 2.58.4 |
| 其他依赖 | `opencv-contrib-python-headless` 4.11.0.86、`numpy`、`scipy`、`python-can` |
| ffmpeg | `/home/robot/.local/bin/ffmpeg`（`shutil.which` 找不到；无 ffprobe，第三人称视频探测以 OpenCV 为主） |

官方参考（只作参考，本项目未直接依赖其 ROS 包）：
`agilexrobotics/Agilex-College` 的 `piper/handeye`、`agilexrobotics/handeye_calibration_ros`。
当前环境没有 ROS，官方 ROS 节点不能直接运行，因此标定流程按同样的数学与采样约定
在本项目内独立实现（`cv2.calibrateHandEye` + `cv2.aruco`）。

### 硬件命令必须在宿主环境运行

CAN 与 USB 相机在受限沙箱/容器内不可见（沙箱网络命名空间会屏蔽 `can0`）。
`doctor`、`camera-probe --open`、`capture`、`handeye sample`、`gripper diagnose/probe`
需要在能直接访问 `can0` 和 USB 的宿主 shell 里运行。

## 2. 快速开始

零重力示教的启动、监控、停止、质检与保存路径见
[数据采集操作步骤](docs/data_collection_guide.md)。

```bash
cd /home/robot/shucai1

# 1) 环境检查（只读，写 dataset/reports/doctor.json）
python3 -m piper_capture.cli doctor

# 2) 查询 D435i 支持的流配置并实测帧率
python3 -m piper_capture.cli camera-probe            # 只列配置
python3 -m piper_capture.cli camera-probe --open     # 真开相机实测

# 3) 采集一个 episode（默认只读机械臂；Ctrl-C 中断后已完成数据保留）
python3 -m piper_capture.cli capture --scene scene-tabletop --episode ep-001 --duration 20

# 只采 D435i（机械臂不可用时可用，样本会标 invalid 并给出原因）
python3 -m piper_capture.cli capture --camera-only --scene scene-tabletop --duration 20
```

全局参数：`--config <json>`（与默认配置深度合并）、`--dataset-root <dir>`（覆盖数据集根目录）。

### 数采起始姿态

用户约定的“回起始位”：关节 1～6 为 **`[90°, 0°, 0°, 0°, 0°, 0°]`**。
以现有机械零位为参考，关节 1 正方向是沿基座 +Z 轴从上向下看逆时针；
这里将用户所述“向左 90°”按此方向保存。机械零位仍为六关节全零。
配置位于 `piper_capture/config.py` 的 `robot.start_pose`，所有配置文件默认继承。
新 episode 的 `metadata.json → capture.configured_start_pose` 会记录这一目标，
它不是实测起始状态；实际姿态以 `samples.jsonl` 的关节反馈为准。
启动采集不会自动移动机械臂；回起始位是独立的运动操作。

## 3. 命令一览

| 命令 | 作用 |
| --- | --- |
| `doctor` | 环境检查：ROS、piper_sdk、RealSense 设备与目标流配置、CAN 接口状态、正运动学三方交叉校验 |
| `camera-probe [--open] [--seconds N]` | 列出 D435i 支持流配置；`--open` 时实际打开并实测帧率 |
| `capture [--scene S] [--episode E] [--duration T] [--camera-only] [--verbose]` | 旧版 PNG/JSONL 格式采集一个 episode |
| `capture-lerobot-interactive [--episodes N] [--return-on-q]` | 空格开始；按 q 结束并自动回到采集初始位姿 |
| `capture-lerobot [--duration T] [--arm-button-return-zero]` | 单条 LeRobot 采集；可用机械臂按钮结束并自动回零 |
| `robot go-zero --allow-motion [--wait-for-button]` | 显式驱动机械臂回到配置目标姿态，不改写电机零点 |
| `handeye sample --session S` | 手眼标定采样：人工摆姿态，回车记录一帧（不自动规划运动） |
| `handeye solve --session S [--method M] [--verify-session S2] [--redetect]` | 求解 + 留出验证，写入 `calibrations/handeye/` |
| `handeye verify --calibration-id ID --session S` | 用指定会话验证已有标定 |
| `handeye check --session S` | 只检查样本数量、姿态变化、异常样本 |
| `handeye list` | 列出标定结果与采样会话 |
| `gripper diagnose [--seconds N]` | 夹爪只读被动诊断（不发命令） |
| `gripper probe --allow-motion` | 主动探测（**会真实驱动夹爪**） |
| `gripper calibrate --calibration-id ID --point raw:mm ...` | 保存多点实测开度校准 |
| `gripper pending` | 输出未校准占位记录与所需测量步骤 |
| `video add --role R --file F [--scene S] [--episode E] [--reference-only]` | 登记/导入第三人称视频 |
| `video list` / `video verify [--check-content]` / `video probe --file F` | 列出 / 校验 / 探测视频 |
| `quality check --episode E` | 单个 episode 的数据质量检查（结论 pass / needs_attention / fail） |
| `quality dataset` | 整个数据集的质量检查 |
| `verify-fk [--urdf P]` | 正运动学三方交叉校验（本项目 DH / piper_sdk / 官方 URDF） |

退出码：`0` 成功；`1` 环境或运行失败；`2` 前置条件不满足（标定未通过验证、点数不足、文件不存在、未加 `--allow-motion` 等）。
`doctor` 在 CAN 接口健康但**总线上无节点发帧**时（见 9.5）会报 `fail` 并返回 `1`。

## 4. 采集语义（重要约定）

- **样本基准是 RGB-D 帧对**，目标 30 样本/秒；机械臂原生反馈以完整频率单独写入
  `robot_states.jsonl`，不参与降采样。
- **关节角用实测反馈**（`0.001°` 原始值换算为弧度），顺序固定为 `joint1..joint6`。
- **EE 位姿由反馈关节角做正运动学得到**（`ee_pose_source: "feedback_joint_fk"`），
  不用目标关节角，不用相机测量代替。`ee_pose = [x, y, z, qw, qx, qy, qz]`，xyz 单位米，
  四元数 wxyz、已归一化、已做 q/-q 符号连续性处理。
- **基站/末端坐标系**：`base_frame=piper_base_link`，`ee_frame=link6`（默认）。
  `tool_offset_m` 未实测时为全零，来源记为 `unconfigured_default_zero`；法兰/link6/TCP
  的区别见数据字典。
- **夹爪是独立字段**，不混入弧度数组：`gripper_feedback_raw`（驱动原始值，单位 0.001 mm）、
  `gripper_width_mm`（校准后两指实际间距）、`gripper_calibration_id`、`gripper_valid`。
  未完成物理校准时 `gripper_width_mm` 为 `null` 并写明原因，**不填虚构毫米值**。
- **时间同步是软件匹配，不是硬件同步**（数据集里显式写 `is_hardware_synchronized: false`）。
  超过容差的关节状态不会被静默复用：样本标 `invalid` 并计入统计。
- **PiPER 协议没有设备侧时间戳**，关节反馈只有主机接收时间
  （`joints_clock_source: "none:piper_can_protocol_has_no_device_timestamp"`）；
  RealSense 有设备时间戳（`global_time` 域），通过 `DeviceClockMapper` 映射到统一时间轴。
- **命令与状态分开**：本项目未接入下发命令记录，`commands.jsonl` 会显式写明
  `command_recording_enabled: false`，不从反馈状态反推命令。

## 5. 手眼标定流程（人工摆姿态）

相机装在末端（官方支架），采用 **eye-in-hand**，求 `T_ee_camera`：

```
p_ee = T_ee_camera @ p_camera        # p_camera 在 camera_color_optical_frame 下
```

**当前实物板（用户照片）是 calib.io 棋盘格：8 行×11 列方格、10 列×7 行内角点，标签格长 15 mm。**
使用专用配置 `configs/handeye_checkerboard_eye_in_hand.json`，启动：

```bash
bash scripts/handeye_checkerboard.sh he-checker-01
```

需要实时预览时加 `--preview`；续接已有会话加 `--resume`（棋盘位置、相机安装、
板参数和机械臂参考系必须与此前保持一致）：

```bash
bash scripts/handeye_checkerboard.sh he-checker-01 --resume --preview
```

打开 `http://127.0.0.1:8765`，预览与采样共用相机。页面有记录、删除上一帧、结束采样
并求解按钮。也可在本机终端记录一帧：

```bash
curl -X POST -H 'X-Handeye-Local: 1' http://127.0.0.1:8765/record
```

命令返回“已接收”仅代表操作已排队；是否保存成功、当前样本数以页面状态为准。

先将纸面平整贴在硬板上，再把板固定在桌面；整个采样过程板不能移动。
照片中的纸张有翘曲和折痕，需处理后再采。用尺测量连续 10 格应为 150 mm，
也核对另一方向的格长；如果打印缩放了，修改 `handeye.board.square_size_m` 为实测单格边长
（米），并将 `size_source` 改为 `measured`。当前配置记录 `printed_label_unmeasured`。
相机应看到完整棋盘和少量外侧白边，每次调整姿态后静止、回车采一帧，建议至少 16 帧，
绕不同轴改变角度；`q` 保存并求解。程序不会驱动机械臂。

棋盘格用 `findChessboardCornersSB` 检测全部 70 个内角点，再用 `solvePnP` 求位姿，
手眼求解流程保持一致。程序利用黑白格排列固定原点与坐标方向，防止相机旋转后
角点顺序翻转；目前支持一奇一偶的内角点数。清晰度、重投影误差和完整入镜检查
不能代替对纸面平整度与实际格长的现场检查。

以下 ArUco 参数和脚本仅用于官方教程里的单码板，与当前棋盘格配置分别保存。

### 公开官方名义安装值（可作临时参考）

本机还保存了一份来自 AgileX `piper_isaac_sim` 官方 `realsense_mid_stand` URDF、
并串接 Intel D435/D435i 官方光学坐标系的名义矩阵：
`configs/official_piper_realsense_mid_stand_nominal.json`。
它给出 `T_link6_camera_color_optical_frame`，平移约
`[-73.12, 3.49, 36.25] mm`。文件明确标为 `nominal_unverified` / `valid=false`，
因为它不是现场手眼采样，且假设打印支架、装配方向和相机机身完全符合官方模型。
安装到数据集目录：

```bash
python3 scripts/install_official_nominal_handeye.py
```

这份值可以让程序记录其来源和坐标变换，但质量报告仍会提示“手眼标定未验证”；
不能把它改写成 `valid=true` 来冒充实测结果。Z‑Robotics‑Lab 的
`piper_wrist_camera_calibration.example.json` 同样明确写着 `calibrated=false`，
且使用 `piper_gripper_base` / `d435_joint` 口径，与当前 `link6` 不同，因此不直接采用。

1. **实测标定板尺寸**并更新配置：`handeye.board.marker_size_m`（当前默认 `0.0677 m`
   取自官方示例，**必须用卡尺实测后复核**；配置里同时记录 `length_unit: "m"`）。
   默认板型 `aruco_single`，字典 `DICT_ARUCO_ORIGINAL`、id `582`。`charuco` 尚未实现，
   配错会直接抛 `ValueError` 而不是静默按单码处理。

   **量哪一段**：`marker_size_m` 是**黑色方框外边缘到对边外边缘**的边长，即
   `7 × 单模块宽`（`DICT_ARUCO_ORIGINAL` 为 5×5 数据 + 每边 1 模块黑框），
   **不含外圈白色留白**。已实测确认：渲染 id=582 码（420 px，模块 60 px）时黑像素
   铺满整张位图，`detectMarkers` 角点落在 `[40,40]→[459,40]`，与黑框外边缘重合。
   量法：横竖各一次取平均，另量对角线做交叉校验（应 = 边长 × 1.4142）。若误把
   1 模块白边量进去（9 个模块），尺度偏大 9/7 ≈ 1.29 倍，该比例误差会原样进入
   `T_ee_camera`。

   **字典必须与实物板一致**：同一个 id 在不同字典下图案完全不同（`DICT_ARUCO_ORIGINAL`
   是 1024 个 5x5 码，`DICT_4X4_1000` 是 1000 个 4x4 码）。已实测：用
   `DICT_ARUCO_ORIGINAL` 渲染 id=582 板，配 `DICT_ARUCO_ORIGINAL` 时检出且重投影
   0.00 px；配 `DICT_4X4_1000` 时**一个码都检测不到**（检测结果里会打印当前字典与
   期望 id 供核对）。

2. **确认相机固定连杆**：官方 URDF 将末端参考设为 `link6`；你已确认使用官方
   眼在手上配件，因此配置已填写 `camera.mount.link=link6`。这只确定父连杆，
   不能替代相机光学坐标系相对连杆的实测外参。
3. 采样（机械臂静止、标定板检测有效才记录）：
   ```bash
   python3 -m piper_capture.cli handeye sample --session he-001
   ```
   程序会给出质量提示（是否静止、检测是否有效、姿态是否有变化），回车记录一帧。
   不自动规划扫描运动，姿态由人工调整。
4. 求解与验证：
   ```bash
   python3 -m piper_capture.cli handeye solve --session he-001
   ```
   保存标定图像、关节反馈、正运动学位姿与检测结果，支持 `--redetect` 离线重新计算；
   多方法对比（TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS）；用未参与求解的留出样本输出
   可量化误差（位置 RMS/最大偏差 mm、旋转 RMS/最大测地偏差 deg）。

   `handeye.mode` 也支持官方流程中的 `eye_to_hand`。该模式按
   `agilexrobotics/handeye_calibration_ros` 的实现先把 `T_base_ee` 求逆为
   `T_ee_base`，输出 `T_base_camera`；默认仍为相机随末端移动的 `eye_in_hand`，输出
   `T_ee_camera`。两种模式都会要求末端姿态有足够旋转变化，并在独立留出姿态上验证。

   使用官方示例 ArUco 板时，可使用仓库内的配置和脚本：
   ```bash
   # 先把 Original ArUco id=582 标定板固定在桌面，并确认相机能看到完整码面
   ./scripts/handeye_official.sh he-piper-d435i-01
   ```
   脚本会先采样，再求解并写入 `dataset/calibrations/handeye/`。每次回车只记录一个
   已静止姿态；建议至少记录 16 个姿态，位置和绕 X/Y/Z 的旋转都要变化。输出的
   `transform.name` 是 `T_ee_camera`，父坐标系是 `link6`，子坐标系是
   `camera_color_optical_frame`。脚本不会用官方 URDF 名义值冒充实测外参。

### 5.1 官方参数与本项目的差异

官方资料核实结果（`Agilex-College` 的 `master` 分支，当前核对提交
`c2688be41e1bc99a9addd555f237a16bc839d936`；`Piper_ros` 的官方仓库只提供驱动、URDF
和启动文件，不包含手眼标定结果）：

| 项 | 官方 | 本项目 |
| --- | --- | --- |
| 标定流程文档 | `piper/handeye/README.md` | 同左（本项目按同样数学独立实现） |
| 现成标定数据 | **该目录下只有 README.md，没有任何标定数据/结果文件** | 需现场采样产生 |
| 标定板 | `marker_id:=582`、`marker_size:=0.0677`、`Original ArUco` 字典 | 配置默认值同左，边长待实测复核 |
| 求解 | `cv2.calibrateHandEye`（`handeye_calibration_ros`，默认 TSAI） | 同算法，另做多方法对比 + 留出验证 |
| 末端位姿来源 | 驱动上报的 `/end_pose` | **反馈关节角正运动学 FK**（驱动上报值另存 `driver_end_pose_raw`） |
| 采样 | 交互回车采样，仅存 JSON 结果 | 样本 + 图像 + 关节反馈全部落盘，可 `--redetect` 复算 |
| 验证 | 无验证环节 | 留出样本上 `T_base_target` 一致性，输出 mm/deg 误差 |

方向约定一致：官方 eye-in-hand 为 `T_base_cam = T_base_ee × T_ee_cam`，与本项目
`p_ee = T_ee_camera @ p_camera` 同一含义。

**检测环节的实测校验**：`detect_board` 用 `cv2.SOLVEPNP_ITERATIVE` 求解单码位姿
（与 OpenCV `cv2.aruco.estimatePoseSingleMarkers` 结果一致）。曾用 `SOLVEPNP_IPPE_SQUARE`，
实测在**板面正对相机**时退化为非精确解：重投影 RMS 27.15 px、位置误差 4.96 mm
（板距 0.17 m，约 3%），倾角 ≥0.01 rad 才恢复精确；ITERATIVE 在 0～0.4 rad 倾角下
重投影 RMS 均为 0.0000 px，故改用 ITERATIVE。

**端到端合成自检**（不接硬件，已知 `T_ee_camera` 真值反渲染标定板图像）：
24/24 检出，`handeye solve` 结果 `valid`，求解值相对真值平移误差 2.06 mm、
旋转误差 0.09°，留出验证 位置 RMS 0.73 mm / 旋转 RMS 0.37°。

**状态约定**：没有真实样本或未通过验证时 `status` 只会是 `pending`（样本不足/姿态变化不够）
或 `invalid`（留出误差超阈值），`transform` 不会出现，**不会用单位矩阵假装标定完成**。
标定结果记录 `parent_frame`、`child_frame`、平移单位、矩阵方向、相机使用 RGB 光学坐标系
（`camera_color_optical_frame`）、算法、依赖版本与采样时间。

## 6. 夹爪诊断与开度校准

```bash
# 只读诊断：不发送任何夹爪命令
python3 -m piper_capture.cli --dataset-root dataset gripper diagnose --seconds 10
```

诊断内容：原始值范围与量纲核对（配置 `raw_unit_mm=0.001`、标称行程 `[0, 70] mm`）、
`status_code` 位（bit6 使能 / bit5 驱动错误 / bit4 传感器异常 / bit7 回零）、
扭矩是否接近堵转、夹爪反馈更新率与间隔。**没有持续反馈时会明确报 blocker**，
并且不会把默认值 0 当作实测行程/使能状态。

开度校准（需要游标卡尺实物测量）：

```bash
python3 -m piper_capture.cli gripper calibrate --calibration-id grip-v1 \
  --point 0:25.3 --point 20000:38.1 --point 40000:51.0 --point 60000:64.2
```

输出线性拟合（`width_mm = slope * raw + intercept`）、残差 RMS/最大偏差/标准误、
单位自检（斜率与 0.001 的比值，用于发现把单侧行程当总行程，比值≈0.5）、覆盖范围警告。
少于 3 个点或原始值重复会直接报 `ValueError`。
未完成测量时用 `gripper pending` 输出占位记录与测量步骤（`valid: false`）。

## 7. 第三人称视频（只做导入与登记）

```bash
# 侧视任务视频：按 episode 关联
python3 -m piper_capture.cli video add --role side_task --episode ep-001 --file /path/side.mp4

# 环境环视视频：按 scene 保存，可被多个 episode 引用
python3 -m piper_capture.cli video add --role environment_overview --scene scene-tabletop --file /path/env.mp4

python3 -m piper_capture.cli video list
python3 -m piper_capture.cli video verify --check-content
```

记录文件路径、角色、分辨率、帧率、时长、`scene_id`、`episode_id`、sha256 校验值，
以及可获得的录制时间、时间偏移、同步状态。**没有同步依据时标 `unsynchronized`，
不伪造时间对齐**，也不要求与机械臂样本一一对应。未导入第三人称视频时，
机械臂与 D435i 仍可独立采集。

## 8. 数据目录与质检

目录结构与字段见 [docs/data_dictionary.md](docs/data_dictionary.md)。

```bash
python3 -m piper_capture.cli quality check --episode ep-001
python3 -m piper_capture.cli quality dataset
```

检查项包括：JSONL 可解析性、图像文件存在且尺寸一致、深度为 uint16、对齐深度与 RGB 同尺寸、
几何对齐声明、无效深度语义、实测帧率与丢帧、同步误差与过期反馈、标定文件存在性、
是否把软件匹配描述成硬件同步等。结论为 `pass` / `needs_attention` / `fail`，
报告写入 `<episode>/quality_report.json`。

数据集整体可搬移：所有路径都相对 `manifest.json` 所在目录记录。

## 9. CAN 链路注意事项（已定位过的真实故障）

### 9.1 曾出现的"总线完全静默"根因：**USB-CAN 适配器 USB 通路故障，不是机械臂**

2026-09-18 排查结论（`ip` + syslog + 被动监听）：

- 现象：`can0` 显示 UP / ERROR-ACTIVE / 错误计数全 0，但被动监听 0 帧，
  `piper_sdk` 读到 `time_stamp: 0`、关节全 0、`enable` 全 False。
- 根因：candleLight 适配器（`1d50:606f`）USB 数据通路已死 —— 发送返回
  `ENOBUFS: No buffer space available (105)`，syslog 记录 6 次 `USB disconnect`、
  5 次重枚举、6 次 `Error -71 while reading timestamp`、1 次
  `gs_usb: failed to set bittiming: -EPROTO`。
- **关键教训**：USB 通路死掉时内核不会再向适配器发起事务，`can0` 的
  `ERROR-ACTIVE` 与全 0 错误计数是**过期状态**，不能作为"链路正常"或"机械臂正常"的依据。
- 处理：把适配器换到主板直连的另一个 USB 口（`3-4` → `3-6`），并重新激活 `can0`。
  换口后总线恢复：`0x251`–`0x256`、`0x261`–`0x266`、`0x2A1`–`0x2A8` 全部正常，
  关节反馈 200 Hz、夹爪反馈 200 Hz。

该适配器为 **USB 总线供电（MaxPower 150 mA）**，对线材与 EMI 敏感。若再次出现
`-71` / `EPROTO` / `ENOBUFS` / 反复重枚举，优先查 USB 线材、接口与电机线缆干扰，
而不是怀疑机械臂。

### 9.2 每次重插适配器后必须重新激活

`can0` 是 NetworkManager 的 `unmanaged` 设备，**没有任何服务自动激活**：
重插 USB 后接口会以 DOWN/STOPPED 重建，必须人工激活（需 root，本项目不代为执行）：

```bash
sudo ip link set can0 up type can bitrate 1000000
# 或
bash /home/robot/mujoco/piper_ros/can_activate.sh can0 1000000
```

### 9.3 反馈帧率实测（`can0` 正常时）

| CAN ID | 内容 | 实测频率 |
| --- | --- | --- |
| `0x251`–`0x256` | 关节相关 | 200 Hz |
| `0x261`–`0x266` | 关节相关 | 40 Hz |
| `0x2A1`–`0x2A8` | 状态/末端位姿/关节反馈/夹爪反馈 | 200 Hz |
| 合计 | | ≈ 3040 帧/秒 |

### 9.4 夹爪"发指令后开合不稳定"的实测证据

只读诊断（`gripper diagnose`，全程未下发任何指令）实测：

- 夹爪原始反馈**恒为 `-3220`**（按 `0.001 mm` 换算约 −3.22 mm，为负值，超出
  `[0, 70] mm` 标称行程对应的 `[0, 70000]` 原始范围），整段窗口数值不变；
- `status_code` 恒为 **0**：所有错误位为 0，但 **bit6 使能位 = 0（驱动未使能）**、
  **bit7 回零位 = 0（未做过回零/set_zero）**。

即：**驱动器从未使能、也从未回零**。官方流程是先发 `status_code = 0x01` 使能再发
行程指令；未使能时行程指令不会正常执行，表现为"发了指令但开合不稳定/无响应"。
`gripper calibrate` 需要的多点实测开度仍然缺失，`gripper_width_mm` 保持为 `null`。

### 9.5 `doctor` 现在会探测总线流量（接口健康 ≠ 机械臂在发帧）

第 9.1 节的教训是：`can0` 显示 `UP` / `ERROR-ACTIVE` / 错误计数全 0，**并不能**说明
机械臂在通信。因此 `doctor` 的 CAN 检查已改为**只读比较 1.5 秒前后的 `rx_packets`**：

- 有增量 → `pass`，并给出 `≈N 帧/s`；
- 增量为 0 → `fail`，提示"接口本身正常，但总线上没有节点在发帧"，
  让 `doctor` 退出码为 `1`。此时应查机械臂供电与 CAN 接线，而不是反复激活接口。

2026-09-18 最近一次实测仍是**增量为 0**（`13406863 → 13406863`）：控制器零错误、
`cansend` 无 `ENOBUFS`、适配器仍绑在 `gs_usb`（`parentdev 3-7:1.0`、`1d50:606f`），
即适配器侧健康但总线上无节点发帧，与 9.1 的 USB 通路故障表现**不同**，
更像机械臂未上电或未接到总线。**该项仍未闭环，需人工检查硬件。**

## 10. LeRobotDataset v3（第二阶段）

双 D435i + PiPER 直接写官方 `LeRobotDataset` v3。环境用 `uv` 在用户目录装 Python
3.12（系统 Python 是 3.10，**不要**用 `python3.12 -m venv`，本机没有 `python3.12`）：

```bash
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv-lerobot
uv pip install --python .venv-lerobot/bin/python -r requirements-lerobot.txt
. .venv-lerobot/bin/activate

python -m piper_capture.cli --config configs/lerobot_v3_two_d435i.json \
  capture-lerobot --duration 5
python -m piper_capture.cli verify-lerobot \
  --root dataset/lerobot_v3 --repo-id piper_two_d435i
```

输出目录由官方写入器管理（`meta/`、`data/`、`videos/` 与 episode 索引）。
已有兼容数据集通过 `resume()` 追加下一条；新目录通过 `create()` 创建。
日常多场景采集使用本页开头的 `capture-lerobot-interactive`，下面的 5 秒命令仅作单条示例。
采集器不会用旧帧或黑帧填补缺帧；实时编码队列溢出会清理未完成 episode 并返回失败。

### 10.1 视频编码器必须写死，不能用 `auto`（已实测的坑）

**`rgb_encoder.vcodec` 必须显式写 `h264`（即 libx264），不要用 `"auto"`。**

LeRobot 的 `auto` 只调用 `detect_available_encoders_pyav()` 检查编码器**是否被编译进
PyAV**，不检查它**能否真正打开**。本机 PyAV 15.1.0 列出了 `h264_nvenc`/`hevc_nvenc`，
于是 `auto` 选中 `h264_nvenc`，随后在编码线程里失败：

```
RuntimeError: Encoder thread for observation.images.wrist failed:
  [Errno 22] Invalid argument: 'avcodec_open2(h264_nvenc)'
```

实测环境（PyAV 15.1.0 捆绑 libavcodec 61.19.101、NVIDIA 驱动 580.65.06 / CUDA 13.0、
RTX 5080、`libnvidia-encode.so.580.65.06` 存在）：**nvenc 无论沙箱内外、无论
640×480 还是 1280×720，`avcodec_open2` 一律失败**（`EINVAL` 或 `AVERROR_EXTERNAL`），
即该 PyAV 捆绑的 nvenc 在本机不可用。系统静态 ffmpeg 7.0.2 也没有编译 nvenc。
因此两台配置都改成显式软件编码器：

```json
"rgb_encoder": {"vcodec": "h264", "crf": 18, "g": 2, "preset": "veryfast"}
```

深度仍用 `hevc` + `pix_fmt: gray12le` + `x265-params: lossless=1`（软件 libx265）。

### 10.2 软件编码吞吐实测（4 路并发，预生成帧后纯计时编码）

| 分辨率 | 流 | 实测 | 需要 | 余量 |
| --- | --- | --- | --- | --- |
| 1280×720 | RGB h264 veryfast ×2 | 126.6 / 125.9 fps | 30 | ≈4.2× |
| 1280×720 | depth hevc lossless ×2 | 143.7 / 142.0 fps | 30 | ≈4.8× |
| 640×480 | RGB h264 veryfast ×2 | 57.6 / 62.0 fps | 30 | ≈2× |
| 640×480 | depth hevc lossless ×2 | 199.2 / 202.2 fps | 30 | ≈6.6× |

CPU 为 Ryzen 9 9950X（32 线程），软件编码有充足余量，不需要 nvenc。

### 10.3 已验证 / 未验证

**已验证**（合成帧走官方写入器全链路，不代表任何实测机械臂/相机数据）：用配置里的
编码器设置 `LeRobotDataset.create` → `add_frame` ×30 → `save_episode` → `finalize`，
无编码队列丢弃；再用官方加载器重载，`frames=30`、四路 `video_keys`/`depth_keys` 齐全、
RGB 解码 `(3,720,1280)`、深度 `(1,720,1280)`，`observation.state` 与
`observation.ee_pose` 均为 `(7,)`。

**现场进展（2026-09-19）**：已生成真机 episode，曾检查到旧目录 `dataset/lerobot_v3`
包含 1 条、196 帧；另一次试采生成 1 条、677 帧，已按用户要求删除。
这些记录不代表 150 条采集已完成，也不代表当前硬件始终在线。
此前 CAN 静默和相机 USB 异常见故障记录；每次采集仍需检查实时状态。
交互采集已接入官方 `resume()`；已有检查覆盖续开数据集及计数保留，
完整 150 条连续真机采集尚未验证。

另：`torchcodec` 在本机加载失败（缺 `libavdevice.so.58`），LeRobot 会告警并自动回退
到 `pyav`，不影响编码与解码，可忽略。

### 10.4 深度量化是有损的，`depth_min` 必须写 0.0

**结论先说**：LeRobot 的深度是 **12-bit 对数量化**，这一步本身**有损**；
`x265-params: lossless=1` 只保证 HEVC 编解码对那 12-bit 码值无损，**不能**把
"深度→码值"的量化误差变回 0。而且 LeRobot **没有为无效值保留码位**，
所以 `depth_min` 必须写 `0.0`，否则传感器"无测量"会被伪造成一个真实距离。

**1) `depth_min` 的语义**：源码 `.venv-lerobot/.../lerobot/datasets/depth_utils.py`
的 `quantize_depth()` 里，`depth_min` 是 **quantum 0 对应的深度**：

```python
norm = (np.log(depth_f + shift_u) - log_min) / (log_max - log_min)   # use_log=True
quantized = np.rint(norm * DEPTH_QMAX).clip(0, DEPTH_QMAX).astype(np.uint16)
```

没有任何分支把"无效值"映射到专用码位；D435i 的 `raw=0`（该像素无测量）会顺着公式
落到 `quantum 0`，解码回来就是 `depth_min`。

**2) 实测对比**（`/tmp/lr_depth_quant.py`，往返 `quantize → dequantize`）：

| `depth_min` | 输入 `raw=0`（无测量） | 读回 | 语义 |
| --- | --- | --- | --- |
| `0.01`（LeRobot 默认） | code 0 | **0.010000 m** | 把"无测量"伪造成 10 mm |
| `0.0`（本项目） | code 0 | **0.000000 m** | 保住 `0 = 无效` |

因此两个配置都写 `"depth_min": 0.0`（`depth_encoder_note` 同步记录了理由）。

**3) 量化误差实测**（`depth_min=0.0` / `depth_max=10.0` / `shift=3.5` / `use_log=true`，
走真实数据集写入 + 重载，含 HEVC lossless 编解码，`/tmp/lr_depth_e2e.py`）：

| 真实深度 | 读回 | 往返误差 |
| --- | --- | --- |
| 0.000 m（无测量） | 0.000000 m | 0（语义保住） |
| 0.05 m | 0.049966 m | −0.03 mm |
| 0.10 m | 0.099458 m | −0.54 mm |
| 0.50 m | 0.499911 m | −0.09 mm |
| 1.00 m | 0.999462 m | −0.54 mm |
| 1.50 m | 1.500044 m | +0.04 mm |
| 2.00 m | 1.999826 m | −0.17 mm |
| 3.00 m | 3.000313 m | +0.31 mm |
| 5.00 m | 5.001029 m | +1.03 mm |
| 10.00 m | 10.000000 m | 0 |
| 12.00 m（超 `depth_max`） | 10.000000 m | 被截断到 `depth_max` |

误差量与理论一致：12-bit 对数刻度的码距是
`ln(depth_max+shift) - ln(depth_min+shift)` 均分 4095 份，
在深度 `d` 处的步长 ≈ `步长_log × (d + shift)`，故**近处细、远处粗**：
0.5 m 处约 1.0 mm/码、1 m 处约 1.5 mm/码、5 m 处约 2.8 mm/码
（上表误差均在半码内）。工作距离内误差 ≤1 mm，满足采集需求；
但**这不是无损**，不能对外宣称深度无损。超 `depth_max` 的值会被 `clip` 到 10 m，
D435i 本身在 10 m 外基本无有效测量，影响可忽略。
需要完全无损时只能不用视频编码器（本项目第一阶段 PNG `uint16` 路径是逐像素无损的，
见第 4 节与数据字典）。

## 11. 双相机调参界面

降低深度噪点、调节 D435i 红外曝光 / 增益 / 投射器功率的操作顺序见
[深度与红外调参指南](docs/data_collection_guide.md)。界面默认打开深度 / 红外参数，
提供左右红外预览、原始 / 对齐深度切换及红外暗部和饱和比例。

```bash
cd /home/robot/shucai1
python3 -m piper_capture.cli --config configs/lerobot_v3_two_d435i.json camera-ui
```

浏览器打开 **http://127.0.0.1:8766**。使用已有的 `numpy`、`opencv`、
`pyrealsense2`，网页服务使用 Python 标准库，不需要 Tk、Qt 或 ROS。
若未自动打开浏览器，手动访问地址即可；`--no-browser` 可关闭自动打开，`--port` 可换端口。

1. 确认腕部和第三人称对应的序列号，选择各自 RGB / Depth 分辨率，点击
   **连接 / 应用分辨率**。双相机采集保持 30 fps，仅展示设备支持的 BGR8 / Z16 规格。
2. 在各相机的 **RGB 参数 / 深度参数** 页签调整曝光、增益、白平衡、亮度、
   对比度、饱和度、锐度、伽马、防闪烁、深度预设、红外发射器和激光功率等。
   仅显示设备固件支持的图像参数，并使用设备报告的范围/步长。修改即时应用；
   手动曝光/增益先关闭自动曝光，手动白平衡先关闭自动白平衡。
3. 每台同时显示 RGB、左右红外与深度；深度默认原始视角，可切换到对齐 RGB。
   统计包含接收帧率、原始 / 对齐深度有效比例和对齐中心像素距离。
   深度预览固定映射 0–3 m，黑色表示无效值；这只是显示范围，不截断采集深度。
4. 点击 **保存参数**，默认保存到源文件同目录的 `*_tuned.json`，重复保存覆盖此文件。
   可用 `camera-ui --output configs/my_cameras.json` 指定保存位置。
   原文件里的机械臂、同步、编码等设置会保留；参数按相机序列号和 RGB/Depth 分开存放，
   自动模式下不保存当时的曝光/白平衡测量值作为手动设定值。
5. 点击 **停止预览 · 释放相机**，再执行页面给出的采集命令，例如：

```bash
.venv-lerobot/bin/python -m piper_capture.cli \
  --config configs/lerobot_v3_two_d435i_tuned.json capture-lerobot --duration 5
```

采集启动会重新应用 `camera.wrist.sensor_options` 和
`camera.third_person.sensor_options`；不支持或应用失败会明确报错，不静默忽略。
实际传感器参数也会写进采集元数据。RGB 分辨率可两台不同，各自深度几何对齐到各自 RGB。
再次调试已保存的配置时，把 `--config` 和 `--output` 都指定为该文件即可继续保存到同一处。

预览期间相机会被占用，请先结束其他采集/相机预览程序。关闭浏览器页签不会自动释放相机，
用页面停止按钮或服务终端 `Ctrl-C` 释放。界面只绑定本机 `127.0.0.1`。
