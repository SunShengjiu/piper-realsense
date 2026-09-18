# 数据字典（schema_version 1.0.0）

本文件描述数据集目录结构、文件字段、单位与坐标约定。字段名与
`piper_capture` 实际写出的内容一致，可用 `examples/read_dataset.py` 直接读取验证。

## 1. 目录结构

```
dataset/
  manifest.json                     数据集总索引、单位声明、scene/episode 列表
  external_registry.json            第三人称视频登记表（全局唯一）
  reports/
    doctor.json                     环境检查报告
    gripper/<utc>-diagnosis.json    夹爪只读诊断报告
    gripper/<utc>-probe.json        夹爪主动探测报告（含真实下发命令记录）
  calibrations/
    camera/<camera_calibration_id>.json
    handeye/<handeye_calibration_id>.json
    handeye/sessions/<session_id>/{session.json,samples/sample_%04d.json,images/color_%04d.png}
    gripper/<gripper_calibration_id>.json
  scenes/<scene_id>/
    scene.json
    external/environment_overview.mp4
  episodes/<episode_id>/
    metadata.json
    samples.jsonl                   RGB-D 帧对样本（采样基准）
    robot_states.jsonl              机械臂原生反馈状态日志（不降采样）
    commands.jsonl                  下发命令记录（本项目未接入，见 §7）
    rgb/000001.png ...
    depth_raw/000001.png ...
    depth_aligned/000001.png ...
    external/side_task.mp4
    logs/session.log
    quality_report.json             quality check 生成的报告
```

**路径约定**：所有文件路径都是相对路径，整个数据集可整体搬移，不需要改写任何文件。
但**基准目录有两类，读的时候必须区分**：

| 字段位置 | 基准目录 | 例子 |
| --- | --- | --- |
| `samples.jsonl` 的 `cameras.*.path` | **episode 目录** | `rgb/000001.png` → `episodes/<episode_id>/rgb/000001.png` |
| `metadata.json` 的 `camera.camera_calibration_path`、`camera.scene_json` | 数据集根目录 | `calibrations/camera/<id>.json` |
| `external_videos[]` / `environment_overview[]` 的 `stored_path_relative_to_dataset` | 数据集根目录 | `episodes/<episode_id>/external/side_task.mp4` |

`samples.jsonl` 的 `cameras` 块内用 `path_base: "episode_dir"` 显式声明了这一约定。
绝对路径（用 `--reference-only` 登记的外部视频源）不会被复制，搬移后需重新登记。

**跨文件关联键**：`scene_id` → `episode_id` → `sample_id`；
`camera_calibration_id` / `handeye_calibration_id` / `gripper_calibration_id` 指向
`calibrations/` 下的独立文件，样本只引用 id，不重复写矩阵。

## 2. 单位与坐标约定（全局）

| 量 | 约定 |
| --- | --- |
| 关节角 `joint_positions_rad` | 弧度（rad），顺序**固定** `joint1..joint6` |
| 驱动原始关节值 | `0.001°` 整数（`piper_sdk` 报文原始值），字段名带单位后缀 |
| `ee_pose` | `[x, y, z, qw, qx, qy, qz]`；位置单位 m；四元数 **wxyz**、已归一化 |
| 四元数连续性 | 相邻样本做 q/-q 符号连续性处理（`QuatContinuityTracker`） |
| 夹爪原始反馈 `gripper_feedback_raw` | 驱动原始整数，单位 `0.001 mm` |
| 夹爪开度 `gripper_width_mm` | 毫米，**两指外侧间距（总开度）**，需物理校准；未校准为 `null` |
| 深度 `depth_raw` / `depth_aligned` | `uint16` 无损 PNG；`depth_m = raw_depth_uint16 * depth_scale_m`；`raw == 0` 表示**无有效测量**（不是"距离 0"） |
| 时间戳 | `*_host_recv_ns` 为 `time.time_ns()`（主机时钟）；`source_timestamp_ms` 为设备时间戳（毫秒，域见 `source_clock_domain`） |

坐标变换方向：`T_A_B` 表示 `p_A = T_A_B @ p_B`。

| 坐标系 | 说明 |
| --- | --- |
| `piper_base_link` | 机械臂基座系（`robot.base_frame`） |
| `link6` | 第六连杆；**默认 EE 坐标系**（`ee_pose_frame`） |
| 法兰 / TCP | 法兰与 link6 一般重合；夹爪 TCP = link6 再叠加 `tool_offset_m`。当前 `tool_offset_m = [0,0,0]`，来源 `unconfigured_default_zero`，**区别必须显式转换，不能混用** |
| `camera_color_optical_frame` | RGB 光学系（+x 右、+y 下、+z 前），相机内参/标定均基于该系 |
| `camera_depth_optical_frame` | 深度光学系；`aligned_depth` 已几何对齐到 **color** 光学系 |

## 3. `manifest.json`

| 字段 | 说明 |
| --- | --- |
| `schema_version` | `"1.0.0"` |
| `dataset_id` | 数据集标识 |
| `created_at` / `updated_at` | UTC ISO8601 |
| `joint_names` | `["joint1", ..., "joint6"]`，固定顺序声明 |
| `units` | 全局单位声明（见 §2） |
| `ee_pose_layout` | `"[x, y, z, qw, qx, qy, qz]"` |
| `path_convention` | 路径相对本文件所在目录，可整体搬移 |
| `scenes[]` | `{scene_id, camera_calibration_id, handeye_calibration_id, created_at, updated_at, ...}` |
| `episodes[]` | `{episode_id, scene_id, status, started_at, ended_at, samples, invalid_samples, robot_states}` |
| `calibrations` | `{camera:[id...], handeye:[id...], gripper:[id...]}` 索引 |

## 4. `scenes/<scene_id>/scene.json`

| 字段 | 说明 |
| --- | --- |
| `scene_id` | 场景 id |
| `camera_calibration_id` / `handeye_calibration_id` / `gripper_calibration_id` | 该场景使用的标定版本（可为 `null`） |
| `camera_mount` | `{link, source, note, official_reference}`；`link` 未现场确认时为 `null`，`source=unconfigured`。`official_reference` 是官方 URDF 的**参考值**，`usage_policy` 明确它不是实测标定结果 |
| `environment_overview[]` | 环视视频简要记录（见 §8），按 scene 保存、可被多个 episode 引用 |
| `robots` | `["piper"]` |

## 5. `episodes/<episode_id>/metadata.json`

| 字段 | 说明 |
| --- | --- |
| `schema_version` / `episode_id` / `scene_id` | 标识 |
| `status` | `open`（写入中，含被 kill 的情况）/ `closed`（正常结束）/ `aborted`（Ctrl-C） |
| `started_at` / `ended_at` / `duration_s` | UTC 时间与墙钟时长（含开关相机、预热） |
| `capture.configured_start_pose` | 配置的起始姿态目标（关节顺序、角度单位 deg、参考零位、方向约定及来源）；当前为 `[90,0,0,0,0,0]`，不代表机械臂实际已到达，也不触发自动运动 |
| `sample_span_s` | 首末样本主机接收时间跨度；**实测样本率应以它为准** |
| `measured_sample_rate_hz` | `(samples-1) / sample_span_s` |
| `host` | `{hostname, user, platform, python}` |
| `joint_names` / `units` / `ee_pose_layout` | 与 manifest 一致的自描述声明 |
| `robot` | 驱动记录：`backend`、`can_interface`、`dh_is_offset`、`ee_frame`、`base_frame`、`tool_offset_m`、`kinematics`（模型/版本/DH 表）、`sdk_versions`、`command_interface` 等；camera_only 时为 `{enabled:false, reason:...}` |
| `camera` | 相机标定快照（同 §9 camera 文件）+ `camera_calibration_path`、`scene_json` 相对路径 |
| `sync` | 同步配置：`tolerance_ms`、`robot_tolerance_ms`、`robot_match_mode`、`max_robot_state_age_ms`、`allow_stale_fill`、`clock_mapping` |
| `capture` | `target_sample_rate`、`camera_only`、`motion_commands_sent: false` |
| `calibration_ids` | 三个 `*_calibration_id`（可为 `null`） |
| `files` | 各文件/子目录相对名 |
| `counters` | `{samples, invalid_samples, robot_states}` |
| `diagnostics_summary` | 相机诊断（实测帧率、丢帧、无效深度像素、落盘线程统计）、机械臂诊断、同步计数器、`loop_error` |
| `external_videos[]` | 该 episode 关联的第三人称视频简要记录（见 §8） |

## 6. `samples.jsonl`（每行一个 JSON 对象）

采样基准是 **RGB-D 帧对**，目标 30 样本/秒。

### 6.1 标识与有效性

| 字段 | 说明 |
| --- | --- |
| `schema_version` | `"1.0.0"` |
| `episode_id` / `sample_id` / `seq` | `sample_id` 形如 `<episode_id>-000001` |
| `valid` | 综合有效性；任一必需数据缺失或超同步容差即为 `false` |
| `invalid_reasons[]` | 无效原因字符串列表（见 §11） |

### 6.2 `timestamps`

| 字段 | 说明 |
| --- | --- |
| `rgb.source_timestamp_ms` | RealSense 设备时间戳（毫秒） |
| `rgb.source_clock_domain` | 实测为 `global_time`（`rs.timestamp_domain`） |
| `rgb.host_recv_ns` | 主机接收该帧的时刻（`time.time_ns()`） |
| `rgb.device_clock_mapped_host_ns` | 设备时间戳映射到主机时间轴后的值（`DeviceClockMapper`） |
| `rgb.device_vs_host_dt_ms` | 映射值与主机接收时刻之差 |
| `rgb.clock_source` | `device:realsense_frame_timestamp` |
| `depth.*` | 同上（深度帧自己的设备时间戳与主机接收时间） |
| `aligned_depth.source_timestamp_ms` | 继承**原始深度**时间戳（同一帧集几何对齐生成） |
| `robot_joints.source_timestamp_ns` | 恒为 `null`：PiPER 协议无设备侧时间戳 |
| `robot_joints.source_clock_domain` | `"unavailable"` |
| `robot_joints.clock_source` | `host_receive_only:piper_can_protocol_has_no_device_timestamp` |
| `reference_host_ns` | 该样本的统一基准时间（帧集主机接收时刻） |
| `reference_clock` | `host_receive_time_frameset(time.time_ns)` |

### 6.3 `sync`

| 字段 | 说明 |
| --- | --- |
| `matching_method` | 关节匹配方式，如 `nearest` / `linear` / `nearest_no_robot_state` |
| `rgb_depth_device_dt_ms` | RGB 与 Depth **设备时间戳**之差 |
| `rgb_depth_tolerance_ms` | 容差（配置 `capture.sync.tolerance_ms`） |
| `robot_joints_dt_ms` | 所选关节状态与基准时间之差 |
| `gripper_dt_ms` | 夹爪反馈与基准时间之差 |
| `is_hardware_synchronized` | 恒为 `false`：**软件时间匹配，不是硬件同步** |
| `note` | 文字声明 |

### 6.4 机械臂状态（由 `RobotStateMatcher` 注入）

| 字段 | 说明 |
| --- | --- |
| `robot_joints_match_method` | 实际使用的匹配方式 |
| `robot_joints_dt_ms` | 关节状态与样本基准的时间差 |
| `robot_state_ids_used` | 参与该样本的原始状态 id（插值模式为两个） |
| `joint_names` | 固定顺序 |
| `joint_positions_rad` | 六轴**实测反馈**角（弧度）；无状态时为 `null` |
| `ee_pose` | `[x,y,z,qw,qx,qy,qz]`，由反馈关节角 FK 得到；无状态时为 `null` |
| `ee_pose_frame` | 默认 `link6` |
| `ee_pose_source` | `feedback_joint_fk`（**不是**目标关节角，也不是相机测量） |
| `ee_pose_units` | `{position:"m", quaternion:"wxyz"}` |
| `gripper_feedback_raw` | 驱动原始反馈（`0.001 mm`） |
| `gripper_width_mm` | 经校准的两指间距；未校准或超有效范围为 `null` |
| `gripper_valid` | 只看 `status_code` bit4 传感器异常 / bit5 驱动错误 |
| `gripper_calibration_id` | 使用的夹爪校准版本（可为 `null`） |
| `gripper_dt_ms` / `gripper_source_state_id` | 夹爪反馈时间差与来源状态 id |
| `robot_state_stale` | 是否使用了过期状态（超 `max_robot_state_age_ms`） |

### 6.5 `cameras`

| 字段 | 说明 |
| --- | --- |
| `path_base` | 固定 `"episode_dir"`：`cameras.*.path` 相对 **episode 目录**解析 |
| `path_base_note` | 文字说明（与 metadata 中相对根目录的路径区分） |
| `camera_frame` / `depth_frame` / `aligned_depth_frame` | 光学系声明 |
| `color.{path, frame_number, width, height, encoding, format}` | `rgb/000001.png`，`bgr8`，无损 PNG |
| `depth_raw.{path, frame_number, width, height, dtype, format, depth_scale_m, depth_formula, invalid_value, invalid_pixels}` | `uint16` 无损 PNG；`invalid_value = 0`，`depth_formula = "depth_m = raw_depth * depth_scale_m"` |
| `depth_aligned.{..., alignment, same_size_as_rgb}` | `alignment = "realsense_rs_align_to_color (geometry based)"`，与 RGB 同尺寸 |
| `is_geometric_alignment` | `true`：对齐由 RealSense 几何对齐生成，**不是**普通缩放 |

三个路径互相独立保存：原始深度**不覆盖**、对齐深度另存；彩色深度图（展示用）不作为数据保存。

### 6.6 `calibrations`

`camera_calibration_id`、`handeye_calibration_id`、`gripper_calibration_id`（可为 `null`）。

## 7. `robot_states.jsonl`

机械臂**原生频率**反馈日志（不降采样），用于事后复核反馈频率与掉帧。
每行字段来自 `RobotState.to_dict()`：

| 字段 | 说明 |
| --- | --- |
| `robot_state_id` | 递增 id |
| `joint_names` / `joint_positions_rad` | 六轴反馈（弧度） |
| `joint_positions_raw_0p001deg` | 驱动原始整数（`0.001°`） |
| `joints_host_recv_ns` | 主机接收时刻 |
| `joints_source_timestamp_ns` | `null`（协议无设备时间戳） |
| `joints_clock_source` | 同上 |
| `gripper_feedback_raw` / `gripper_feedback_raw_unit` | 原始反馈与单位 `0.001 mm` |
| `gripper_effort_raw` / `gripper_effort_raw_unit` | 扭矩原始值与单位 `0.001 N*m` |
| `gripper_status_code` / `gripper_status_bits` | 状态码与逐位解码（bit4 传感器异常、bit5 驱动错误、bit6 使能、bit7 回零） |
| `gripper_host_recv_ns` | 夹爪反馈主机接收时刻 |
| `gripper_valid` / `gripper_width_mm` / `gripper_calibration_id` | 同样本 |
| `ee_pose` / `ee_pose_frame` / `ee_pose_source` | FK 结果与来源 |
| `driver_end_pose_raw` | **驱动自带**末端反馈，独立字段，注明来源；与 FK 结果分开保存 |
| `arm_status` / `enable` / `feedback_hz` | 状态字、使能数组、反馈频率 |

## 8. 第三人称视频

`external_registry.json`：`{schema_version, created_at, updated_at, roles:[side_task, environment_overview], videos:[...]}`。

单条视频记录：

| 字段 | 说明 |
| --- | --- |
| `role` | `side_task`（侧视任务，按 episode 关联）或 `environment_overview`（环视，按 scene 保存，可被多个 episode 引用） |
| `path_mode` | `copied_into_dataset` 或 `referenced_only`（`--reference-only`） |
| `source_path` / `stored_path` / `stored_path_relative_to_dataset` | 原始与数据集内路径 |
| `sha256` | 文件校验值（导入时的副本校验） |
| `scene_id` / `episode_id` | 关联键（另一个为 `null`） |
| `video` | `{file_name, file_size_bytes, probe_tool, width, height, fps, frame_count, duration_s, fourcc, codec}` |
| `recorded_at` / `time_offset_s` / `offset_basis` | **可获得的**录制时间、时间偏移及其依据（未提供则为 `null`） |
| `synchronization_status` | 默认 `unsynchronized`；没有同步依据时不得填其它值 |
| `operator` / `notes` | 操作者与备注 |

episode `metadata.json` 的 `external_videos[]` 与 scene.json 的 `environment_overview[]`
存**简要记录**（`role`、路径、`sha256`、`duration_s`、`synchronization_status`、`scene_id`/`episode_id`），
完整记录在登记表中。侧视视频不要求与每条机械臂样本一一对应。

## 9. `calibrations/`

### 9.1 `camera/<id>.json`

`device{name,serial,firmware_version,sdk}`、`color_intrinsics{width,height,fx,fy,ppx,ppy,model,coeffs}`、
`depth_intrinsics{...}`、`depth_to_color_extrinsics{rotation_row_major, translation_m, note}`
（`note` 明确是 RealSense 原始外参，未做手眼标定替换）、`color_to_depth_extrinsics`、
`depth_scale_m`、`depth_formula`、`actual_streams` / `requested_streams`、
`spec_match`、`spec_mismatch_reason`、`optical_frames`、`content_sha256`。

### 9.2 `handeye/<id>.json`

| 字段 | 说明 |
| --- | --- |
| `status` / `valid` | `valid` / `pending` / `invalid`；未通过验证时不会把结果标为可用，也不会用单位矩阵顶替 |
| `status_reason` | 未通过时的原因 |
| `mode` | `eye_in_hand` / `eye_to_hand` |
| `algorithm` | `{solver:"cv2.calibrateHandEye", method_used, methods_compared, opencv_version, numpy_version}` |
| `transform` | `eye_in_hand` 时为 `T_ee_camera`（`p_ee = T_ee_camera @ p_camera`），`eye_to_hand` 时为 `T_base_camera`（`p_base = T_base_camera @ p_camera`）；另含 `parent_frame`、`child_frame`、`matrix_row_major`、`translation_m`、`translation_unit`、`matrix_direction`、`camera_optical_frame_convention`、`camera_stream_used`、`frame_conversion{solved_frame,dataset_ee_frame,converted,note}` |
| `board` | 公共字段 `{type, length_unit, source_of_size}`；`aruco_single` 使用 `{marker_id, dictionary, marker_size_m}`；`checkerboard` 使用 `{inner_corners:[列数,行数], square_size_m}`。当前实物棋盘为 `[10,7]` 内角点，标签格长 `0.015 m`，来源 `printed_label_unmeasured` |
| `camera_calibration_id` | 使用的相机内参版本 |
| `session_id` / `verify_session_id` | 样本来源会话 |
| `sample_checks` | 样本总数/可用数/最少要求、被剔除样本、EE 平移与旋转跨度、目标距离范围、`ready_to_solve` |
| `samples_used` | `train_count` / `holdout_count` / 具体索引 / `holdout_source` |
| `train_consistency` / `verification` | 位置 RMS/最大偏差（mm）、旋转 RMS/最大测地偏差（deg）、逐样本误差、`T_base_target_mean` |
| `thresholds` | 判定阈值（`verify_max_position_rms_mm`、`verify_max_rotation_rms_deg`） |
| `sampling_times` | 会话起止时间 |
| `dependencies` | 算法与依赖版本，注明参考实现来源 |
| `content_sha256` | 自校验 |

验证原理：眼在手上时标定板在基座系静止，检查
`T_base_target = T_base_ee @ T_ee_camera @ T_cam_target`；眼在手外时标定板固定在
末端，按官方实现使用 `T_ee_base = inv(T_base_ee)`，检查
`T_ee_target = T_ee_base @ T_base_camera @ T_cam_target`。留出样本上该量的离散程度
即为可量化误差（不依赖标定板真值）。

会话目录 `handeye/sessions/<session_id>/`：`session.json`（配置/时间/`motion_commands_sent: false`）、
`samples/sample_%04d.json`（关节反馈、`T_base_ee`、检测结果、时间戳）、`images/color_%04d.png`
（离线可 `--redetect` 重新检测）。

### 9.3 `gripper/<id>.json`

`valid` / `calibrated` / `status_reason`、`method: manual_physical_measurement_linear_fit`、
`unit_of_measured_width`、`model: width_mm = slope * raw + intercept`、`slope_mm_per_raw_unit`、
`intercept_mm`、`residual_rms_mm` / `residual_max_abs_mm` / `residual_std_error_mm`、
`n_points`、`measured_span_mm`、`measured_raw_range`、`valid_raw_range`、`driver_range_mm`、
`unit_check{expected_slope..., fitted_slope, slope_ratio_vs_expected, note}`、
`coverage_warning`、`points[{raw, measured_width_mm}]`、`operator` / `notes` / `measured_at`。

## 10. `quality_report.json`

| 字段 | 说明 |
| --- | --- |
| `overall` | `pass` / `needs_attention` / `fail` |
| `episode_status` | metadata 中的 status |
| `counters` | 样本/无效样本/状态数 |
| `checks[]` | `{name, status, detail, ...}`，`status ∈ {pass, needs_attention, fail, unknown}` |
| `statistics` | 实测帧率、丢帧、同步误差、过期反馈、无效深度像素等统计 |
| `interpretation` | 对结论的文字解释 |

## 11. `invalid_reasons` 取值

| 原因 | 触发条件 |
| --- | --- |
| `缺少机械臂关节状态` | 该样本时刻没有可用关节反馈 |
| `缺少 EE 位姿` | 无关节状态，无法计算 FK |
| `采集时刻没有任何机械臂反馈状态` | 采样窗口内完全没有状态 |
| `关节状态超出同步容差` / `关节状态过期` | 时间差超 `tolerance_ms` / 超 `max_robot_state_age_ms` |
| `RGB 与 Depth 设备时间戳超出容差` | 帧对内部超容差 |

`--camera-only` 采集时全部样本会标 `缺少机械臂关节状态`，这是**如实的**：
没有机械臂数据时不伪造关节角，也不伪造 `ee_pose`。
