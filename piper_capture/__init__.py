"""PiPER + D435i 数据采集、手眼标定与数据管理。

模块划分：
  schema     数据集 schema 版本、关节顺序、四元数/变换约定
  jsonio     JSON/JSONL 读写（原子写、逐行 flush）
  config     运行配置
  kinematics 正运动学（反馈关节角 -> EE 位姿）、FK 交叉校验
  robot      只读机械臂反馈采集
  motion     显式开启的 PiPER 回零运动与模式触发
  camera     D435i RGB-D 采集
  clock      时钟域与时间匹配
  episode    episode 采集写入与断点保留
  handeye    手眼标定采样/求解/验证
  gripper    夹爪诊断与开度标定
  external   第三人称视频登记
  quality    数据质量检查
"""

SCHEMA_VERSION = "1.0.0"

__all__ = ["SCHEMA_VERSION"]
