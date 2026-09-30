# 30 Hz：初始姿态到首次成功按键的数据集

> 文中的数据集、权重、图像和运行报告属于本地产物，不随 Git 仓库分发；新机器请先按 [README](../README.md) 生成场景。历史结果不代表本次安装已经完成验证。

训练 episode 从机械臂的折叠初始姿态开始，在目标按键的橙色边框**第一次亮起的采样帧**结束，包含这张成功画面。之后的继续按压、停留、撤回、回弹和归位均不属于训练 episode。归位仍用于准备下一次操作，完整原始记录保留作物理验证和故障排查。

本批修正沿用已经采集的 1,200 条原始轨迹，24–35 层各 100 条，任务文字为 `Press 24 floor.` 至 `Press 35 floor.`。不重新仿真或补造图像：逐条依据接触力、按钮行程和实际灯状态确定首次成功的采样点，再同步裁剪数值和两路视频。

| 内容 | 路径 |
| --- | --- |
| 训练数据输出 | `datasets/piper_elevator_lerobot_press_30hz` |
| 可恢复的转换分片 | `datasets/piper_elevator_parts_press_30hz` |
| 不可变裁剪计划与验证记录 | `outputs/press_only_30hz/` |
| 完整原始记录 | `datasets/piper_elevator_raw_edge_30hz` |
| 历史完整动作 LeRobot 版本 | `datasets/piper_elevator_lerobot_edge_30hz` |

**全部 1,200 条已转换并通过最终验收。** 同步采样从 **646,736 → 306,230**，移除 340,506 帧；每条保留 223–314 帧，视频时长约 7.43–10.47 秒。新训练目录大小为 **7,720,627,762 字节（7.72 GB / 7.19 GiB）**，包含发布后读回报告，不含原始记录与转换分片。裁剪后每层仍有 100 条不同的规划 action 序列。

最终验收覆盖全部数值行、源文件哈希、120 Hz 接触与灯状态、首次成功终点及末目标处理；96 个输出视频共 **612,460 帧 RGB** 全部解码，30 Hz 时间戳和片段边界正确。全部 1,200 条腕部画面的可见目标区域在终帧之前无亮灯反馈、末帧亮灯。全局画面有 100 条 26 层目标被机械臂遮挡，腕部反馈全部可见。

发布后使用官方 **LeRobot 0.6.1 / PyAV** 再次覆盖每条首/中/末，共 **3,600 个样本、7,200 张图像、3,600 个 8 步动作窗口、1,200 次末尾 padding、900 个 DataLoader batch**，全部通过。参见质量汇总（本地产物：`outputs/press_only_30hz/quality_summary.json`）、全量内容审计（本地产物：`datasets/piper_elevator_lerobot_press_30hz/meta/audit.json`）、发布目录读回（本地产物：`datasets/piper_elevator_lerobot_press_30hz/meta/published_readback.json`）和正式数据末两帧对照（本地产物：`outputs/press_only_30hz/final_first_light_endpoints.png`）。

另将每层一条的缩短动作与此前已执行的完整物理回放逐值核对，12 条前缀一致，且在新的终点处目标按键均已实际亮起。前缀等价验证（本地产物：`outputs/press_only_30hz/replay_prefix_equivalence.json`）复用既有物理记录；本次没有重新仿真 1,200 条。此前回放的关闭阶段退出码 143 仍如实保留，见[回放说明](replay.md)。

## 数据与终点定义

两路 D435 RGB 均保持 **640×480、30 fps**，state/action 为 30 Hz；PhysX 为 120 Hz，每 4 步采样一次。`observation.state` 和 `action` 均为 8 维：

```text
[x_m, y_m, z_m, qw, qx, qy, qz, gripper_width_m]
```

坐标系为机械臂基座 `base_link`，夹爪 TCP 在 link6 局部 `[0, 0, 0.1358]` m；`state` 是实际观测位姿，`action` 是下一采样时刻的绝对规划目标。夹爪总开度保持约 8 mm。

设原始帧 `k` 首次显示目标按钮亮起，则保留帧 `0..k`。此前 action 完全保留；**末帧 action 复制原始 `action[k-1]`**，即已经执行并导致该终端观测的规划目标，避免指向截断点之后。诊断字段 `action.joint_target` 同样复制原始 `q_target[k-1]`。末目标不以测得的 state 替代。LeRobot 的 action chunk 越过 episode 终点时重复这个末目标，并提供 padding mask。

每条 episode 恰好保留一张目标灯亮起的终端采样，其余目标灯采样均未亮；在两次 30 Hz 采样之间可能已有实际接触，最大采样等待为 25 ms。时间戳、实际状态和图像仍对应原采样时刻。图像经 H264 重新编码，因此与原解码图像之间允许有损编码差异。

`meta/collection_metadata.json` 原样保留采集配置、标定和指纹，其中完整动作的成功条件描述的是原始采集。派生训练数据的终点规则、末目标策略与来源哈希由 `meta/export_manifest.json`、`meta/cut_plan.json` 明确定义。转换重新生成视频、全局索引和统计量。

## 读取

在项目根目录使用已配置的 `.conda/envs/lerobot`（LeRobot 0.6.1、PyAV）：

```python
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "local/piper_elevator_press",
    root=Path("datasets/piper_elevator_lerobot_press_30hz").resolve(),
    video_backend="pyav",
)
sample = dataset[0]
state = sample["observation.state"]
action = sample["action"]
wrist = sample["observation.images.wrist"]
global_view = sample["observation.images.global"]
```

图像由官方读取器返回为 `[3, 480, 640]`、浮点范围 `[0,1]`。全局 `episode_index` 重建后与原始编号可能不同，追溯使用 `source_episode_id` 或清单中的显式映射。

## 从完整原始记录重建

以下命令不覆盖已有裁剪计划。当前计划已生成；续转换直接运行第二条命令。对新采集批次，应为原始数据、计划、输出和分片各使用新路径。

```bash
PYTHONPATH=src .conda/envs/pressb/bin/python -m pressb.press_prefix \
  --raw datasets/piper_elevator_raw_edge_30hz \
  --output outputs/press_only_30hz/cut_plan.json \
  --episodes-per-floor 100

.conda/envs/lerobot/bin/python scripts/export_press_dataset.py \
  --raw datasets/piper_elevator_raw_edge_30hz \
  --cut-plan outputs/press_only_30hz/cut_plan.json \
  --output datasets/piper_elevator_lerobot_press_30hz \
  --parts datasets/piper_elevator_parts_press_30hz \
  --episodes-per-task 100 --part-size 25 --workers 4
```

该转换链针对本批 30 Hz / 120 Hz 数据。每个分片和最终合并均通过独立内容审计后才发布；续转换核对源文件、计划、实现版本和已提交内容，禁止混入其他来源或裁剪设置。原始完整视频及历史 LeRobot 版本均不改写。

预检发现 LeRobot 0.6.1 的官方聚合器在重排编号后，`meta/stats.json` 的全局 `index` / `episode_index` 统计仍使用分片内编号。新导出器从最终 Parquet 实际值重新计算这两列的全局统计，并核对全部 episode 的编号统计；其它字段沿用重新裁剪后生成的官方统计。原始失败证据保存在 `outputs/press_only_30hz/failed_pilot_attempt1`，修复后的预检、恢复转换和正式导出均正常退出。
