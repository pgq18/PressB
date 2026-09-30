# 边缘灯带场景：30 Hz LeRobot 数据集

> 文中的数据集、权重、图像和运行报告属于本地产物，不随 Git 仓库分发；新机器请先按 [README](../README.md) 生成场景。历史结果不代表本次安装已经完成验证。

**训练入口为 `datasets/piper_elevator_lerobot_press_30hz`：每条从折叠初始姿态开始，保留至目标按键首次亮起橙色边框的帧。** 后续撤回、释放和归位只保留在原始诊断记录中。1,200 条已完成转换和最终验收，保留 306,230 组同步采样、612,460 帧 RGB；官方 LeRobot 首/中/末样本、批量读取和末端 action padding 全部通过。读取和重建方法见[首次亮灯数据说明](dataset_press_only.md)，证据见质量汇总（本地产物：`outputs/press_only_30hz/quality_summary.json`）。

## 完整原始采集与历史导出

以下记录描述本批的完整原始动作。按用户选择，以 **30 Hz** 采集 24–35 层各 100 条，共 1,200 条；每次按压后回到折叠初始姿态，再开始下一次。按键只有四周橙色边框发光，中央暗面和白色数字不变。

两路 D435 RGB 都为 **640×480、30 fps**，与 state/action 同步。PhysX 保持 **120 Hz**，每 4 个物理步采样一次。state 为当前实测夹爪 TCP 位姿，action 为下一采样时刻（约 **33.333 ms** 后）的绝对规划目标，到轨迹末尾时截断；两者使用机械臂基座坐标系、米、wxyz 四元数与总夹爪开度，均为 8 维。任务文字保持 `Press 24 floor.` 至 `Press 35 floor.`。

本批使用 **raw schema 10** 记录 `fps=30`、`physics_hz=120`、`capture_stride=4`、`action_horizon_s=1/30`，LeRobot 格式仍为 **v3.0**。旧 schema 7/9 的 10 Hz 数据仍可读取，但不能与本批混用或仅修改视频帧率冒充 30 Hz。场景、曝光和边框亮灭机制沿用已验证版本；重复 Dome 仍真正停用。

已从本数据集抽取每层一条，将保存的末端 action 经逆运动学送回 PhysX 执行，**12/12 实际按键任务通过独立验收**，双相机回放视频完整。回放方法、视频和关闭阶段退出码 143 的说明见[动作回放文档](replay.md)。

冻结场景、标定和配置在 `outputs/edge30_source/`。正式输出使用独立目录：

| 内容 | 路径 |
| --- | --- |
| 原始采集 | `datasets/piper_elevator_raw_edge_30hz` |
| 历史完整动作 LeRobot 数据集（含撤回/归位） | `datasets/piper_elevator_lerobot_edge_30hz` |
| 历史完整动作转换分片 | `datasets/piper_elevator_parts_edge_30hz` |
| 验证记录 | `outputs/edge30_validation/` |

30 Hz 预检已完成：6 个并行环境覆盖 12 层，得到 6,417 个同步采样、12,834 帧 RGB，物理、照明、亮灭和全视频几何同步检查通过。另一个 12 条预检批次已完成 LeRobot 转换、全视频解码及发布目录官方读取验证。3 环境预检的一处全局视角颜色分类误报及诊断证据保留于 `outputs/edge30_validation/`：实际边框在第 256 帧仍亮、第 257 帧熄灭，原橙色像素规则漏检偏黄色的边框。

正式采集于 **2026-09-27 23:38:18（北京时间）** 启动，于 **2026-09-28 01:51:33** 采齐。运行清单为 `datasets/piper_elevator_raw_edge_30hz/collection_run.json`，使用 8 张 GPU、每张 6 个环境；全部进程正常退出。共 **1,200 条、646,736 个同步采样、2,583,344 个物理步、1,293,472 帧双相机 RGB**，每层恰好 100 条，1,200 条规划轨迹哈希各不相同。采集源码、场景和标定副本存于原始目录的 `provenance/`，采集指纹始终一致。

**48 个转换分片、合并内容审计和最终目录官方读取验证已全部完成。** 正式 LeRobot v3.0 数据位于 `datasets/piper_elevator_lerobot_edge_30hz`，读取方法见数据集 README（本地产物：`datasets/piper_elevator_lerobot_edge_30hz/README.md`）。合并后的 96 个视频共 1,293,472 帧已完整解码；数值、源文件哈希、基座坐标系、任务数量和每条首/中/末图像对应关系全部通过。发布后再次用官方读取器覆盖全部 1,200 条、3,600 个样本、7,200 张图像，于 **2026-09-28 02:33:41（北京时间）** 通过。

完成状态已写入采集结果（本地产物：`datasets/collection_result_edge_30hz.json`）（`status=complete`、`validation_complete=true`）。全部证据和 SHA 见质量汇总（本地产物：`outputs/edge30_validation/quality_summary.json`），完整内容报告见LeRobot 审计（本地产物：`datasets/piper_elevator_lerobot_edge_30hz/meta/audit.json`）及最终目录读回（本地产物：`datasets/piper_elevator_lerobot_edge_30hz/meta/published_readback.json`）。

原始数据的最终检查已经完成：

- 独立物理/状态/action 检查覆盖全部 1,200 条，每层 100 条。
- 两路相机共 2,400 张首帧照明检查全部通过；相对同配置单环境参考，最大亮度差为腕部 3.86625%、全局 0.53750%，无过曝。
- 每条按压/释放窗口均检查，共 112,162 张 RGB。2,300 个可见视角亮灭对齐，首灭帧反馈像素均为 0；另外 100 个视角均为 26 层全局相机被机械臂遮挡，腕部视角全部通过。
- 完整运动几何同步抽查 68 条、136 路视频，共 71,666 帧，全部实际 30 Hz、时间戳匹配、几何时延为 0。此项结论限于这些抽查视频。

边框在当前曝光和色调映射下呈橙色至琥珀色。反馈审计采用冻结的 `amber_orange_hsv_v2` 色相/饱和度规则，避免原先红绿比阈值漏检仍亮着的边框；可见性和时间对齐阈值保持不变。两组预检、30 Hz 人工提前/延后一帧的负例及旧真实延迟熄灯负例均已验证。原误报、负例、规则与源码 SHA 保留于 `outputs/edge30_validation/amber_classifier_v2_validation.json`。

以下保留原始采集和历史完整动作转换的复现命令；当前训练导出使用[首次亮灯转换命令](dataset_press_only.md#从完整原始记录重建)。GPU 和环境数量须沿用正式运行清单，不要重复启动同一批次：

```bash
.conda/envs/pressb/bin/python scripts/collect_parallel.py \
  --snapshot outputs/edge30_source/scene.usda \
  --config outputs/edge30_source/config.json \
  --output datasets/piper_elevator_raw_edge_30hz \
  --fps 30 --gpus 4,6,0,1,2,3,5,7 --num-envs 6 \
  --episodes-per-task 100 --seed 20260926

.conda/envs/lerobot/bin/python scripts/export_lerobot.py \
  --raw datasets/piper_elevator_raw_edge_30hz \
  --output datasets/piper_elevator_lerobot_edge_30hz \
  --parts datasets/piper_elevator_parts_edge_30hz \
  --episodes-per-task 100 --part-size 25 --watch
```

导出器从不可变采集元数据读取实际 fps，无需再传一次帧率。`--fps` 也接受 120 的其他正整数约数（如 10、20、60）；更改频率必须使用独立原始目录。采集参数会覆盖单场景预览的相机采样间隔，并将有效配置保存到采集指纹中。
