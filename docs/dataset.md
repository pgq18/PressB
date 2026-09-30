# Piper 电梯按键：10 Hz 历史批次与 LeRobot 读取

> 文中的数据集、权重、图像和运行报告属于本地产物，不随 Git 仓库分发；新机器请先按 [README](../README.md) 生成场景。历史结果不代表本次安装已经完成验证。

**当前训练 episode 在目标按键首次亮起橙色边框时结束，不包含之后的撤回或归位。** 训练入口为 `datasets/piper_elevator_lerobot_press_30hz`，1,200 条、306,230 帧已完成转换、全量内容审计和官方 LeRobot 发布目录读取验证，见[训练数据说明](dataset_press_only.md)。完整原始采集和含归位的历史导出见 [30 Hz 采集说明](dataset_30hz.md)。本文以下 v9 批次、10 Hz 命令和边框初版预检均为历史记录。

## 边缘灯带版本

当前场景改为 **仅四周 2 mm 橙色边框发光**，按键中央暗面和白色数字使用固定材质，按下亮、回弹灭。场景为 `outputs/edge_feedback/scene.usda`。初版边框预检采用 schema 9、10 Hz；当前采集器默认采用 schema 10、30 Hz 及独立的 `datasets/piper_elevator_raw_edge_30hz` 输出目录。场景、频率与源码 SHA 参与采集指纹，不会把不同外观或频率混入同一批次。

已有 `piper_elevator_*_v9` 的 1,200 条数据仍保留整面发光外观，没有重写历史视频。下文 v9 和边框初版的 10 Hz 命令仅作历史说明；当前采集使用 [30 Hz 文档](dataset_30hz.md)中的冻结场景、配置和目录，旧数据的审计和读取命令仍可使用。边框初版的验证样例独立保存到 `datasets/piper_elevator_raw_edge_preview`，每层一条；这不是当前 30 Hz 的 1,200 条正式数据集。

```bash
# 历史单场景演示入口：10 Hz RGB-D，非 LeRobot 批量采集
bash scripts/run.sh --headless --gpu 4 --video --output outputs/edge_feedback

# 保留的 10 Hz 边框采集示例；当前 30 Hz 入口见 docs/dataset_30hz.md
.conda/envs/pressb/bin/python scripts/collect_parallel.py \
  --snapshot outputs/edge_feedback/scene.usda \
  --output datasets/piper_elevator_raw_edge \
  --fps 10 --gpus 4 --num-envs 3 --episodes-per-task 100 --seed 20260926
```

新边框随按钮实体回弹，不改变碰撞面、弹簧参数、亮灭阈值、坐标系或动作语义。旧快照仍可显式通过 `--snapshot` 打开，保留其原外观；不会静默改写已采集场景。

**边框初版的历史 10 Hz 验证已完成**：3 个并行环境、12 层各 1 条，合计 2,150 个状态采样、4,300 帧 RGB，物理记录和 24 路完整视频检查通过，全部视频的整体最佳几何滞后为 0 帧。按压亮灭窗口共检查 370 个 RGB 帧，23 个可见视角首灭帧的橙色像素均为 0；26 层全局视角受遮挡，由腕部确认。该轮边框验证没有降低原审核阈值。另有 26 项项目测试通过，USD 材质审计验证 36 个克隆按钮、144 次开关无串扰，数字与中央材质始终不变。见验证汇总（本地产物：`outputs/edge_feedback_validation/summary.json`）、双相机示例（本地产物：`outputs/edge_feedback_validation/camera_sync/episode_000011_floor35_overview.png`）和回弹逐帧图（本地产物：`outputs/edge_feedback_validation/camera_sync/episode_000011_floor35_wrist_release.png`）。按压杆和夹爪仍可能自然遮挡部分数字。

附加单场景演示运行在完成 12 次按压、释放、归位日志后，于额外静态视图导出阶段退出 143，原因未知；其完整 `report.json` 和视频收尾未完成，记录在 `outputs/edge_feedback/run_status.json`。初始场景 USD 已生成并由上述独立采集实际加载验证。完整动作视频请使用 `datasets/piper_elevator_raw_edge_preview/episode_000000` 至 `episode_000011` 中的 `wrist.mp4`、`global.mp4`，不要将这次附加演示声明为完整验收成功。

## 已完成的历史 10 Hz 整面发光 v9 批次

**schema 9 原始数据已采齐 1,200 条，24–35 层各 100 条**，保存在 `datasets/piper_elevator_raw_v9`。8 个 worker 均正常退出，提交记录包含 **216,766 个同步状态采样、2,587,992 个物理步、两路合计 433,532 帧 RGB**，episode ID 完整，1,200 个轨迹哈希互不相同，均记录采集时物理成功；代码与采集指纹保持不变。依据见采集结果（本地产物：`datasets/collection_result_v9.json`）及运行清单（本地产物：`datasets/piper_elevator_raw_v9/collection_run.json`）。

**历史 10 Hz LeRobot 数据集已完成转换与最终验收，读取路径为 `datasets/piper_elevator_lerobot_v9`（LeRobot v3.0 格式）。** 全量独立物理审计通过；96 个视频共 433,532 帧完整解码通过；官方读取器逐 episode 抽取首/中/末样本，共核对 7,200 张图像，并完成状态/动作、基座坐标系、任务数量和来源哈希检查。见数据集说明（本地产物：`datasets/piper_elevator_lerobot_v9/README.md`）、内容审计（本地产物：`datasets/piper_elevator_lerobot_v9/meta/audit.json`）及质量汇总（本地产物：`outputs/dataset_visual_audit/v9_quality_summary.json`）。

所有采集、转换和读取均在本地进行，没有上传 Hugging Face Hub。`local/piper_elevator_1200` 是本地数据集标识；转换和审计脚本使用 Hub/Datasets 离线模式。该历史批次中间分片位于 `datasets/piper_elevator_parts_v9`。

旧 schema 7 的 `datasets/piper_elevator_raw` 与 `datasets/piper_elevator_lerobot` 已确认受重复 Dome 灯光错误影响，**不推荐用于训练**；原文件、哈希和报告完整保留，并附有 `DO_NOT_TRAIN_LIGHTING_BUG.md` 标记。其历史物理、数值和解码检查通过，不能证明照明正确；两条固定视角的释放后残色失败也保留在下文。更早的 `datasets/*_v2_render_delay` 图像相对状态滞后约 100 ms，不得用于训练或作为新批次的续采来源。

## schema 9 已完成的图像检查

| 检查范围 | 当前结果 | 证据 |
| --- | --- | --- |
| 1,200 条的双路首帧，共 2,400 帧 | 全部通过固定 ±5% 亮度差和饱和像素比例 ≤1% 判据；腕部最大差 **3.2413%**、全局 **0.7046%**，实际饱和比例均为 **0** | 全量首帧照明（本地产物：`outputs/lighting_fix_validation/v9_full_initial_lighting.json`） |
| 1,200 条的完整亮灯区间及两侧各 3 帧，共 37,402 个窗口 RGB 帧 | 全部 episode 通过；**2,297 个可见视角**亮灭对齐，首灭帧目标橙色像素均为 **0** | 全量按钮反馈（本地产物：`outputs/dataset_visual_audit/v9_all_button_feedback.json`） |
| 103 个全局视角可见性不足 | 26 层 100 条、32 层 3 条；受机械臂遮挡，不能据该视角判断灯状态。**全部 1,200 条腕部视角均确认亮灭正确** | 同上，逐视角记录保留 `sufficient_visibility=false` |
| 26 条、52 路完整运动视频的状态/图像几何抽查 | 整体最佳几何滞后均为 **0 帧**，视频时间戳和帧数检查通过 | 首批 12 条（本地产物：`outputs/dataset_visual_audit/v9_preflight/report.json`）、环境 2 的 60/63（本地产物：`outputs/dataset_visual_audit/v9_third_env/report.json`）、末批 1188–1199（本地产物：`outputs/dataset_visual_audit/v9_final_samples/report.json`） |

亮度是解码 RGB 的加权均值，不是物理照度。首帧检查只验证每条 episode 的初始照明；全量按钮反馈检查只覆盖按压亮灭窗口；完整运动几何检查覆盖上述 26 条抽查样本。不能将这些不同范围的检查合并表述为“1,200 条每一帧视觉全部验证”。原始物理复算、完整视频解码、官方 LeRobot 读取及转换数值核对由独立内容验收执行。

## 并行 Dome 灯光错误与修复

旧克隆代码将完整 `/World` 引用到每个并行环境，环境 0 的 Dome 强度为 845，其他 Dome 强度设为 0，但节点仍处于活动状态。Dome 是全局环境光，在当前 RTX 配置下这些零强度活动节点仍影响环境光选择，抑制了原本应共享的照明。单纯提高灯光强度未解决这个根因。

`outputs/lighting_ablation/summary.json`（本地产物：`outputs/lighting_ablation/summary.json`） 的三环境对照保持场景和灯光强度相同，仅停用重复 Dome。初始位姿的腕部/全局平均亮度分别从 **18.77/45.37** 恢复到 **85.11/142.47**（0–255）；35 层按压位姿从 **26.13/37.73** 恢复到 **78.44/129.54**。这些是固定对照帧的统计，不代表所有 episode 的平均亮度。两个对照报告确认源配置和快照文件未变。

`src/pressb/dataset_scene.py` 已改为对环境 1 及之后的重复 Dome 调用 **`SetActive(False)`**，并校验整个 Stage 只有环境 0 的一个活动 Dome。各环境的局部 RectLight、机器人、按钮和相机保留。CPU USD 回归测试覆盖 1/2/3 个环境、13 个世界关节锚点、源快照不变和额外全局 Dome 的拒绝。**维持 `lighting_intensity_scale=1.3`，不再调高强度**；Dome/顶灯/补光仍为 845/2470/1430。schema 9 将灯光克隆策略及相关源码哈希纳入采集指纹，防止与旧暗图像混用。旧视频不会因代码修复而自动变亮，必须另行采集。

### 静态对照与生产首帧的区别

静态单环境报告 `outputs/lighting_fix_validation/single/report.json` 使用静态姿态回放、额外渲染收敛和 PNG 保存；生产采集使用实际控制、预热后的物理状态和 MP4 首帧。因此，两者具有相同名义布局和相机标定，但并非完全相同的采图历史或实际关节姿态。生产单环境参考 `outputs/lighting_fix_validation/single_production` 通过同一 schema 9 采集器独立记录 episode 0，采集指纹与并行预检一致，worker 记录确认 `num_envs=1` 且正常完成，灯光记录确认只有环境 0 的一个活动 Dome。

静态首帧的腕部/全局平均亮度为 **85.4819/142.4570**，生产单环境首帧为 **83.5905/141.3098**。修复后 12 条并行预检直接对照静态 PNG 时，ID 8、9、11 的腕部差异为 **−5.069%、−5.010%、−5.058%**，略超预先固定的 ±5% 判据；该失败报告（本地产物：`outputs/lighting_fix_validation/v9_preflight_initial_lighting.json`）保持不变。

独立空间诊断（本地产物：`outputs/lighting_fix_validation/static_vs_production_first_frame_analysis.json`）显示，静态和生产单环境的实测腕部相机差约 **46.6 µm、0.000621 rad**，按钮边框投影差中位 **0.073 px**、最大 **0.156 px**；亮度差分布于多个图像区域。生产单环境与并行 episode 0 的实测关节和 FK 相机位姿相同。证据不支持明显视角偏移，但也不能把余下差异唯一归因于阴影、降噪或编码中的某一项。

使用**相同生产采图链路的单环境首帧**作为参考后，12 条预检两路全部通过相同的 **±5% 亮度差、饱和像素比例 ≤1%** 判据，腕部最大差约 **2.92%**，全局最大差约 **0.70%**，实际饱和比例均为 0，见生产参考预检报告（本地产物：`outputs/lighting_fix_validation/v9_preflight_production_reference.json`）。此检查隔离并行与单环境采集的差异，未放宽阈值，也不覆盖静态参考的原有失败。正式采集沿用此参考；随后全量 1,200 条的 2,400 个首帧均已通过，最大差分别为 **3.2413%/0.7046%**，见上方全量报告。

`scripts/audit_dataset_lighting.py` 每条 episode 只解码两路视频各自的首帧，核对 metadata 与 `frames.npz` 的归位姿态、按钮全灭和首采样索引；检查 640×480 RGB、加权 RGB 均值和任一通道 ≥250 的像素比例，并按 episode、相机和环境编号汇总。它验证**每条 episode 的初始照明**，不证明全运动视频每一帧的照明、亮灭时序或几何同步；这些需由独立视觉和物理审计验证。报告不覆盖已有文件，重复检查应指定新的输出路径。

```bash
.conda/envs/lerobot/bin/python scripts/audit_dataset_lighting.py \
  datasets/piper_elevator_raw_v9 \
  --reference-raw outputs/lighting_fix_validation/single_production \
  --episodes-per-task 100 \
  --output outputs/lighting_fix_validation/v9_full_initial_lighting_recheck.json
```

`--reference-raw` 要求来源与参考的 schema、采样率和采集指纹一致，并校验单环境 worker、灯光和实际初始状态的记录。当前脚本支持 schema 9/10；本节路径对应历史 schema 9。静态模式使用 `--reference outputs/lighting_fix_validation/single/report.json`，结论单独报告。`--allow-partial` 仅放宽完整数量要求，不能作为完成 12×100 的判据。旧 schema 7 的 1,200 条已对静态参考执行负例检查：**2,400 个首帧全部明显偏暗，0 条通过**，结果保留在旧批次负例报告（本地产物：`outputs/lighting_fix_validation/old_schema7_initial_lighting.json`）。

## 旧批次照明调整记录

旧批次采集前将环境照明提高 **30%**：`configs/scene.json` 设置 `lighting_intensity_scale=1.3`，环境光、顶灯和补光灯的 USD intensity 分别从 650、1900、1100 调整为 **845、2470、1430**。`outputs/latest/scene.usda` 是供新采集使用的加亮快照；调整前的场景、报告和双相机内参保存在 `outputs/lighting_before/`。现有 `outputs/latest` 运动视频及 RGB-D 文件仍来自原照明，未重新渲染，报告的 `recorded_episode_config` 和 `lighting_provenance` 区分旧记录与新快照。加亮后已独立完成 24–35 层各 1 条预检：**12/12 条物理审计通过，共 2,150 个同步状态采样、双路合计 4,300 帧 RGB**；该批双相机图像审计也通过。后续全量物理与末段视觉抽查的不同结果见下文。

亮度对比样本的平均亮度（0–255）为：腕部视角 **13.77 → 17.39（+26.3%）**，全局视角 **37.37 → 46.35（+24.0%）**；全局面板区域 **20.47 → 26.67**。这些是选定对比帧的统计，不代表所有运动帧。对比图及数据为 `outputs/lighting_comparison.png`、`outputs/lighting_comparison.json`；抽查按压图像的亮度 ≥250 像素比例均为 0，详见 `outputs/lighting_press_saturation.json`。

加亮后的图像报告 `outputs/dataset_visual_audit/bright_preflight/report.json` 对全部 12 层的两路视角都测得整体最佳几何滞后为 **0 帧**。腕部视角的首个灭灯采样均无目标橙色像素；全局视角第 26 层在接触时被夹爪/腕部组件遮挡，其灯光由腕部视角确认。其它可见目标的主要颜色切换与标签对齐，但少量棕橙边缘残留仍存在，首灭帧最大橙色像素比例为 **16.24%（第 33 层）**，低于原有 25% 判定阈值。保留这些测量值和人工复核记录 `manual_review.json`，不把“审计通过”描述成全程无遮挡或所有全局帧残影为零。

## 旧 schema 7 的图像质量与保留失败

现有 **schema 7** 数据、标签和图像判定阈值保持不变。视觉证据汇总在 `quality_summary.json`（本地产物：`outputs/dataset_visual_audit/quality_summary.json`） 和中文说明（本地产物：`outputs/dataset_visual_audit/README.md`）。抽查范围为首批 ID 0–11、第三个并行环境 ID 60/63，以及后段 ID 1188–1199；共 **26 条、52 路视频、9,344 帧 RGB**，包含不同 seed 和 Y=12 m 的环境偏移。各路整体最佳几何滞后均为 **0 帧**；全部抽查腕部视角的亮灭标签对齐，首灭帧目标橙色像素均为 0。该结论覆盖抽查样本，不代表已逐条完成 1,200 条的颜色过渡检查。

首批与第三环境自动检查通过；后段报告（本地产物：`outputs/dataset_visual_audit/final_samples/report.json`） 保留 `success=false`，原因如下：

| 固定相机样本 | 首灭帧橙色像素 / 亮灯平台 | 可见核心亮度：末亮 → 首灭 | 解码 RGB 颜色混合系数 |
| --- | --- | --- | --- |
| ID 1189，25 层，seed 20262115 | 40 / 135 = **29.63%** | 168.70 → 67.54 | 0.243 |
| ID 1199，35 层，seed 20262125 | 92 / 338 = **27.22%** | 175.83 → 85.37 | 0.247 |

两者超过冻结的 `max(5, 亮灯平台橙色像素数 × 25%)` 判据，主要灭灯检测比标签晚 1 个采样。人工查看首灭帧已明显变暗，但弱棕橙残留真实存在，未用人工判断覆盖失败。像素数比例并不是发光强度；混合系数仅在固定核心像素上按解码 RGB 参考拟合，不是物理辐射测量。测量与细节图（本地产物：`outputs/dataset_visual_audit/final_samples/release_color_analysis.json`） 保留原始比较。26 层在两组全楼层样本中的固定视角灯光受夹爪/腕部遮挡，记为无法判定，由腕部视角确认。

这些局部颜色过渡限制不推翻已通过的物理与状态/图像几何检查；也不能将首批通过外推为“1,200 条视觉全部通过”。样本并非随机抽样，不据此推算全量残色比例。最终 LeRobot 内容审计验证来源、完整解码及数值语义，与这些视觉限制分别报告。

## 历史 10 Hz 批次的数据内容与语义

本节的 0.1 s action horizon、10 fps 和 12 步采样间隔对应历史批次；当前 30 Hz 分别为约 33.333 ms、30 fps 和 4 步，见[当前数据语义](dataset_30hz.md)。两批的基座坐标系、TCP 定义、8 维字段顺序和任务文本一致。

每条 episode 只完成一个指定楼层：从折叠初始姿态出发，接近并按下按钮，接触与位移满足条件后亮橙灯，松开回弹后熄灭，最后收回相同初始姿态。必须通过实际 PhysX 按压、释放、归位及无异常接触检查后，原始 episode 才会提交。不是将一次演示复制成多条样本。

任务文本精确包含以下 12 项，保留大小写、空格和句号：

| 楼层 | task |
| --- | --- |
| 24 | `Press 24 floor.` |
| 25 | `Press 25 floor.` |
| 26 | `Press 26 floor.` |
| 27 | `Press 27 floor.` |
| 28 | `Press 28 floor.` |
| 29 | `Press 29 floor.` |
| 30 | `Press 30 floor.` |
| 31 | `Press 31 floor.` |
| 32 | `Press 32 floor.` |
| 33 | `Press 33 floor.` |
| 34 | `Press 34 floor.` |
| 35 | `Press 35 floor.` |

`observation.state` 和 `action` 均为 **8 维 float32**，按以下顺序保存：

```text
[x_m, y_m, z_m, qw, qx, qy, qz, gripper_width_m]
```

位置和姿态相对于机器人 **`base_link`**，不包含并行环境的世界平移。四元数为 **wxyz**，夹爪宽度是两指之间的总开度，单位米，名义值为 **0.008 m**。

这里的末端是夹爪 TCP：`link6` 沿局部 +Z 平移 **0.1358 m**。按压杆尖端位于 +Z **0.24 m**，仅用于接触和按压规划；数据集的 TCP 位姿不能用杆尖位姿替代。

- **state**：由当前物理步的实际关节角，通过官方 URDF 正运动学计算 TCP 位姿；开度使用实际两指关节位置之差。
- **action**：由未来 **12 个物理步，即 0.1 s** 后的规划关节目标计算绝对 TCP 目标位姿；接近 episode 末尾时截断到最后一个规划目标。它是未来绝对目标，不是增量，也不是当前实测 state 的复制。
- **图像**：`observation.images.wrist` 和 `observation.images.global`，两路均为 **640×480 RGB、10 fps**，同一渲染时刻采集。固定全局相机使用当前已验证的桌面机位；机械臂允许裁切，按压时可自然遮挡面板局部。
- **物理原始记录**：120 Hz，保留目标/实际关节角、夹爪位置、按钮行程、接触力和灯状态，供独立复算。此次 LeRobot 图像任务不导出深度。

另外保存 `observation.joint_position`、`action.joint_target`、`observation.sim_time`、`source_episode_id`、`source_seed` 和 `floor`。LeRobot 自动生成的 `timestamp`、`frame_index`、`episode_index` 等保留官方语义，实际仿真时间单独存放在 `observation.sim_time`。

旧批次原始格式为 **schema 7**，当时修复 Dome 后的采集器使用 **schema 9**；当前 30 Hz 采集器使用 schema 10。以下描述历史 10 Hz 采图时序。每 12 个物理步，在读取实际状态并更新灯光后同步采图。普通采样调用 `rep.orchestrator.step(delta_time=0.0, pause_timeline=False, rt_subframes=2)`；若自上次采图以来有按钮亮灭变化，先额外执行 **4 次独立的零时间步长、2 子帧调用**，再执行 **16 子帧的最终采图调用**。仅把最终 tiled RGB 复制写入视频，前面的稳定化渲染不增加视频帧数或标签。初始化关闭 `capture_on_play`；抗锯齿使用 **FXAA（`anti_aliasing=2`）**。同步由 Replicator 的渲染等待完成，不能用两次 `world.render()` 替代。采集器逐次断言整个采图过程前后物理时间和实际关节角完全不变，确保额外渲染不推进动作。

同时保留 `renderer_history_controls` 配置：关闭间接光时间滤波和直接光时空重采样，并将所列 NRD 历史长度设为 0。这些是固定的渲染控制参数；记录键和值不代表对应 NRD 分支必然启用，也不将颜色残影消除归因于这些参数。各进程的 `render_settings_worker_XX.json` 保留设置前值、应用值及降噪方法信息，实际效果以图像审计为准。

schema 7 已在原照明下的 29、35 层双相机诊断中通过几何同步与亮灭检查，释放后的首个灭灯采样在目标按钮区域检测到的橙色像素均为 0；报告位于 `outputs/dataset_visual_audit/capture_settle/report.json`。`collection_metadata.json` 记录 schema、采图方法、普通/亮灭采样子帧数、额外采图次数、历史控制参数及抗锯齿设置，并将其计入采集指纹；旧格式数据不能混入本批正式输出。原照明高层诊断只证明该次诊断；旧 schema 7 加亮场景的抽查结果及两项残色失败见上文。

schema 9 默认在灯光变化后、稳定化渲染前调用 `omni.usd.get_context().reset_renderer_accumulation()`，其余零时间步采样语义不变；`--reset-renderer-accumulation` 仅作为兼容参数保留，不再切换 schema。此前可选开启此处理的 **schema 8** 在 24、25、29、35 层四条独立诊断中测得两路几何滞后为 0、首灭帧橙色像素为 0，见 `outputs/dataset_visual_audit/reset_flag/report.json` 与 `reset_history/report.json`。其中 25、35 层复用了失败样本的 seed 和环境偏移。**四条历史诊断没有重写旧 1,200 条；schema 9 的实际检查使用上方独立的新批次报告。**

轨迹按 seed 进行小幅变化：接近距离在配置值附近 ±3 mm，关节运动速度在基准的 0.88–1.0 倍，按压和回撤时长各 1.30–1.65 s，保持按压 0.32–0.46 s，最终归位停稳 0.60–0.75 s。场景、按钮中心与按压深度、初始姿态、相机及 8 mm 夹爪开度保持一致。每条轨迹都经过限位、速度、几何和实际物理检查；随机参数和轨迹哈希保存在 episode 元数据中。

## 两个独立环境

物理采集继续使用 `.conda/envs/pressb` 的 Python 3.10 / Isaac Sim 4.5。LeRobot 转换与审计使用独立的 `.conda/envs/lerobot`：Python 3.12、LeRobot 0.6.1、数据格式 v3.0、CPU PyTorch 2.10.0 和 torchvision 0.25.0。不要将 `requirements-dataset.txt` 安装到 Isaac 环境。

```bash
cd /path/to/PressB
bash scripts/setup_dataset_env.sh
```

安装脚本先从 PyTorch 官方 CPU 索引安装 torch/torchvision，再安装 `requirements-dataset.txt`，并执行依赖检查。环境路径可通过 `PRESSB_DATASET_ENV` 覆盖，Conda 可通过 `CONDA_BIN` 指定；自定义环境后，以下转换和审计命令应使用对应的 Python。安装日志、版本和冻结依赖分别保存为 `logs/install-dataset-env.log`、`logs/install-dataset-versions.json`、`logs/install-dataset-requirements.txt`。系统需已有 `ffmpeg`，采集用它流式编码 RGB 视频。

## 历史 schema 9 采集命令（非当前入口）

以下命令保留当时 10 Hz、schema 9 的运行方式，对应目录已采齐 1,200 条。当前采集器默认 schema 10、30 Hz，源码与采集指纹也已变化，不能直接用这些旧命令续采 schema 9，或仅增加 `--fps 10` 绕过指纹检查。当前采集入口统一见 [30 Hz 数据集说明](dataset_30hz.md)；历史原始数据及其审计、读取入口继续保留。

当时先用独立目录执行少量诊断，检查物理结果、双相机图像和数据语义。以下保留的命令只采集最多 2 条，不代表完整任务通过：

```bash
bash scripts/collect_dataset.sh \
  --output datasets/piper_elevator_smoke_v9 \
  --gpu 4 --num-envs 1 --episodes-per-task 100 --max-episodes 2
```

该历史批次使用 `scripts/collect_parallel.py`，GPU 顺序为 **`4,6,0,1,2,3,5,7`**，依次对应 worker 0–7；每个进程 **3 个环境**，共 24 个环境槽位。当时每楼层 1 条的全楼层预检命令为：

```bash
.conda/envs/pressb/bin/python scripts/collect_parallel.py \
  --output datasets/piper_elevator_raw_v9 \
  --gpus 4,6,0,1,2,3,5,7 --num-envs 3 --episodes-per-task 1
```

待 12 个楼层的物理和图像预检均通过、上一轮所有进程退出后，使用相同 GPU 顺序及 seed 续采至每楼层 100 条：

```bash
.conda/envs/pressb/bin/python scripts/collect_parallel.py \
  --output datasets/piper_elevator_raw_v9 \
  --gpus 4,6,0,1,2,3,5,7 --num-envs 3 \
  --episodes-per-task 100 --seed 20260926
```

schema 9 已默认启用亮灭时的渲染累积清理和唯一活动 Dome 策略，不需要额外参数。预检完成后沿用同一目录和参数，将 `--episodes-per-task` 改为 `100`。采集指纹拒绝混用不同 schema、灯光克隆策略、源码或配置；不要覆盖已有元数据绕过检查。当时的 schema 9 采集器不能续采 schema 7/8 目录。以上是历史 schema 9 使用的运行方式；该批原始采集完成情况以 `datasets/collection_result_v9.json` 为准，不能用单个 worker 的状态代替完整验收。

监督器启动并监控各独立 worker，每 20 秒输出已提交数量及子进程状态。每次运行写入 `logs/collection_<UTC时间>/worker_XX.log` 和同目录 `run.json`；`<原始输出目录>/collection_run.json` 保存该目录最近一次运行清单；本批目标路径为 `datasets/piper_elevator_raw_v9/collection_run.json`，不会覆盖旧 `datasets/collection_run.json`。运行清单包含 GPU/worker 映射、PID、完整命令、日志路径及关键源码 SHA256。某个 worker 失败或监督器退出时，只清理该次调用创建的子进程组。不要并发启动两个指向同一原始目录的监督器。

资源较少时也可从空目录选择单 worker 调度。以下单/双 worker 示例是八 GPU 方案的替代方案，不能与正式八 worker 采集同时运行：

```bash
bash scripts/collect_dataset.sh \
  --output datasets/piper_elevator_raw_v9 \
  --config configs/scene.json --snapshot outputs/latest/scene.usda \
  --gpu 4 --num-envs 2 --episodes-per-task 100 \
  --seed 20260926 --workers 1 --worker-index 0
```

`--num-envs` 表示一个 Kit 进程中的环境数量，`--gpu` 为 Isaac/Vulkan 设备编号。请依据实际空闲显存和小规模测试选择环境数。每个环境独立机器人、按钮和灯状态，共享一个物理时钟，以拼接渲染同步取得多路图像。

本机有 384 个逻辑 CPU。为避免多个 Kit 进程各自创建过多线程，采集入口在启动前设置 `PXR_WORK_THREAD_LIMIT=8`，以及 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`；SimulationApp 已同时启用 `--/plugins/carb.tasking.plugin/threadCount=8`、`--/plugins/omni.tbb.globalcontrol/maxThreadCount=8` 和 `--/persistent/physics/numThreads=4`。这些限制只影响采集子进程，不修改系统或其他任务。无界面采集另使用 `create_new_stage=False`，由 World 创建场景，避免等待无用的 viewport handle。参数依据见 [官方 CPU 线程配置说明](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/reference_material/sim_performance_optimization_handbook.html#cpu-thread-count)。

需要两个 GPU worker 时，在两个终端分别运行以下命令；二者共享原始输出目录，但处理不同的 episode ID：

```bash
# 终端 A
bash scripts/collect_dataset.sh \
  --output datasets/piper_elevator_raw_v9 \
  --gpu 4 --num-envs 2 --episodes-per-task 100 \
  --seed 20260926 --workers 2 --worker-index 0
```

```bash
# 终端 B
bash scripts/collect_dataset.sh \
  --output datasets/piper_elevator_raw_v9 \
  --gpu 6 --num-envs 2 --episodes-per-task 100 \
  --seed 20260926 --workers 2 --worker-index 1
```

**八 worker、单 worker 与双 worker 示例是不同的调度配置，请选定一种后保持一致。** episode 分片规则为 `episode_id % workers == worker_index`；恢复时必须沿用同一约定的 `--workers` 数量、seed、配置和场景快照，且所有 worker 的这些参数一致。监督器续采还应保持相同 `--gpus` 顺序，以保留 worker 与设备的对应关系。**严禁同时运行相同 worker-index 的两个进程**，也不要混用不同 workers 数量的活动采集进程。

楼层由 `24 + episode_id % 12` 确定，episode seed 为基础 seed 加 episode ID。已提交且成功的 `episode_XXXXXX/metadata.json` 会被跳过，因此可使用原命令续采。`.inprogress` 是尚未提交的数据；转换器只接收成功提交的 episode，失败与中断目录不应手动改名冒充成功结果。`--max-episodes` 只限制当前进程本次处理的数量；某个 worker 状态为 complete 不等于 1,200 条数据已经齐全。

运行期间不要替换配置、场景快照或相机标定。`collection_metadata.json` 保存配置、场景 SHA、任务列表与相机内参；各 `worker_XX_status.json` 记录对应采集进程的进展。最终每个任务恰好 100 条的判定由转换和独立审计完成。

## 历史 schema 9 的 LeRobot 转换

下面保留历史 v9 目录的转换命令；当前 30 Hz 使用 [dataset_30hz.md](dataset_30hz.md) 中独立的 `*_edge_30hz` 目录。转换器可按完整分块持续转换，集齐所有任务后生成最终本地目录：

```bash
.conda/envs/lerobot/bin/python scripts/export_lerobot.py \
  --raw datasets/piper_elevator_raw_v9 \
  --output datasets/piper_elevator_lerobot_v9 \
  --parts datasets/piper_elevator_parts_v9 \
  --episodes-per-task 100 --part-size 25 --watch
```

如果所有原始 episode 已采齐，可去掉 `--watch`。该历史批次显式使用中间目录 **`datasets/piper_elevator_parts_v9`**；续转也应保留同一 `--parts` 参数，不应依赖省略参数后的默认命名。`--part-size 25` 将目标 1,200 条转换为 **48 个分片，每片 25 条 episode**；这是转换分片，与 GPU worker 的取模分配不同。中间分块可续转；原始目录、分块目录和最终目录必须不同，保持原始文件以供来源校验。转换器对分块目录使用进程锁，避免两个转换器同时写入。

LeRobot 转换使用 H.264、`ultrafast`、每路编码器 **4 个线程**。每路队列容量设为当前分片的 **最大 episode 帧数 + 1**，可容纳整条 episode 和结束标记，避免 LeRobot 0.6.1 的实时流式队列满后超时丢帧。转换后仍以原始帧数、完整视频解码和官方读取结果核对零丢帧；不能仅凭编码进程退出成功判定完成。此策略用于 LeRobot 转换，原始视频由采集器单独编码。

`--allow-partial` 仅用于诊断数据的小规模转换，不能用于宣称满足 12×100 的正式验收。`--watch` 中的最终发布只是完成校验后生成本地目录，不涉及 Hub 上传。

## 历史 schema 9 的独立审计和读取

本节命令读取保留的 10 Hz 数据。当前审计脚本也支持 schema 10、30 Hz，并从元数据读取采样率；当前批次的路径与验收状态见 [30 Hz 文档](dataset_30hz.md)。本页历史视觉报告保留原颜色判据和计数；当前脚本已采用文档说明的 `amber_orange_hsv_v2`，重新审计的颜色像素计数可能不同，应另存报告，不覆盖历史证据。

先检查原始 120 Hz 物理证据及其与 10 Hz 图像标签的关系：

```bash
.conda/envs/pressb/bin/python scripts/audit_raw_dataset.py \
  datasets/piper_elevator_raw_v9 --episodes-per-task 100
```

报告保存为原始目录中的 `physics_audit.json`。此审计逐步重算“接触力和位移触发亮灯、回弹后灭灯”，要求只按中目标楼层，首尾归位且按钮释放，并验证图像确实覆盖亮灯及最终归位。它还核对 120 Hz 实际/目标关节记录、夹爪开度、轨迹哈希和唯一性、每 12 步采样索引，以及 action 使用未来 12 步目标的对齐关系。异常碰撞和 USD/FK 一致性仍使用采集器记录的诊断值。

诊断目录可加 `--allow-partial` 检查现有完整 episode 的物理证据；这会放宽每楼层 100 条的数量要求，不能作为正式采集完成判据。不要用 Python `-O` 运行该脚本，因为它使用断言执行验收。

全量图像按钮反馈按每条 episode 的完整亮灯区间及两侧各 3 帧核对，要求至少一个视角能够确认正确亮灭；视角被遮挡时单独报告，不能将零橙色像素解释为灯已灭。现有结果是 1,200 条全部通过、2,297 个可见视角首灭橙色像素均为 0，详见上方全量报告（本地产物：`outputs/dataset_visual_audit/v9_all_button_feedback.json`）。复核时写入新报告，保留已有证据：

```bash
.conda/envs/lerobot/bin/python scripts/audit_button_feedback.py \
  datasets/piper_elevator_raw_v9 --episodes-per-task 100 \
  --output outputs/dataset_visual_audit/v9_all_button_feedback_recheck.json
```

该检查不替代完整视频解码或全运动几何同步。状态/图像几何使用独立抽查脚本；例如复核末批 12 条：

```bash
.conda/envs/lerobot/bin/python scripts/audit_camera_sync.py \
  datasets/piper_elevator_raw_v9 \
  --episode-ids 1188 1189 1190 1191 1192 1193 1194 1195 1196 1197 1198 1199 \
  --output outputs/dataset_visual_audit/v9_final_samples_recheck
```

不指定 ID 时可使用 `--floors 24 25 26 27 28 29 30 31 32 33 34 35`，按各层最早 episode 抽查。脚本保留可见性不足、残色像素与失败结论，不调整阈值使数据通过。若要复现旧批次后段的两项失败，必须将输入路径改回 `datasets/piper_elevator_raw`，并将审计输出写入新的独立目录，保留既有报告。

然后检查最终 LeRobot 数据集：

```bash
.conda/envs/lerobot/bin/python scripts/audit_lerobot.py \
  datasets/piper_elevator_lerobot_v9 \
  --raw datasets/piper_elevator_raw_v9 --episodes-per-task 100
```

默认报告写入 `datasets/piper_elevator_lerobot_v9/meta/audit.json`。审计检查每楼层数量、精确任务文本、原始来源与哈希、双路视频帧数和完整解码，以及独立 URDF 正运动学复算的 state/action。通过官方 `LeRobotDataset` 读取每条 episode 的首、中、末帧，验证图像、标签与数值字段。 它还检查导出的 `meta/collection_metadata.json` 与源文件逐字节一致、采集指纹/schema 与 export manifest 相符，以及每条 episode 的指纹和基座/环境坐标未混用。

合并数据集的 LeRobot 审计会另行调用全量原始物理复算，并将其报告保存到原始目录及最终数据集的 `meta/physics_audit.json`。正式验收同时要求物理复算与 LeRobot 内容检查通过；采集时 metadata 的成功标记本身不能替代独立复算。保留完整原始目录及报告，以分别追溯实际物理行为与转换后的数据内容。

```python
import os
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
dataset = LeRobotDataset(
    repo_id="local/piper_elevator_1200",
    root=Path("datasets/piper_elevator_lerobot_v9"),
    video_backend="pyav",
)
sample = dataset[0]
print(dataset.num_episodes, len(dataset), sample["task"])
print(sample["observation.state"].shape, sample["action"].shape)
print(sample["observation.images.wrist"].shape)
```

官方读取器默认图像为 **CHW float32、范围 [0,1]**，而视频写入和采集帧为 HWC uint8。LeRobot v3 的一个视频文件可能包含多条 episode，应通过 `dataset.meta.get_video_file_path(episode_index, camera_key)` 和 episode 元数据定位，不能假定“一个 episode 对应一个最终视频文件”。

## 历史文件与验收状态

已完成的历史 10 Hz、schema 9 批次主要阅读路径如下。当前 30 Hz、schema 10 的独立路径及最终验收状态见 [30 Hz 数据集说明](dataset_30hz.md)。

| 路径 | 内容 |
| --- | --- |
| `datasets/collection_result_v9.json` | 1,200 条完整计数、各层数量、总采样/物理步、唯一轨迹数及 worker 完成记录 |
| `datasets/piper_elevator_raw_v9/collection_metadata.json` | schema 9 指纹、配置、唯一活动 Dome 策略、固定曝光、源代码/快照哈希及双路标定 |
| `datasets/piper_elevator_raw_v9/collection_run.json` | 本批并行命令、GPU 映射及日志位置 |
| `datasets/piper_elevator_raw_v9/worker_XX_lighting.json` | 各 worker 的活动/停用 Dome 和各环境局部灯记录 |
| `outputs/lighting_fix_validation/v9_full_initial_lighting.json` | 全量 2,400 个初始 RGB 帧的照明检查 |
| `outputs/dataset_visual_audit/v9_all_button_feedback.json` | 全量 1,200 条按钮反馈、窗口帧和可见性检查 |
| `outputs/dataset_visual_audit/v9_preflight/`、`v9_third_env/`、`v9_final_samples/` | 26 条完整几何同步抽查报告及联系图 |
| `datasets/piper_elevator_parts_v9` | 历史 schema 9 的 LeRobot 中间分片 |
| `datasets/piper_elevator_lerobot_v9/meta/audit.json` | 历史合并数据集的正式内容验收报告 |

下表是**保留的旧 schema 7 批次**及历史诊断路径，目前不推荐用于训练。历史统计不得算作新批次验收。

| 路径 | 内容 |
| --- | --- |
| `datasets/piper_elevator_raw/collection_metadata.json` | 场景来源、配置、任务文本、TCP 与动作语义、两路标定 |
| `datasets/piper_elevator_raw/render_settings_worker_XX.json` | 渲染历史控制设置前值、应用值及方法信息 |
| `logs/collection_<UTC时间>/run.json`、`worker_XX.log` | 单次并行运行清单、源码哈希和各进程日志 |
| `datasets/collection_run.json` | 保留的旧批次并行运行清单 |
| `episode_XXXXXX/metadata.json` | 任务、seed、微扰、实际成功判定、按压/释放事件、归位误差 |
| `episode_XXXXXX/frames.npz` | 10 Hz state/action、对应关节角、仿真时间、阶段与灯状态 |
| `episode_XXXXXX/physics.npz` | 120 Hz 物理记录 |
| `episode_XXXXXX/wrist.mp4`、`global.mp4` | 两路 RGB 原始视频 |
| `datasets/piper_elevator_raw/physics_audit.json` | 独立复算物理按压/释放/灯光/归位以及 state/action 采样对齐的报告 |
| `outputs/dataset_visual_audit/quality_summary.json`、`README.md` | 三组现有数据视觉抽查、两项保留失败、遮挡说明及独立 schema 8 诊断 |
| `datasets/piper_elevator_parts` | 旧批次中间 LeRobot 分块，不能混入修复后的新批次 |
| `datasets/piper_elevator_lerobot/meta/export_manifest.json` | 原始 episode 与最终数据的来源映射 |
| `datasets/piper_elevator_lerobot/meta/audit.json` | 最终独立审计报告 |
| `datasets/*_v2_render_delay` | 含约 100 ms 图像滞后的诊断归档，禁止用于训练 |

**旧 schema 7 历史核对（灯光错误，当前不推荐训练）：** 1,200 条、每楼层 100 条、216,766 个同步采样，8 个 worker 全部退出码为 0，无残留 `.inprogress`；所有记录使用同一采集指纹和 schema 7，全量原始物理审计通过。**图像抽查限制：** 26 条样本几何同步，后段 25、35 层固定视角颜色过渡检查失败，26 层固定视角受遮挡，原数据及阈值不变。**旧批次数值及格式内容验收通过：** LeRobot v3.0 共 1,200 条、216,766 个采样，双路 433,532 帧完成解码；独立 pose/来源复算和官方读取通过，`meta/audit.json` 的 errors 为空。上述视觉限制仍保留在旧数据的 `meta/visual_audit/`；独立 schema 8 诊断没有覆盖现有记录。

**历史 schema 9 结果：** 原始 12×100 采集完成，全量初始照明与按钮反馈检查通过，26 条几何同步抽查通过；该历史批次最终合并内容验收已通过，见上文 v9 报告；当前 30 Hz 的状态单独记录于 [dataset_30hz.md](dataset_30hz.md)。
