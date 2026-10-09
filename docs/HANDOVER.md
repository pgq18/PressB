# PressB / VLA-JEPA 工作交接

## 2026-10-09 仓库发布整理

- 主 README 已汇总三节点框架、8 个配置的完整评估、实际训练耗时、XYZ 平滑结论及 3 段保持 1× 时间的精选视频。小型统计在 `docs/results/`，媒体在 `docs/media/`，原始日志／轨迹／checkpoint 仍留在忽略的 `outputs/` 中。
- 本轮将此前未提交的 5090 适配、在线 RL、训练评估与审计代码统一纳入 PressB；下文历史“工作区未提交”提示是当时的状态，当前以 `git status` 为准。CPU 验证 835 项通过；另用隔离 USD 环境补跑 35 项，合计 870 项通过。
- VLA-JEPA 的 `base` 文档提交为 `2cd35ecbc5a5ce6c4074e3ca7e35a0bb6c656b1b`，已推送；**本机推理 checkout 有意保持 clean detached `9c9dc71`**，因为现有 RL 权重绑定该 commit、路径和源码哈希。发布前后完整推理身份逐值相同，H200 也保持原干净版本。不要把 detached 状态当作需要修复的异常，也不要直接切换最新 `base` 运行旧权重。
- ZPRL 参考仓库的新远端是 `https://github.com/pgq18/ZPRL-PressB.git`，已推送 `main` 文档提交 `4bd52f695bad813406a2327cdb2e448565c8abe9`；原算法来源仍为 `a34a1cb`，实际 Isaac RL 适配位于 PressB，不依赖运行 ZPRL。
- 两个独立克隆均从父仓库忽略，版本见 `repos.lock.json`。`/outputs` 的忽略规则同时覆盖目录和本机的数据盘符号链接，避免提交本机绝对路径链接。

## 2026-10-09 XYZ 动作后处理平滑实验

- 已冻结原 XYZ scale=0.02 的 400k checkpoint、噪声策略和 VLA，在 64 环境中完成 **1,320 次**评估，尝试位置 EMA 和位置＋姿态 SLERP 平滑；没有训练或修改模型权重。新增入口 `scripts/eval_smoothed_residual.py`，实现及复现见 [residual_on_frozen_noise.md](residual_on_frozen_noise.md)。
- **没有候选通过全部预设成功率门槛，默认保持 `none`，不自动启用后处理。** 门槛为每组至少 119/120、不低于同期基线，并用预先声明的新种子确认。原始 XYZ 自动评估的 119/120 仍是单次结果；本次无新增平滑基线跨种子为 116–119/120。
- XYZ alpha=0.8：筛选 119/120（同期基线 117），独立种子 117/120（基线 116）；实际共同成功轨迹的关节 jerk 中位数下降约 14%，末端位置 jerk 下降约 30%，时间变化约 +0.7%～+2.7%。平滑改善可复现，但未稳定达到绝对 119/120 门槛，仅保留为实验选项。
- 位置＋姿态 alpha=0.8：筛选 120/120，独立确认 117/120 < 基线 118，拒绝采用；更轻 XYZ alpha=0.9 为 118/120 < 基线 119，也拒绝。alpha=0.6 两种模式均为 117/120，未进入确认。保留所有失败记录，不将筛选最佳成绩当作独立验证结果。
- 证据根目录 `outputs/online_rl_xyz_smoothing_20261009`；初筛为根目录，姿态确认 `confirm/`，XYZ 确认 `confirm_xyz/`，轻度 XYZ 筛选 `conservative/`。各阶段 `analysis_screen/` 或 `analysis_confirm/` 含成功率、失败明细、实际运动指标、滤波重算和冻结审计；源码快照位于对应 `source_archive/`。
- 汇总为根目录 `smoothing_study_report.json`、`all_phase_metrics.csv`、`all_phase_per_button.csv`。**实验性对比视频**为 `videos_exploratory_xyz_a08/videos/xyz_vs_smoothed.mp4`：左无新增平滑、右 XYZ alpha=0.8，42.57 秒、30 fps、原速 1×；来自 seed=20261010 两组实际 120 次评估的首轮中心任务，两侧视频均 12/12，不代表完整评估通过。轨迹、渲染、合成 QA 全通过，GPU 服务已清理。
- `confirm_xyz/plan.json` 的旧描述字段 `confirmation_seed` 残留 20261009，实际 `plan.seed`、两个运行 manifest 及采样均为 **20261010**，说明见 `metadata_clarification.json`；后续脚本已修复该元数据字段。不同方法异步重置导致部分后续样本初态有微小数值差异；初态完全相同子集也复现了平滑改善，不宣称全部 120 对逐位相同。

## 2026-10-08 XYZ 残差 400k 续接入口

- **2026-10-09 结果更新**：本轮训练和自动评估于北京时间 06:14 完成，400,000 条训练 transition、398,000 次更新；120 条固定条件 eval 为 **119/120（99.17%）**。24 键 9/10，其余 25–35 各 10/10；唯一失败为偏移 `(−0.01,−0.025)m` 第 2 次重复的 24→25 误按。已逐条核对仿真原始终止与 reset 记录，统计及审计见运行目录 `eval/per_button_success.csv`、`eval/failed_episodes.csv`、`eval/audit_20261009.json`。训练服务已自动清理。以下启动说明保留用于复现。
- **执行视频已完成**：`outputs/online_rl_residual_on_noise_gamma0995_xyz002_videos_20261009/videos/pose9_vs_xyz.mp4`，42 秒，左侧原 scale 的 9D 残差、右侧本次 XYZ 残差，均冻结同一噪声策略。1× 仿真时间同步，包含全局和腕部视角。另录的 12 个中心布局任务为左 12/12、右 11/12（24→25），与原 120 次固定评估不同；轨迹、渲染、合成三阶段 QA 全通过，录像服务已清理。
- 用户要求冻结原来训练完成的初始噪声策略，仅对每步 XYZ 学习动作残差，scale 为 `[0.02,0.02,0.02]`；沿用 γ=0.995，从头训练 400k，不恢复旧残差。
- 本轮于 2026-10-08 22:45（北京时间）启动，目录为 `outputs/online_rl_residual_on_noise_gamma0995_xyz002_400k_20261008`；实时状态读取该目录 `queue_status.json`、`train/status.json`，不要把本段启动记录当作最新状态。
- `outputs/online_rl_current.json` 已指向本轮。GPU0 训练，GPU1 运行 64 环境仿真和 batch64 冻结 VLA 推理。监督程序自动完成 400k 后的 120 条固定条件评估并清理自建服务。
- [W&B 实时日志](https://wandb.ai/penggq2025-southern-university-of-science-technology/pressb-online-rl/runs/d154fb90) 包含 recent SR、每按键近期成功率、损失及采样速度。训练 SR 与固定条件评估分开。
- 核心新增 `LearnerConfig.residual_mode="xyz"`：actor/replay 真正为 21 维（7×3），actor 仍以完整 63 维基础动作作为条件，critic 使用完整合成 pose9；旋转不加残差，执行时四元数和夹爪沿用解码器原值。默认目标熵为 −10.5；旧 checkpoint 缺省 `pose9` 保持兼容。
- 初始噪声仍为 `outputs/online_rl_fast_20261003/initial_noise_train/last.pt`，noise_scale=1.5；VLA 仍为冻结 step10600。初始化和冻结检查见本轮 `initial.pt`、`composition.json`（均在 `train/`）及 `startup_verification.json`；配置、源码快照、运行指令均留存在运行目录。
- 本轮启动前 338 项相关 CPU 测试通过。详细方法与命令见 [residual_on_frozen_noise.md](residual_on_frozen_noise.md)。历史 γ=0.995 原 scale 的 120 次 eval 为 118/120；半 scale 为 87/120，不能将中心布局演示视频 12/12、9/12 当作完整 eval。

## 2026-10-03 在线 RL 续接入口

本节覆盖下文 2026-10-02 原始交接中的“无运行任务/仅 H200 推理”状态；下文保留原评估和部署历史。

- 用户已授权优化并运行长程在线 RL，动作残差与初始噪声两种方法各 1M transition。三个 HTTP 节点保留跨机器能力，目前部署改为本机：GPU0 SAC，GPU1 Isaac 并行仿真与冻结 VLA-JEPA。
- 当前运行入口始终读取 `outputs/online_rl_current.json`，再读所指 `queue_status.json`；不要依据本文静态 PID 或计数启动重复服务。完整设计与命令见 [online_rl_fast.md](online_rl_fast.md)。
- VLA-JEPA 已在 H200 创建提交 `9c9dc71c8e199c7428b25d103ab666395832c001`，H200 DNS 无法推送，由本机中继成功 push。独立本地克隆为 `src/VLA-JEPA`，独立推理环境 `.conda/envs/vlajepa-inference`；原 Isaac 环境保留。冻结 step10600 权重位于 `.cache/vlajepa/`，SHA 验证和跨 GPU 数值对照见 `outputs/online_rl_local_inference_20261002/`。
- 已实现 64 环境独立结束/重置、批量受限 IK、PhysX 视图、chunk 边界渲染、批量视觉/flow 推理、无损 packed 特征、SAC 与 RPC 等待重叠。一个 transition 仍是一个环境的最多 7 动作 chunk，不能称 episode/s。warmup 后 UTD=1。
- GPU 物理缓冲溢出曾导致不可信速度，已显式配置容量并加入同步原生日志错误检查；真实溢出负例退出 1，正常配置 62/62 专家接触、错按和无接触负例通过。证据见 `outputs/online_rl_fast_probe_20261003/`。
- `outputs/online_rl_fast_20261002` 候选因原生 224 正方形渲染改变垂直视野而停止，120 条 base 仅 2 成功；此候选不能当作任务有效加速。修正版 `outputs/online_rl_fast_20261003` 保持 320×240 的 4:3 相机投影，再缩放为模型的 224×224。运行状态和最新基线以该目录实测为准。
- 旧 12 环境训练目录 `outputs/online_rl_long_20261002_parallel` 在 10207 transition/8001 update 停止并保存。新 collector 仅迁移 actor，critic、replay 和计数重新初始化；不把两套仿真约定的 replay 混合。
- PressB 工作区仍含此前未提交改动，本轮只按用户要求提交和推送 VLA-JEPA 仓库。不要清理工作区或删除历史证据。

更新日期：**2026-10-02（Asia/Shanghai）**。本文面向直接运行在 **5090 工作站**上的新 agent。用户要求在该机器继续接手机械臂按电梯按钮的仿真、数据采集、训练与评估工作。

**5090 项目根目录：`/home/pengguanqi/Workspace/Research/PressB`。** 除明确写出机器名和绝对路径外，本文中的相对路径均以此目录为根。这是交接时的事实快照；GPU、进程、SSH socket 和 Git 工作区状态可能随后变化。

## 1. 当前结论与接手入口

1. 5090 的仿真环境已经部署并完成实际运行验收，支持黑色 PiPER、官方腕部 D435、桌面 D435、真实按钮接触及灯光反馈。
2. 两张 5090 均验证可用。完整专家轨迹演示已成功按下全部 12 个按钮。不要把专家规划演示的 12/12 与模型自主闭环成功率混为一谈。
3. 带面板位置偏移的 1,200 条训练数据已经在原采集机完成，用户已传到 H200。5090 目前只有两批小规模采集预检数据，**没有在 5090 重新采集完整 1,200 条，也没有安装独立 LeRobot 导出环境**。
4. H200 上 VLA-JEPA 已适配 LeRobot v3 和 PiPER 位姿接口。带偏移数据从官方预训练重新微调的运行已完成 **3 epoch / 12,942 step**。当前 eval 选用该运行中验证最佳的 **step_010600**；不是最终 step_012942，也不是旧固定面板权重。
5. 已完成 **5090 仿真 → SSH → H200 推理**的 60 条闭环评估，以及与原采集机的同条件对照。5090 成功 **20/60**，原采集机历史基线 **5/60**；两套仿真结果不完全一致，详细结论见第 6 节。
6. 交接核查时，5090 无 GPU 计算任务；H200 四卡显存均为 0 MiB。本任务四个临时推理服务已停止，SSH 主连接仍可复用。没有等待本 agent 继续盯守的训练或采集任务。
7. **5090 工作区有重要未提交改动。** GitHub 的旧提交不包含本机全部 5090 适配和跨机器 eval 改动。先看 `git status`，保留工作区，见第 8 节。

新 agent 优先阅读：

- 本文；`docs/rtx5090_setup.md`：5090 安装、运行与验收。
- `docs/policy_eval.md`：推理接口、动作语义、闭环执行与审计。
- `outputs/rtx5090_eval_comparison/results/verification.json`、`outputs/rtx5090_eval_comparison/results/summary.md`：本次对比结果。
- `docs/dataset_panel_stratified.md`：带偏移数据集及验收边界。
- `logs/handover_20261002/`：本次交接附带的机器状态、训练证据和历史基线摘要。

## 2. 用户已确定的场景与数据要求

### 2.1 场景

- 黑色 PiPER 放在桌沿，面向墙壁上的电梯面板。
- 面板 2 列 × 6 行：左列由下到上 24–29，右列由下到上 30–35。
- 初始姿态为大臂、小臂折叠于桌面上方，与桌面平行、垂直于桌沿。每次演示/采集结束后归位，再开始下一条。
- 夹爪默认保持夹住按压杆的姿态；配置为两个关节 `+0.004/-0.004 m`，总开度 **0.008 m**。这是持杆状态，不应改成几何零开度。
- 按下时仅按钮**边缘**亮橙色，数字保持清楚；释放回弹后熄灭，不锁存灯光。
- 腕部 D435 使用官方开源适配资产中的原支架和相机几何，不重新设计替代支架。
- 另一台 D435 在机械臂左侧桌面上的矮支架，靠近机械臂。初始视角完整覆盖面板；机械臂不必全部入镜，操作时允许实际遮挡。
- 已使用官方/现有 Isaac 素材搭建桌子等物体；固定来源清单和 SHA 校验见 `assets/sources.lock.json`、`THIRD_PARTY.md`、`scripts/fetch_assets.py`。

### 2.2 面板偏移

基于名义面板位置，**X 表示前后方向，Y 表示左右方向**；高度不随机。

| 方向 | 已使用范围 |
|---|---:|
| 前后 X | −0.010 ～ +0.010 m |
| 左右 Y | −0.025 ～ +0.025 m |

采集配置为 `configs/dataset_panel_stratified_1200.json`：每层 10×10 分层格，各格一条，四个角格取精确角点，其余格内小幅随机采样；每层独立打乱，seed **20260930**。轨迹和时序也有小幅变化。每个候选位置均检查 12 个按钮可达性、关节/碰撞约束和桌面相机面板投影。

### 2.3 数据接口，不可随意改动

- 12 个 task 文本严格为 `Press 24 floor.` … `Press 35 floor.`，每层 100 条，共 1,200 条。
- 最新数据为 **30 Hz**；物理仿真 **120 Hz**；相机每路 **640×480 RGB**。采集器也支持整除 120 的频率，如 10/20/30/60 Hz，但每种频率须单独配置并保存在独立数据目录。
- 训练 episode 从折叠初始状态到目标按键首次按亮的采样帧，**包含成功画面，不包含撤回和归位**。完整原始诊断轨迹包含撤回，导出器裁剪成训练前缀。
- LeRobot 中 state/action 为 8 维 `[x,y,z,qw,qx,qy,qz,gripper_width]`。坐标系是机械臂 **base_link**，位置单位米。state 为当前实测位姿；action 为下一采样时刻的**绝对目标**，不是位姿增量。终止帧使用既有前缀导出的末端 action 规则，不能补撤回动作。
- 学习的 TCP 是 `link6` 局部 Z=**0.1358 m** 的夹爪 TCP；按压杆尖端是局部 Z=**0.24 m**。二者不能混用。
- 模型训练/推理内部为 9 维 `XYZ + rotation6D`：旋转矩阵**前两行**按行展开 `[r00,r01,r02,r10,r11,r12]`。XYZ 保留原值，不归一化、不改坐标系；夹爪不学习。
- 模型双相机顺序为 **global,wrist**，服务将原始 RGB 双线性缩放至 224×224。每次预测 7 个绝对目标，diffusion inference timesteps=4。

## 3. 三台机器、代码与环境

| 机器 | 用途 | 关键位置 |
|---|---|---|
| 5090 工作站 | 当前主工作区、仿真、双相机渲染与闭环控制 | `/home/pengguanqi/Workspace/Research/PressB` |
| H200，5090 上可直接 `ssh h200` | 训练、模型权重、完整训练数据、远程推理 | `/home/pengguanqi/Worksapce/Research/VLA-JEPA` |
| 原采集机 | 历史 Sim 4.5 数据和本机 eval 基线 | `/data/pgq/Workspace/Research/PressB` |

**H200 路径中的 `Worksapce` 是实际拼写，不能自动纠正成 `Workspace`。** 新 agent 在 5090 上，不要把原采集机 `/data/pgq/...` 当成本地可访问目录。

5090 仓库远程：`https://github.com/pgq18/PressB.git`，交接时 HEAD 为 `5fac67e`。H200 仓库远程：`https://github.com/pgq18/VLA-JEPA.git`，HEAD 为 `d90dec8`，交接核查时工作区干净。H200 的 PiPER 适配已经 commit/push；5090 这轮适配尚未 commit/push。

### 3.1 5090 软件栈

Ubuntu 24.04.4 LTS，驱动 580.173.02，两张 RTX 5090，每卡约 32 GB。项目 Python 直接使用：

```bash
cd /home/pengguanqi/Workspace/Research/PressB
.conda/envs/pressb/bin/python --version
```

| 组件 | 已验收版本 |
|---|---|
| Python | 3.11.16 |
| Isaac Sim | 5.0.0，包元数据为 5.0.0.0 |
| Isaac Lab | Git tag v2.2.0，SHA `46dff135f44683f031edf346e544fcfd8456b2bb` |
| Isaac Lab 核心 Python 包 | 0.44.9；与上述源码发布标签并不矛盾 |
| PyTorch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 |
| NumPy / SciPy / Pillow | 1.26.0 / 1.15.3 / 11.2.1 |
| Warp / PyAV / Gymnasium | 1.7.1 / 15.1.0 / 1.2.0 |

PressB 以 editable 安装。完整依赖见 `logs/install-5090-requirements.txt`，`pip check` 已通过。Conda 程序为 `/home/pengguanqi/miniconda3/condabin/conda`。

- 需要重建时使用 `scripts/setup_env_5090.sh` 和 `configs/constraints-5090.txt`。
- **不要对现有环境运行旧 `scripts/setup_env.sh`**：它固定 Python 3.10 / Sim 4.5 / Lab 2.0.2。
- `.cache/usd-inspect` 是隔离的 usd-core 24.11，用于素材处理和独立 USD 审计。只给这些工具单独设置 `PYTHONPATH`，不要全局导出或覆盖 Kit 内置 USD。正式运行报告的 Kit USD `[0,24,5]` 是正常情况。
- `.conda/envs/lerobot` 尚未在 5090 安装。后续需要导出时使用 `scripts/setup_dataset_env.sh` 建独立环境，不把 `requirements-dataset.txt` 装进仿真环境。
- 5090 的有效设备编号是 **0/1**。旧入口有默认 GPU=4，`collect_parallel.py` 默认多卡列表也不适合本机；所有启动命令显式指定 GPU。

### 3.2 H200 环境与代码入口

H200 有四张 H200 NVL，每卡约 140 GiB 显存。Python 为 `/home/pengguanqi/miniconda3/envs/vlajepa-piper/bin/python`；此前实际推理环境为 Torch 2.6.0+cu124。

| 功能 | H200 仓库内路径 |
|---|---|
| LeRobot/PiPER 加载与数据窗口 | `starVLA/dataloader/piper_lerobot.py` |
| PiPER 训练器 | `starVLA/training/train_piper.py` |
| 训练入口/默认配置 | `scripts/run_piper_ft.sh`、`scripts/configs/vlajepa_piper_ft.yaml` |
| 离线 checkpoint 评估 | `scripts/eval_piper_checkpoint.py` |
| 推理服务/模型包装 | `scripts/serve_piper_policy.py`、`starVLA/inference/piper_policy.py` |
| 训练、续训及接口文档 | `docs/piper_lerobot.md`、`docs/piper_continuation.md`、`docs/piper_status.md` |

SSH 首次连接可能很慢，用户明确要求 **ConnectTimeout=120** 并尽量复用连接。5090→H200 现有 socket 是 `/tmp/pressb-5090-h200-eval.sock`。不要在 5090 上使用原采集机→5090 的 `/tmp/pressb-5090-setup.sock`。

## 4. 数据集与训练已完成的范围

### 4.1 当前主数据集

H200：`/home/pengguanqi/Datasets/piper_elevator_lerobot_panel_stratified_press_30hz`。某些 home 路径实际指向 scratch，保留已有链接和布局。

- LeRobot v3.0；官方 LeRobot 0.6.1 加载器已验证。
- 1,200 episodes、12 tasks、306,830 帧，双路共 613,660 张图像，30 Hz。
- 原采集机最终整包大小记录为 7,704,437,651 字节，约 7.70 GB。
- 完整原始轨迹为 648,454 帧；首次亮灯裁剪移除后续 341,624 帧。
- 每层 100 格各一条，四个角点齐全；物理、位置覆盖、整包解码、训练前缀及官方读取通过。
- 相机视锥覆盖不等于操作中完全无遮挡。腕部 1,200 条末帧均提供清晰亮灯证据；桌面有 109 条终点遮挡/像素不足，由腕部与物理记录确认。
- 完整原始 episode 776 的桌面橙色阈值诊断有一帧迟滞，原失败报告仍保留；腕部正确、几何同步正确，训练前缀没有延长或改阈值。详见 `docs/dataset_panel_stratified.md`，不要将其描述成所有全局视角严格首亮检查都通过。

历史固定面板数据还在 H200：`/home/pengguanqi/Datasets/piper_elevator_lerobot_press_30hz`，同为 1,200 条，但为 306,230 帧。两者不要混用。

5090 `datasets/` 中当前只有 `5090_smoke_30hz/` 和 `5090_smoke_30hz_v2/`。第一批保留已定位的灯光残影，第二批才是验收通过版本。eval 输入目录中的 `dataset_metadata/` 只是原训练集元信息副本，不是完整训练数据集。

### 4.2 带偏移数据微调

H200 运行目录：

```text
/data/scratch/pengguanqi/VLA-JEPA-runs/piper_panel_stratified_press30hz_pose9_3epochs_20260929
```

从官方预训练重新开始，`initialization.json` 为 `step=0, continuation=false`。官方文件通过 `/home/pengguanqi/Models/VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt` 访问，实际文件在 scratch；已记录 SHA `fd929c79d9bbd0bda56c0b952c7acb470d93c6241a519013fe5248c3f3ea5fab`。Qwen 与 V-JEPA2 基础资源在 `/home/pengguanqi/Models/Qwen3-VL-2B-Instruct`、`/home/pengguanqi/Models/vjepa2-vitl-fpc64-256`，可离线加载。

- 按完整 episode 分层划分，seed=42，每层 90 训练/10 验证：**1,080 训练、120 验证**，无重叠，总体覆盖 1,200 条。
- 训练帧/锚点 276,063；验证帧 30,767。不能说 1,200 条全部参与梯度更新。
- 4 卡，每卡 batch=16，全局 batch=64，梯度累积 1，sample_stride=1；每 epoch 4,314 步，分布式每 epoch 补齐 1 个重复样本。
- 完成 **3 epoch / 12,942 步**，`completed.json` 为 `success=true, epoch=3, batch_in_epoch=0`。
- 北京时间 2026-09-30 04:59:01–10:14:03，监督进程总耗时约 **5 小时 15 分 2 秒**，正常退出；训练内部计时为 18,766.898 秒，两种口径不同。
- 每 200 步验证；每条验证 episode 固定 5 个锚点，共 600 个。最佳按首动作位置误差（米）+0.1×旋转误差（弧度）选择。

| 检查点 | 定位 | 离线首动作位置误差 | 离线首动作旋转误差 |
|---|---|---:|---:|
| `checkpoints/step_010600` | 3 epoch 运行中验证最佳，当前闭环使用 | 6.135 mm | 0.330° |
| `checkpoints/step_012942` | 最后一步，完整 3 epoch | 7.779 mm | 0.305° |

模型 `model.pt` SHA256：

```text
step_010600: 2456b1fff5ef2d94a173b502d55b244a6b0810b92c6e39fc6b1527e1d6019312
step_012942: ec4d9a889ff4123e456342efd72c7ee38317e99e4f590e73865d5df08f56621d
```

最佳、最终的 model.pt 和 training.pt 共四文件已做远端 SHA 复核。`checkpoints/best` 和 `last` 分别解析到上述目录。离线误差不是按键成功率；现有跨机器 60 条闭环结果属于 **10600**，不能归到 12942。

旧固定面板运行 `piper_press30hz_pose9_20260929` 的 step2000、`piper_press30hz_pose9_3epochs_from2000_20260929` 的 step12800 是另一条历史链路。当前 `eval_policy.py` 默认权重检查仍指向旧 step2000，必须显式传入本轮 step 与 SHA。

## 5. 5090 上已经验收的仿真

- 35/35 下载资产 SHA 通过，官方腕部支架/D435 来源审计通过。
- 两卡实际 PyTorch CUDA、矩阵乘法及 Warp kernel 通过。
- `outputs/edge_feedback/`：GPU 0 完整专家演示，23,206 个物理步，12 按钮全部按下、回弹、归位，无意外碰撞；两相机各 1,934 帧有效 RGB-D。该演示默认拍照 10 Hz，与数据采集和 eval 的 30 Hz 区分。
- `datasets/5090_smoke_30hz_v2/`：GPU 1 两个并行环境、两条 30 Hz 原始轨迹，517/512 帧；几何同步零帧延迟，首亮与熄灭同步，灭灯后首帧橙色像素为 0。
- Isaac Lab AppLauncher / SimulationContext 在 GPU 1 reset、步进 10 次、正常退出。
- 部署验收时全套测试 **414 passed**；跨机器 eval 改动后，5090 新增资产重定位/结果对比的针对性测试 **36 passed**。不要把这两次测试说成当前全套 450 项已统一重跑。

主证据：`logs/setup-5090-summary.json`、`logs/verification-5090-pytest-final.log`、`outputs/edge_feedback/audit_summary.json`、`outputs/5090_smoke_feedback_v2.json`、`outputs/5090_smoke_camera_sync_v2/report.json`。

运行视频：`outputs/edge_feedback/episode.mp4`；两路为 `wrist_camera/rgb.mp4`、`global_camera/rgb.mp4`。GUI 要在工作站桌面会话启动，普通 SSH 无桌面显示。

## 6. 最新闭环 eval：配置、结论与证据

### 6.1 两边用的是什么

两边都使用第 4 节**带偏移训练**的同一 step10600，权重 SHA 完全一致；都使用该数据集冻结的场景、配置、任务与种子。

eval 模式是 **`center_corners`**：中心 `(0,0)` 与四角 `(±0.01,±0.025) m`，每个位置测全部 12 层，每条件一次，即中心 12 条、偏移 48 条。为便于一一比较，位置固定成五点，并非每次重新随机抽取。两边 60/60 均有 `measured_layout_verified=true`，实际移动过面板。

| worker | 楼层 | 5090 仿真 GPU | H200 推理 GPU / loopback 端口 |
|---|---|---:|---|
| 0 | 24,25,26 | 0 | 0 / 19765 |
| 1 | 27,28,29 | 0 | 1 / 19766 |
| 2 | 30,31,32 | 1 | 2 / 19767 |
| 3 | 33,34,35 | 1 | 3 / 19768 |

每 worker 三个环境、15 条；`seed=20260930`、`max_seconds=15`、`smoothing_window=3`、动作 30 Hz、物理 120 Hz。每请求 7 步动作，在四个物理子步上线性插值，再做三点因果关节均值滤波；历史跨 chunk 保留，每 episode 重置，名义延迟 8.33 ms。`window=1` 才是关闭均值滤波。

模型只接收实时双相机、实测基座 TCP 和 task 文本；没有输入目标按钮坐标、面板偏移或录制专家动作。IK 是受关节位置/速度约束的连续单初值全位姿求解，原模型目标、可执行目标和跟踪残差分别保留。

成功要求目标按钮位移至少 **1.5 mm**、按压杆接触力 **>0.02 N**，无误按、无异常碰撞。首次按钮按下或异常/超时即终止，不执行撤回。首次按错是失败。运行异常不是模型超时。

等待 H200 响应时仿真时钟暂停；它测闭环策略能力，**不是 30 Hz 实时墙钟控制性能**。

### 6.2 结果

| 结果 | 原采集机 / Sim 4.5 | 5090 / Sim 5.0 |
|---|---:|---:|
| 成功 | 5/60（8.3%） | 20/60（33.3%） |
| 误按 | 27 | 22 |
| 超时 | 28 | 18 |
| 异常碰撞 | 0 | 0 |
| 无效策略动作 | 0 | 0 |

34/60 条终止原因及实际按下楼层一致；原 5 条成功全部保留，新增 15 条成功。两边完整运行通过独立审计。

四组历史无损图像、状态和 seed 经 5090→H200 重放，pose9 与 pose8 动作逐值完全一致，最大差为 0。实时闭环的初始 TCP 差异极小（最大位置约 2.84×10⁻⁹ m），但全局/腕部图像平均像素 MAE 为 3.43/5.79（0–255），首个预测 chunk 的 XYZ RMSE 平均约 1.22 mm。

**结论边界**：远程推理链路可用，四组相同输入的输出一致；跨版本完整闭环并未逐条复现。本次 5090 成功更多，不能据此称“5090 硬件提升了模型能力”。Sim 版本、渲染硬件、灯光稳定刷新次数等同时不同，目前没有消融实验单独区分渲染与物理影响；60 个条件各测一次也不是泛化成功率的充分统计。

### 6.3 5090 上的现成证据

以 `outputs/rtx5090_eval_comparison/` 为根：

| 路径 | 内容 |
|---|---|
| `input/scene.usda`、`input/config.json`、`input/dataset_metadata/meta/collection_metadata.json` | 原始冻结输入 |
| `input/assets/` | 完整且已验证的资产包 |
| `input/golden_requests/worker_0..3/` | 四组相同输入复测材料 |
| `center_corners_v1/worker_0..3/` | 正式 60 条完整 eval，含所有 chunk 无损观测、双视频、动作及物理记录 |
| `center_corners_v1/aggregate_report.json`、`center_corners_v1/summary.md`、`center_corners_v1/episodes.csv` | 四 worker 汇总 |
| `results/comparison.json`、`results/summary.md`、`results/verification.json` | 与原采集机的配对比较 |
| `golden_replay.json` | 固定输入逐值对照 |
| `services.json`、`formal_processes.json`、`service_cleanup.json` | 当时的服务/启动/清理证据，PID 不代表现在仍有效 |
| `results/compare_floor33_center.mp4` | 33 层中心任务四宫格，左旧机误按 34、右 5090 按中 33 |
| `results/compare_floor33_center_video.json` | 来源哈希及视频制作说明 |
| `pilot/` | 前置 1 秒链路检查，其超时不计入正式 60 条 |

每 worker 的 `audit.json` 为独立审计，`diagnosis.json` 为模型/IK/跟踪分解，`eval_manifest.json` 为配置与源码身份。`audit_pass` 只代表证据有效，不表示任务成功。MP4 使用 30 Hz CFR，精确终止物理时刻看 `frames.npz`，不能只按视频时长推断。

对照视频右列在 6.517 秒成功终止后定格，左列在 10.742 秒误按；已明确标注定格，不能解释为右侧继续执行。

原采集机的完整历史 baseline 不在 5090；它仍位于 `/data/pgq/Workspace/Research/PressB/outputs/policy_eval_step10600_stratified/center_corners_v1`。本次交接复制了摘要到 `logs/handover_20261002/baseline_evidence/`，配对比较 JSON 也已在 5090，若需重算所有原图指标再取回原始历史记录。对比 JSON 中一些输入路径属于原采集机，不能直接作为 5090 本地路径。

原采集机取回的 `eval_evidence.tar` 只含每条首个 chunk PNG，加全量视频/轨迹；**全部中间观测 PNG 在 5090 原目录**。独立审计最好在 5090 原位置运行，重定位报告引用绝对路径；不要删除工作站上的原始观测。

## 7. 必须保留的兼容修复

1. **Dome 全局光**：并行环境只保留一个有效 Dome，额外实例 `SetActive(False)`。仅设置 intensity=0 曾造成采集图像显著变暗，不能回退。局部 RectLight 另行处理，`lighting.json` 和有效 Dome 数量检查保留。
2. **灯光残影**：5090 `scripts/collect_dataset.py` 的 `LIGHT_SETTLE_CAPTURES=16`，原采集机为 4。灯光切换后做不推进物理时间的刷新，已实测解决灭灯余色。保留图像/状态不推进断言与原验收阈值；更改渲染策略会改变采集 fingerprint，不能续写旧目录。
3. **相机字段**：Sim 5.0 的 `rendering_frame` 可能是 Fabric 分子/分母字典；5090 记录器保留字典和类型说明，也兼容旧整数。`rendering_time` 仍为 API 仿真秒数。不要把分子当帧号，也不要把两个字段写反。
4. **冻结场景资产重定位**：原 snapshot 有旧机器绝对路径。传 `--asset-bundle`，由 `src/pressb/scene_portability.py` 生成本次输出的 `runtime_scene.usda`；不得重建一个“看起来相同”的 scene 并伪造旧 SHA。
5. **资产包闭包**：10 文件包括 PiPER 主 USD+三个配置 USD、桌子/支架、纹理、官方腕部兼容 USD 和两张生成标签。七个唯一绝对资产路径被映射到包内；源文件字节以 SHA 校验，运行副本经逆映射验证 authored USD 内容一致，`OmniPBR.mdl` 保留引擎资源名。
6. **审计身份**：原场景 SHA `d5f5556825db36635dd1636561bc0caa75bdd156d93b6e254922239450e9741a`；collection fingerprint `6ab95889499aa7e4735b1d94d90358c18439814217e43108308fa007035f08a4`。`scene_relocation.json` 及其自身 SHA 写入 manifest；审计重新检查资产和逆映射，并比较完整报告。不要改 collection 元信息绕过检查。
7. manifest 中 `renderer_settings` 是代码请求设置，不是运行时逐项读回，不能单凭它证明跨引擎渲染语义相同。

## 8. 未提交改动与代码地图

交接前核对到的 5090 改动如下；本文及 README 交接入口也会成为本次未提交文档改动。以 `git status --short` 当前输出为准。

| 文件/组 | 目的 |
|---|---|
| `README.md`、`pyproject.toml` | 5090 入口；Python 3.10–3.11 支持 |
| `configs/constraints-5090.txt`、`scripts/setup_env_5090.sh` | Sim 5.0/Lab 2.2 依赖组合 |
| `scripts/verify_gpu_5090.py`、`scripts/verify_simulation_5090.sh`、`docs/rtx5090_setup.md` | 硬件/仿真验收与部署文档 |
| `scripts/collect_dataset.py` | 16 次灯光稳定刷新 |
| `src/pressb/camera_recording.py`、`tests/test_camera_recording_metadata.py` | Sim 5.0 相机元数据兼容 |
| `tests/test_dataset_scene.py`、`tests/test_motion_smoothing.py` | USD 双精度偏移及浮点断言适配 |
| `scripts/eval_policy.py`、`scripts/audit_policy_eval.py`、`src/pressb/scene_portability.py`、`tests/test_scene_portability.py` | 冻结资产跨机器重定位、证据与独立审计 |
| `scripts/compare_policy_eval_runs.py`、`tests/test_compare_policy_eval_runs.py`、`docs/policy_eval.md` | 条件配对、跨机器对照与使用说明 |

**不要 `git reset --hard` / `git clean` 清掉工作区；不要从原采集机整目录覆盖。** 原机仍有 Sim 4.5 的相机与 4 次灯光刷新代码。若后续需要提交，先区分兼容性改动与实验产物，针对性检查后提交。数据、权重、环境、素材、输出、日志都在 `.gitignore` 中；本次交接未执行新的 commit/push。

其他主要模块：`src/pressb/scene.py` 搭场景，`dataset_scene.py` 并行场景/灯光/面板布局，`dataset_planning.py` 专家采集规划，`press_prefix.py` 首次亮灯裁剪，`replay_control.py` 位姿 IK，`motion_smoothing.py` 平滑，`policy_layouts.py` 五位置评估调度，`policy_eval.py` 远程接口/动作检查。

## 9. 常用复现命令

以下是操作参考，不是要求新 agent 接手后自动全部重跑。已有结果已经完成。所有新运行使用未占用的新目录，保留原始证据。

### 9.1 接手检查与专家演示

```bash
cd /home/pengguanqi/Workspace/Research/PressB
git status --short
nvidia-smi
.conda/envs/pressb/bin/python scripts/fetch_assets.py --check-only

# 只在需要新演示时执行；此目录必须未使用。
bash scripts/run.sh --headless --gpu 0 --video --output outputs/handover_demo_new
.conda/envs/pressb/bin/python scripts/audit_episode.py outputs/handover_demo_new
.conda/envs/pressb/bin/python scripts/audit_global_camera.py outputs/handover_demo_new
.conda/envs/pressb/bin/python scripts/check_results.py outputs/handover_demo_new
```

完整演示判据为 `PASS: 12/12 physically pressed in order` 及独立审计成功，不能截断几步就当完整验收。GUI 示例见 `docs/rtx5090_setup.md`。

### 9.2 复用 5090→H200 SSH，并启动一个推理服务

在 5090 检查已有主连接：

```bash
ssh -S /tmp/pressb-5090-h200-eval.sock -o ConnectTimeout=120 -O check h200
```

仅当没有活动主连接时，在单独终端建立并保持该终端：

```bash
ssh -M -S /tmp/pressb-5090-h200-eval.sock -o ConnectTimeout=120 \
  -o ControlPersist=no -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N h200
```

另一个 5090 终端登录 H200：

```bash
ssh -S /tmp/pressb-5090-h200-eval.sock -o ConnectTimeout=120 h200
```

以下在 **H200 shell** 执行。先看显存，GPU 编号按当时余量选择。服务前台运行，保留这个终端或按既有监督进程方式管理；不要使用旧 PID 当作新服务 PID。

```bash
cd /home/pengguanqi/Worksapce/Research/VLA-JEPA
nvidia-smi
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/pengguanqi/miniconda3/envs/vlajepa-piper/bin/python \
  -u -m scripts.serve_piper_policy \
  --checkpoint /data/scratch/pengguanqi/VLA-JEPA-runs/piper_panel_stratified_press30hz_pose9_3epochs_20260929/checkpoints/step_010600 \
  --port 19765 --device cuda:0
```

回到 **5090**，为尚未存在的转发增加映射；已有同端口转发不要重复添加：

```bash
ssh -S /tmp/pressb-5090-h200-eval.sock -o ConnectTimeout=120 \
  -O forward -L 127.0.0.1:19765:127.0.0.1:19765 h200
curl --fail http://127.0.0.1:19765/health
```

确认 `ready`、`checkpoint_verified=true`、step=10600 和上述 SHA。模型首次加载此前约 90 秒，HTTP 端口不应公开监听。每个服务此前约占 10.5 GiB 显存。其他三服务用 GPU 1/2/3 和端口 19766/19767/19768。任务结束只停止自己创建且 argv 匹配的服务，不要广泛 `pkill python`。

### 9.3 5090 一组正式闭环（15 条）

```bash
cd /home/pengguanqi/Workspace/Research/PressB
bash scripts/eval_policy.sh \
  --endpoint http://127.0.0.1:19765/predict \
  --dataset outputs/rtx5090_eval_comparison/input/dataset_metadata \
  --config outputs/rtx5090_eval_comparison/input/config.json \
  --snapshot outputs/rtx5090_eval_comparison/input/scene.usda \
  --asset-bundle outputs/rtx5090_eval_comparison/input/assets \
  --expected-checkpoint-step 10600 \
  --expected-checkpoint-sha256 2456b1fff5ef2d94a173b502d55b244a6b0810b92c6e39fc6b1527e1d6019312 \
  --floors 24,25,26 --panel-layouts center_corners --episodes-per-floor 1 \
  --num-envs 3 --gpu 0 --max-seconds 15 --seed 20260930 \
  --smoothing-window 3 --timeout 120 \
  --output outputs/handover_eval_new/worker_0
```

链路短检查可另设新目录、单层、单环境和 `--max-seconds 1`，通常超时是预期，不能算正式成功率。完整复现按第 6 节四组执行；四并发之前看主机内存和显存，上次可成功运行，内存占用较高。

每 worker 完成后，在 **5090 原位置**审计、诊断、出报告：

```bash
PYTHONPATH=.cache/usd-inspect:src .conda/envs/pressb/bin/python \
  scripts/audit_policy_eval.py --run outputs/handover_eval_new/worker_0 \
  --allow-partial --report outputs/handover_eval_new/worker_0/audit.json
.conda/envs/pressb/bin/python scripts/diagnose_policy_eval.py \
  --run outputs/handover_eval_new/worker_0 \
  --output outputs/handover_eval_new/worker_0/diagnosis.json
.conda/envs/pressb/bin/python scripts/summarize_policy_eval.py outputs/handover_eval_new/worker_0
```

`--allow-partial` 允许只选三层，但该 worker 声明的 15 条仍必须完整；不是忽略失败或跳过缺失记录。上述首次报告文件使用新目录，重新审计旧运行时另取报告文件名。

四组均完成后汇总；此汇总器针对每组 15 条的标准 60 条方案：

```bash
.conda/envs/pressb/bin/python scripts/aggregate_policy_eval.py outputs/handover_eval_new \
  --expected-checkpoint-step 10600 \
  --expected-checkpoint-sha256 2456b1fff5ef2d94a173b502d55b244a6b0810b92c6e39fc6b1527e1d6019312
```

若是与当前 **5090** 基线比较，在两次身份/条件相同且完整后：

```bash
.conda/envs/pressb/bin/python scripts/compare_policy_eval_runs.py \
  --baseline-root outputs/rtx5090_eval_comparison/center_corners_v1 \
  --candidate-root outputs/handover_eval_new \
  --output outputs/handover_eval_new/comparison
```

`comparison_valid=true` 表示条件可比较，不代表行为一致。这个脚本会拒绝不同 checkpoint；如果要比较不同权重，应明确标注为模型对比并使用对应分析流程，不要删掉身份检查硬凑跨机一致性。

### 9.4 后续在 5090 新采 1,200 条的入口

当前尚未在本机执行此整批。先按 `docs/rtx5090_setup.md` 做两条预检，通过后才沿用下面配置：

```bash
cd /home/pengguanqi/Workspace/Research/PressB
.conda/envs/pressb/bin/python scripts/collect_parallel.py \
  --gpus 0,1 --num-envs 3 --fps 30 --episodes-per-task 100 --seed 20260930 \
  --config configs/dataset_panel_stratified_1200.json \
  --snapshot outputs/edge_feedback/scene.usda \
  --output datasets/piper_elevator_raw_panel_stratified_5090_new
```

这是新采集批次，使用 5090 当前场景和软件指纹；不要续写原采集机 Sim 4.5 数据。后续按 `docs/dataset_panel_stratified.md`、`docs/dataset_press_only.md` 和 README 完成物理/覆盖/图像同步审计、首次亮灯裁剪、独立 LeRobot 环境导出及官方读取。不能直接把含撤回的 raw 目录当训练集。

如需重新训练，从 H200 既有 `scripts/run_piper_ft.sh`、运行目录 `config.yaml` 与 `docs/piper_lerobot.md` 入手。根据用户后续要求明确官方重启还是 checkpoint 续训，使用新的 run id；不要重跑旧 `launch.py` 或覆盖已完成的目录。新数据也不能沿用旧 dataset/split fingerprint。

## 10. 尚未完成的研究与后续方向

当前用户最新确认的是：两端 eval 都用了带偏移权重和带偏移环境；本次要求完成交接。没有一项已启动但未完成的实验需要恢复。

后续可能工作，应按用户的新要求选择：

1. 以现有 5090 60 条为新平台基线，定位低成功率和误按上一层的问题。检查 `diagnosis.json`，区分模型原始目标、IK 可行投影、平滑与实际关节跟踪；避免直接用目标坐标修正策略而改变评估含义。
2. 若研究跨机差异，分别设计图像固定输入、固定动作物理回放或渲染参数消融，保持模型/条件一致。现有结果尚未完成这种单因素归因。
3. 评估最终 step12942、增加重复次数或更多内部面板位置；这些均没有包含在当前 60 条结果中。改变 checkpoint 要同步服务、step/SHA 与分析口径。
4. 如果需要在 5090 重新采集/微调，先装独立导出环境，再形成一个新的完整验证数据批次，保留版本差异和训练/验证划分。
5. 整理并提交 5090 的兼容性改动，处理与 GitHub 旧 Sim 4.5 入口并存的问题。当前只完成工作区实现与验收，没有替用户做本轮 commit/push。

## 11. 交接时的操作习惯与判断标准

- 用户希望自主完成已授权的具体工作，减少反复确认；SSH 慢时复用连接并设 120 秒连接超时。
- 开始训练、采集或 GPU 推理前读当前占用；不要影响其他用户作业。交接时空闲不代表后续一直空闲。
- 保留已生成数据、冻结场景、模型与报告，使用独立新输出目录；不要改历史元信息、阈值或 SHA 来让审计“通过”。
- 报告分别说明运行是否完成、独立审计是否通过、任务是否成功；这三件事不同。
- 没有证据就不要承诺与旧版本逐帧一致，也不要把计划或短预检说成完整采集/完整测试。
- H200 和 5090 过去出现过系统时钟约 8 小时差异，跨机器耗时优先看单进程单调计时或同机起止记录；用户日期/时区按 Asia/Shanghai。不要把不同主机日志的墙钟直接相减。

本次交接附带的 `logs/handover_20261002/manifest.json` 列出复制证据的来源、字节数和 SHA256；`state_snapshot.json` 记录交接时的 Git、GPU、环境及连接状态。证据是历史记录，新 agent 应先读再按当前实际情况继续。
