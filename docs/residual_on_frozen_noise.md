# 固定初始噪声策略，重新训练动作残差

本实验固定原始 VLA-JEPA step 10600 和已完成 1M 训练的初始噪声 actor，仅从随机初始化训练动作残差。原噪声权重来自 `outputs/online_rl_fast_20261003/initial_noise_train/last.pt`。动作残差不载入旧 actor、critic、优化器、回放或随机数状态。

每个 observation 先编码，再由冻结的噪声 actor 生成 flow 初始噪声，批量解码得到 `a_noise`。残差 actor 接收 observation 和 **解码器原始 pose9**，执行 `a_noise + scale * residual`。同一 observation 的噪声只采样一次；回放中的 `next_base_action` 与下一次实际执行所使用的基础动作相同。零残差时直接执行噪声解码返回的 pose8，避免旋转转换带来变化。

冻结权重不等于固定输出：保留原噪声策略的 stochastic 采样，每个 episode/chunk 使用独立可复现的 seed。噪声采样保存、恢复 CPU 和本 learner 所用 GPU 的 RNG，不能影响残差探索或 SAC 更新。噪声模块无优化器、无梯度；每次保存 checkpoint 和运行结束均核验 actor SHA256。新残差 checkpoint 的 identities 绑定噪声 checkpoint、actor、采样与动作语义，不能作为原始 VLA 基座上的残差误用。

配置为 `configs/online_rl_residual_on_noise_400k.json`：64 个仿真环境，seed 42，2,000 条零残差预热，50,000 条渐进探索，replay 100k，batch 256，UTD 1，沿用原 residual SAC 的网络和损失。仅前 **400,000 transitions** 进入 replay 和获得更新额度，完成 **398,000 updates** 后冻结训练权重，收尾尚未结束的 episode。`training_transitions` 记录训练预算内采样，`drain_transitions` 记录额外收尾采样，`transitions` 为二者之和。收尾数据不进入 replay。每 10k 保存 `last.pt`，每 100k 另存归档，`initial.pt` 保存训练前随机初始化证据。

运行整个实验（新建独立目录，GPU 0 为 learner，GPU 1 为仿真及 VLA 推理）：

```bash
env -u PYTHONPATH .conda/envs/pressb/bin/python -u \
  scripts/run_residual_on_noise_experiment.py \
  --output outputs/online_rl_residual_on_noise_400k_20261006
```

supervisor 保存服务、learner 的 PID 和命令，监督运行并在完成或失败时清理其自身创建的服务。正式训练完成后自动使用 `train/last.pt` 和同一冻结 noise actor 做 120 条固定条件评估（seed 20260930，12 层 × 5 个面板位置 × 2 次），不更新参数。训练成功率不等同于该独立评估结果。

2026-10-07 的折扣率实验使用独立配置，只把每个 30 Hz 控制步的 `single_gamma` 从 0.99 改为 0.995，完整 7 步 chunk 的折扣为 `0.995**7 = 0.9655206468`。仿真节点的成功回报折扣和 transition discount 同步使用新值。冻结 noise actor 仍来自原来的 1M checkpoint；该 actor 的输入不包含 gamma，所以仅允许其来源与运行环境的 `simulation.single_gamma` 不同，其他模型、物理、控制与源码身份仍须全部匹配。新残差仍从零初始化；残差 checkpoint 自身的 gamma 校验不放宽。

```bash
env -u PYTHONPATH .conda/envs/pressb/bin/python -u \
  scripts/run_residual_on_noise_experiment.py \
  --config configs/online_rl_residual_on_noise_gamma0995_400k.json \
  --output outputs/online_rl_residual_on_noise_gamma0995_400k_20261007
```

新 `composition.json` 的 `frozen_noise_objective_transfer` 记录来源 gamma 和当前 gamma。完成后的比较表会加入上一轮 gamma=0.99 的 400k 结果，比较相同 120 条条件下的任务成功率，不混用不同 gamma 的折扣回报。

2026-10-08 的半尺度实验保留 gamma=0.995，仅将动作残差 scale 改为
`[0.015, 0.015, 0.015, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]`。
位置每维最大修正为 ±1.5 cm，旋转 6D 每维为 ±0.05。冻结 noise actor 的权重、
采样方式及 `noise_scale=1.5` 均沿用原设置，动作残差仍从随机参数和空 replay/优化器开始。
supervisor 允许 gamma、`learner.residual_scale` 和 `learner.residual_mode` 相对基准配置改变，实际训练和评估
使用的 scale 均会校验；其余训练设置相同。

```bash
env -u PYTHONPATH .conda/envs/pressb/bin/python -u \
  scripts/run_residual_on_noise_experiment.py \
  --config configs/online_rl_residual_on_noise_gamma0995_scale050_400k.json \
  --output outputs/online_rl_residual_on_noise_gamma0995_scale050_400k_20261008
```

本轮同样在 400k 训练结束后自动执行 120 条固定条件评估，比较表额外包含上一轮
gamma=0.995、原 scale 的结果，JSON 中保存各残差方法的实际尺度。
W&B 监控继续独立读取日志；运行配置写入后启动：

```bash
env -u PYTHONPATH .cache/wandb-monitor-venv/bin/python -u \
  scripts/watch_online_rl_wandb.py \
  --output outputs/online_rl_residual_on_noise_gamma0995_scale050_400k_20261008 \
  --name residual-on-frozen-noise-gamma0.995-scale0.5-400k-20261008
```

主要产物：`queue_status.json` 为整体进度，`train/status.json`、`metrics.jsonl`、`episodes.jsonl` 为训练记录；`composition.json` 记录初始化与冻结策略来源；`freeze_verification.json` 核验冻结状态。需结束实验时创建实验目录内 `STOP` 文件，supervisor 将请求 learner 完成当前 episode、保存并退出，不启动后续评估。

## 只学习 XYZ 的动作残差

`configs/online_rl_residual_on_noise_gamma0995_xyz002_400k.json` 设置
`learner.residual_mode="xyz"`、`residual_scale=[0.02,0.02,0.02]`。
每个 chunk 的残差 actor 只输出 7×3=21 维，三个位置坐标每步各允许最大 ±2 cm 修正，
不输出旋转残差。其条件输入仍保留冻结噪声策略解码后的完整 7×9 基础动作；
critic 接收补上 XYZ 残差后的完整 pose9。回放中的 RL action 为 21 维，
SAC 默认目标熵随有效动作维度调整为 −21/2=−10.5。

执行时只替换 pose8 的 XYZ，直接保留解码器返回的四元数和夹爪值，避免额外的旋转转换。
旋转仍由冻结的 VLA＋噪声策略根据当前观测决定，并非在整个 episode 中保持固定姿态。
`residual_mode` 缺省为 `pose9`，旧 63 维残差和 9 维噪声 checkpoint 继续兼容。

本轮保持 γ=0.995、原噪声 checkpoint/采样及 noise_scale=1.5、64 个并行环境、
其他 SAC 参数和训练日程。残差 actor/critics/优化器/replay 全部重新初始化，训练 400k，
结束后自动执行相同 120 条固定条件评估；比较表包含原尺度与半尺度的 γ=0.995 结果。

```bash
env -u PYTHONPATH .conda/envs/pressb/bin/python -u \
  scripts/run_residual_on_noise_experiment.py \
  --config configs/online_rl_residual_on_noise_gamma0995_xyz002_400k.json \
  --output outputs/online_rl_residual_on_noise_gamma0995_xyz002_400k_20261008

env -u PYTHONPATH .cache/wandb-monitor-venv/bin/python -u \
  scripts/watch_online_rl_wandb.py \
  --output outputs/online_rl_residual_on_noise_gamma0995_xyz002_400k_20261008 \
  --name residual-on-frozen-noise-gamma0.995-xyz0.02-400k-20261008
```

CPU 验证：

```bash
env -u PYTHONPATH PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .conda/envs/pressb/bin/python -m pytest -q \
  tests/test_residual_on_noise.py tests/test_online_rl_learner.py \
  tests/test_fast_online_rl_runner.py tests/test_combined_online_rl_eval.py \
  tests/test_xyz_residual.py
```

## 冻结 XYZ 策略后的动作平滑评估

`src/pressb/online_rl/action_postprocessing.py` 在冻结噪声策略和 XYZ 残差完成动作合成后，
对发送给仿真的绝对目标 pose8 做推理后处理。VLA、噪声 actor、XYZ 残差 actor 和 critic 的
权重全部保持不变，XYZ 残差 scale 仍为 `[0.02,0.02,0.02]`；不重新训练，不修改 replay，
不调整原仿真的关节平滑窗口（仍为 3）。新增后处理器只允许评估 XYZ checkpoint。

支持三种模式：

| 模式 | 位置 XYZ | 四元数旋转 | 夹爪 |
|---|---|---|---|
| `none` | 原指令数值原样发送 | 原样发送 | 原样发送 |
| `xyz_ema` | 因果指数滑动平均 | 原样发送，不重新归一化 | 原样发送 |
| `pose_ema` | 因果指数滑动平均 | 最短弧、归一化四元数 SLERP | 原样发送 |

对每个 30 Hz 控制步，位置滤波为
`p_sent[t] = alpha * p_raw[t] + (1-alpha) * p_sent[t-1]`，`0 < alpha <= 1`。
alpha 越小，历史目标占比越大，可能增加跟踪延迟。`pose_ema` 同时执行
`q_sent[t] = SLERP(q_sent[t-1], q_raw[t], alpha)`，四元数采用 wxyz 顺序并选择最短旋转弧。
这里平滑的是最终绝对目标，不是残差网络单独输出，因此原来的 ±2 cm 仍约束网络的
XYZ 残差，不能解读为后处理目标相对当前基础动作的偏移上限。

状态按仿真环境槽位和 episode ID 分别保存。新 episode 的上一目标初始化为第一帧
`observation.state` 中实测末端姿态；同一 episode 连续处理 7 步 chunk，下一 chunk
接续上一 chunk 的最后一个已发送目标，不用新观测重新初始化。每步只使用当前及过去
目标，不读取未来目标。episode ID 改变时立即清空该槽位历史，避免前一任务影响新任务。

原有模型、噪声、残差 checkpoint 和仿真身份校验全部保留。推理后处理作为独立覆盖项
写入 `postprocessing.json`，不伪装成训练时的仿真设置；其中记录模式、alpha、频率、
源码 SHA256、残差和噪声身份、评估前后残差权重及更新计数核验。
`action_postprocessing.jsonl` 为每个环境的每个 chunk 保存原始与发送的 7×8 指令、
实测姿态、chunk 起始滤波状态、episode/chunk 编号和 reset 标记，可独立重算滤波结果。
终止发生在 chunk 中间时，记录仍包含整组 7 步目标；物理平滑度统计必须按照实际执行
的物理步数截取，不能将未执行的指令计入机械臂运动。

以下是 **XYZ alpha=0.9 候选方案的评估示例，不表示该参数已被采用**。直接运行方式
要求仿真和推理服务已启动，且其身份与原 XYZ checkpoint 一致。输出目录必须尚不存在。

```bash
env -u PYTHONPATH .conda/envs/pressb/bin/python -u \
  scripts/eval_smoothed_residual.py \
  --config outputs/online_rl_residual_on_noise_gamma0995_xyz002_400k_20261008/train_config.json \
  --noise-checkpoint outputs/online_rl_fast_20261003/initial_noise_train/last.pt \
  --checkpoint outputs/online_rl_residual_on_noise_gamma0995_xyz002_400k_20261008/train/last.pt \
  --simulation http://127.0.0.1:19880 --inference http://127.0.0.1:19891 \
  --mode eval --eval-episodes 120 --seed 20261011 --device cuda:0 \
  --smoothing-mode xyz_ema --smoothing-alpha 0.9 \
  --output outputs/xyz_smoothing_a09_example
```

`scripts/run_xyz_smoothing_study.py` 可自动启动并清理它创建的服务，使用 64 个并行环境
依次评估指定候选。每个候选均测试 12 个按键 × 5 种面板布局 × 2 次，沿用相同 checkpoint，
并保存实测轨迹供运动指标分析。默认候选为无新增滤波基线、XYZ alpha=0.8/0.6、完整姿态
alpha=0.8/0.6。可通过 JSON 文件和 `--seed` 设置自定义的成对评估；例如先保存
`xyz_smoothing_cases.json`：

```json
[
  {"method": "baseline", "mode": "none", "alpha": 1.0},
  {"method": "xyz_ema_a09", "mode": "xyz_ema", "alpha": 0.9}
]
```

然后分别运行筛选和独立种子确认，两个输出目录均须是新目录：

```bash
env -u PYTHONPATH .conda/envs/pressb/bin/python -u \
  scripts/run_xyz_smoothing_study.py \
  --cases-file xyz_smoothing_cases.json --seed 20261011 --phase screen \
  --output outputs/xyz_smoothing_screen_example

env -u PYTHONPATH .conda/envs/pressb/bin/python -u \
  scripts/run_xyz_smoothing_study.py \
  --cases-file xyz_smoothing_cases.json --seed 20261012 --phase confirm \
  --output outputs/xyz_smoothing_confirm_example
```

候选筛选同时检查成功次数至少 119/120、不低于同次运行的无新增滤波基线，并在两侧
共同成功的条件上比较实测运动平滑度；后续使用预先选定的另一种子确认。
`--phase` 记录评估阶段，supervisor 本身不会替代研究者执行上述采用标准。
成功率、关节与末端轨迹指标应同时检查，滤波指令看起来更平滑不能单独证明实际运动
改善或成功率保持。统计脚本为 `scripts/analyze_rl_smoothing.py`，滤波与冻结契约的
CPU 测试为 `tests/test_action_postprocessing.py`。

### 2026-10-09 实测结果：保留默认无新增滤波

共完成 11 组 × 120 = **1,320 次**任务。每组覆盖 12 个按键 × 中心和四角偏移 × 2 次。
所有 VLA、噪声和 XYZ 残差权重冻结，评估更新为 0；scale 始终为 `[0.02,0.02,0.02]`。
每组配套的无新增滤波基线避免把不同种子的成功率波动误判为滤波效应。

| 阶段 / seed | 同期基线 | XYZ alpha=0.8 | XYZ alpha=0.6 | 位置＋姿态 alpha=0.8 | 位置＋姿态 alpha=0.6 | XYZ alpha=0.9 |
|---|---:|---:|---:|---:|---:|---:|
| 初筛 / 20260930 | 117/120 | 119/120 | 117/120 | 120/120 | 117/120 | — |
| 姿态候选独立确认 / 20261009 | 118/120 | — | — | 117/120 | — | — |
| XYZ 候选独立确认 / 20261010 | 116/120 | 117/120 | — | — | — | — |
| 最后轻度候选筛选 / 20261011 | 119/120 | — | — | — | — | 118/120 |

**没有候选通过全部门槛，因此没有推荐或自动启用的滤波配置。** XYZ alpha=0.8 在两组中
均高于同期基线，但独立确认未达到预定的至少 119/120；不能据此保证保持此前单次
99.17% 的成功率。姿态 alpha=0.8 和轻度 XYZ alpha=0.9 均出现低于同期基线的结果。
alpha=0.9 筛选未通过，因此没有运行原计划的 seed=20261012 确认，也没有继续搜索参数。
各选择/拒绝决策分别保留于 `selection.json`、`selection_xyz.json`、`conservative_plan.json`。

XYZ alpha=0.8 的真实运动平滑改善可复现：

| 指标（匹配且双方成功的任务） | 初筛 116 对 | 独立确认 113 对 |
|---|---:|---:|
| 关节 jerk RMS 的中位数变化 | −14.48% | −13.58% |
| 末端位置 jerk RMS 的中位数变化 | −30.30% | −30.52% |
| 任务时长中位数变化 | +2.69% | +0.67% |

这些指标来自 120 Hz 实测关节速度和 TCP 轨迹，并按实际终止截取，未使用指令曲线、
插入的终止停帧或视频播放时间代替实际运动。统计先对每个 episode 计算向量 jerk RMS，
再在双方共同成功的匹配条件内比较中位数。异步重置导致部分匹配条件的初态存在微小
数值差异；初态逐位相同子集也有相近的平滑改善，详见各阶段敏感性分析。
关节 jerk 的下降不等价于整段运动完全无抖动，旋转也可能贡献可见抖动。

所有结果位于 `outputs/online_rl_xyz_smoothing_20261009`。初筛分析为
`analysis_screen/`；其他阶段分别为 `confirm/analysis_confirm/`、
`confirm_xyz/analysis_confirm/`、`conservative/analysis_screen/`。
`smoothing_metrics.json` 和配套 CSV 记录逐方法、逐按键和逐 episode 结果，
动作重算与冻结审计均通过。后处理器 27 项、轨迹指标分析 23 项 CPU 测试通过。
四轮汇总为根目录 `smoothing_study_report.json`、`all_phase_metrics.csv`，逐按键结果为
`all_phase_per_button.csv`。XYZ alpha=0.8 两轮合计 236/240、同期基线 233/240，仅作
描述性统计；其中包含用于筛选的样本，不能将该合计当作独立验证成绩。

实验性视频 `videos_exploratory_xyz_a08/videos/xyz_vs_smoothed.mp4` 对比左侧无新增平滑
和右侧 XYZ alpha=0.8。两侧取独立确认 seed=20261010 中每个按键的第一次中心布局，
不根据结果挑选；视频内两侧均 12/12，完整评估仍为基线 116/120、平滑 117/120。
42.57 秒、30 fps，以 1× 仿真时间同步，保留终止画面但不加速；同一实测轨迹渲染为
全局和腕部视角，回放中不推进物理。24 条轨迹、48 个视角和合成审核均通过。
此视频用于观察滤波效果，不表示该候选已获采用。
