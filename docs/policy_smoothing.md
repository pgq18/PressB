# Eval 执行平滑

> 文中的数据集、权重、图像和运行报告属于本地产物，不随 Git 仓库分发；新机器请先按 [README](../README.md) 生成场景。历史结果不代表本次安装已经完成验证。

参考用户提供的《告别机械臂“帕金森”：手把手教你用线性插值与均值滤波优化 PiPER 上的 LeRobot 运动控制》，eval 默认启用关节命令平滑。这里采用文章的动作历史、线性插值和均值滤波思路，按项目的 30 Hz 位姿策略与 120 Hz 物理控制实现。

执行顺序是：模型七步绝对 TCP 位姿 → 原有带关节限位和速度限制的 IK → 每步四个 120 Hz 关节线性插值点 → 因果均值滤波 → PhysX。

均值作用于六个关节角。窗口保留前面的命令，跨模型 action chunk 连续；每个并行环境独立，每条 episode 用折叠初始关节命令填满历史。每个预测动作仍占四个物理步，七步预测全部按顺序执行。模型仍接收真实运动后的图像和实测状态。

默认 `--smoothing-window 3`，即当前点和前两个点的均值。窗口工作在 **120 Hz**，名义延迟为 `(window-1)/(2×120)`，默认约 **8.33 ms**。支持 1、3、5、7、9、11；**1 关闭均值滤波，保留原来的线性插值行为**。5 点窗口延迟为 16.67 ms。平滑能缓和命令的高频变化，但没有施加硬性的加速度/jerk 上限，也不能保证消除误按或提高任务成功率。

初始历史和全部输入满足关节位置及逐步速度约束时，均值作为凸组合保持这些约束；独立审计还会核对实际浮点下发命令的限位和速度。

## 使用

启动相同的 H200 模型服务后，在原固定面板场景验证三个楼层：

```bash
bash scripts/eval_policy.sh \
  --config outputs/edge30_source/config.json \
  --snapshot outputs/edge30_source/scene.usda \
  --expected-checkpoint-step 12800 \
  --expected-checkpoint-sha256 c2502396b093ec45d2b3fa2ac1d602b29ecafaa147a7bfe14768971028834c72 \
  --floors 24,33,35 --num-envs 3 --gpu 4 \
  --max-seconds 15 --seed 20260930 --smoothing-window 3 \
  --output outputs/policy_smoothing/eval_step12800_window3_v1
```

删除 `--floors` 参数即可测试全部 12 层。复现实验需使用新输出目录。做关闭平滑的对照时显式设置 `--smoothing-window 1`。

推理种子按 `base_seed + (repeat×12+floor−24)×10000+chunk_index` 确定。这样单独测某几个楼层也保留它们在完整 12 层评估中的随机数序列；完整顺序评估的种子与原实现相同。

## 记录与验证

- 原始模型 `action_pose8`、`action_pose9` 及请求响应继续完整保存。
- `physics.npz/q_command_unsmoothed` 保存进入滤波器的 float64 插值命令，`q_command` 保存实际 float32 下发命令。
- `actions.jsonl/q_before`、`q_target` 是原 IK 插值端点；`q_executed_before`、`q_executed_endpoint` 是实际滤后执行端点。`executed_*_residual` 用后者计算，终止发生在部分步长中时也按实际终止点记录。
- manifest 与 episode metadata 保存窗口、物理频率、延迟和初始化/重置规则。独立审计从原始插值命令重建滤波结果，并分别核对 IK 误差、滤后执行误差和真实物理跟踪误差。
- 首次按键亮起、误按、碰撞或超时仍立即终止，不额外执行滤波尾部。

离线检查使用上一轮 **12 条完全相同的已录制关节命令**。关节命令 jerk RMS 原为 19769.03 rad/s³，3 点降为 6593.45（下降 **66.65%**），5 点降为 3954.50（下降 **80.0%**）；窗口 1 逐值还原原始命令，滤后关节位置和速度检查全部通过。该指标不代表实际 TCP jerk 或闭环成功率。见窗口对照（本地产物：`outputs/policy_smoothing/recorded_window_comparison.json`）及5 点详细记录（本地产物：`outputs/policy_smoothing/recorded_command_check.json`）。

## Isaac 实测

在固定居中场景，以相同 `step_012800` 权重和逐楼层随机种子，对 24、33、35 层分别测试 3 点和 5 点窗口，每条最多 15 秒。两次进程均正常退出，六条 episode、十二路视频全部通过独立审计，未出现异常碰撞。

| 目标楼层 | 先前无均值滤波 | 3 点 | 5 点 |
| --- | --- | --- | --- |
| 24 | 误按 25 | 误按 25 | 误按 25 |
| 33 | 成功，11.90 s | 成功，12.14 s | 超时 |
| 35 | 超时 | 超时 | 超时 |

因此默认采用响应延迟更小、在本次小批对照中保留 33 层成功的 **3 点**。5 点仍可选。这是每配置每层一次的验证，不证明成功率提高，也不是新的完整 12 层评估。

3 点运行中，实际命令相对同一运行的平滑前输入，jerk RMS 降低 66.64%–66.65%。实测 TCP jerk RMS 在这三层从先前约 228–235 m/s³ 降为约 137–141 m/s³；由于闭环轨迹与终止时刻也改变，这项跨运行对照只作观察指标。见3 点报告和录像（本地产物：`outputs/policy_smoothing/eval_step12800_window3_v1/summary/README.md`）、5 点报告和录像（本地产物：`outputs/policy_smoothing/eval_step12800_window5_v1/summary/README.md`）及验证汇总（本地产物：`outputs/policy_smoothing/summary.json`）。

独立审计子集时使用 `--allow-partial`，并显式指定摘要脚本需要的文件名：

```bash
.conda/envs/lerobot/bin/python scripts/audit_policy_eval.py \
  --run outputs/policy_smoothing/eval_step12800_window3_v1 --allow-partial \
  --report outputs/policy_smoothing/eval_step12800_window3_v1/audit.json

.conda/envs/pressb/bin/python scripts/diagnose_policy_eval.py \
  --run outputs/policy_smoothing/eval_step12800_window3_v1 \
  --output outputs/policy_smoothing/eval_step12800_window3_v1/diagnosis.json

.conda/envs/pressb/bin/python scripts/summarize_policy_eval.py \
  outputs/policy_smoothing/eval_step12800_window3_v1
```
