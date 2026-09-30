# 从 LeRobot 末端 action 回放实际按键

> 文中的数据集、权重、图像和运行报告属于本地产物，不随 Git 仓库分发；新机器请先按 [README](../README.md) 生成场景。历史结果不代表本次安装已经完成验证。

本页记录含撤回和归位的完整动作回放实验。当前训练 episode 已改为首次按键亮灯结束，见[首次亮灯数据说明](dataset_press_only.md)；下述完整动作验收包含释放和归位条件，不能直接用它评判截断后的训练 episode。

新训练数据已经完成前缀等价验证（本地产物：`outputs/press_only_30hz/replay_prefix_equivalence.json`）：从正式发布目录读取每层一条，与本页已执行回放的 action 前缀逐值核对，12/12 一致，且新终点处的实际按键均亮起。只调整末帧用于保持的目标，之前已执行的控制序列保持一致。该证据复用已有回放的物理前缀，不代表本次新执行了 1,200 条仿真。可用 `scripts/audit_press_replay_prefix.py --dataset datasets/piper_elevator_lerobot_press_30hz --report outputs/my_prefix_check.json` 在 LeRobot 环境中复核，报告路径须未使用。

本次完整回放入口读取 `datasets/piper_elevator_lerobot_edge_30hz` 数据集，复用采集时冻结的场景和配置。默认选择原始 `source_episode_id=0..11`，即 24–35 层各一条。LeRobot 的 episode 顺序按转换分片排列，脚本通过导出清单查找对应行，不假定其 `episode_index` 等于原始编号。

控制输入为保存的 8 维 `action=[x,y,z,qw,qx,qy,qz,gripper_width]`，使用 `base_link` 坐标系。每个 action 是下一采样时刻的绝对夹爪 TCP 目标；TCP 位于 link6 局部 z=0.1358 m，按压杆尖端位于 z=0.24 m。回放使用完整位置和姿态的连续逆运动学，限制关节位置和速度，并在每两个 30 Hz 端点之间以 120 Hz 线性插值关节驱动目标。图像同样按 30 Hz 记录。

机械臂只在每条 episode 开始时用记录的首帧关节状态初始化，之后由新求解的关节驱动目标和 PhysX 执行动作。完整的记录关节轨迹只作为诊断参考，控制器不读取它；灯状态也根据新的接触力和按钮位移实时产生。按钮不由脚本移动。

回放时间 0 对应首个采样状态；物理执行仍先进行一次 1/120 秒步进，与采集器的采样原点一致。第 4 个后续物理步到达 `action[0]`，第 8 个到达 `action[1]`。末帧 action 是重复的终端目标，不额外增加一段运动，因此 N 帧对应 `(N-1)*4+1` 个物理记录。

在项目根目录运行以下命令。提取与回放输出必须使用新目录，避免覆盖已有证据。

```bash
# 在 LeRobot 环境中读取官方数据，默认提取 12 层各一条。
.conda/envs/lerobot/bin/python scripts/prepare_replay.py \
  --dataset datasets/piper_elevator_lerobot_edge_30hz \
  --output outputs/my_replay/input

# 在 Isaac 环境中执行真实物理回放并录制两个相机。
bash scripts/replay_dataset.sh \
  --input outputs/my_replay/input \
  --output outputs/my_replay/run \
  --gpu 4 --num-envs 3

# 独立检查实际物理证据及两路视频。
.conda/envs/lerobot/bin/python scripts/audit_replay.py \
  --input outputs/my_replay/input --output outputs/my_replay/run
```

`prepare_replay.py --source-episode-ids 12 13 14` 可选择其他原始 episode，编号用空格分隔。提取器核对 LeRobot 源行、任务、采样时钟、场景 SHA 和数据来源，保存逐数组无损 NPZ。回放器再次核对输入、场景和配置的来源哈希。

每条回放目录中包含 `wrist.mp4`、`global.mp4`、初始/按压/释放/最终截图、120 Hz 的 `physics.npz`、30 Hz 的 `frames.npz` 和 `metadata.json`。运行级 `replay_manifest.json` 保存源数据、控制方法及源码 SHA；`report.json` 为执行器结果，独立验收结论在 `audit.json`。

独立验收根据实测接触力与按钮行程重算触发和释放，检查目标按键、误按、异常碰撞、末端 action 与关节命令的对应关系，以及初始状态和最终归位。源轨迹与回放实测轨迹的差异仅作为诊断；回放不是图像或状态的逐帧复写。

当前试验输出为 `outputs/replay_edge_30hz/run_v1`。**所选 12 条已全部完成实际按压、释放、归位，独立验收 12/12 通过。** 共 25,632 个物理记录、6,417 个双相机采样；24 路视频合计 12,834 帧全部完整解码，时间戳和动态画面检查通过。每条仅触发对应楼层一次，没有误按或异常碰撞；按键峰值行程 2.866–3.156 mm，峰值接触力 0.941–1.039 N，最大最终归位关节误差 0.000727 rad（约 0.042°）。

此轮回放使用保存的笛卡尔末端 action，经完整姿态逆运动学执行；没有使用记录的关节目标轨迹驱动机械臂。关节命令在 30 Hz 端点处对原 action 的位置误差最大约 0.327 微米，实测末端轨迹与原 state 的最大位置差约 0.031 mm。后一个数值为诊断结果，不是任务成功判据。

查看各楼层回放视频（本地产物：`outputs/replay_edge_30hz/README.md`）、独立审计（本地产物：`outputs/replay_edge_30hz/run_v1/audit.json`）与12 层按压画面（本地产物：`outputs/replay_edge_30hz/run_v1/audit_images/all_floors_pressed.png`）。该实验覆盖每层一条，结论不外推为全部 1,200 条已做物理回放，也不代表训练策略的泛化表现。

仿真执行器写完所有结果后，在 `Simulation App Shutting Down` 阶段返回 **143**，不能记为进程正常退出。已落盘的全部物理记录和视频经独立验收通过；退出原因尚未确定，证据保留于进程退出记录（本地产物：`outputs/replay_edge_30hz/run_v1/process_exit.json`）和 `logs/replay_edge30_run_v1.log`。原始 LeRobot 数据集没有修改。
