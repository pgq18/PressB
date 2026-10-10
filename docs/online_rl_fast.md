# 在线 RL 高吞吐采样

**2026-10-09 状态：** 两种独立 RL 已完成各 1M 训练，固定条件评估均为 81/120，基础模型为 25/120。实际训练总耗时分别为动作残差 19.81 h、初始噪声 15.16 h；下文早期短窗口外推保留为性能诊断，不能代替最终耗时。后续固定噪声并从头训练 400k 的 XYZ 残差得到 119/120，约 7.43 h；完整比较见 [中文 README](../README.zh-CN.md#当前结果)，配置与复现见 [组合残差训练](residual_on_frozen_noise.md)。

三个节点仍通过 HTTP 通信，冻结 VLA-JEPA 不参与梯度更新。新增入口是 `serve_rl_fast_simulation.py`、`serve_rl_inference.py --batch-size 64`、`run_fast_online_rl.py`。一个 transition 仍表示一个环境实际执行的一段动作，最多 7 个 30 Hz 控制动作 / 28 个 120 Hz 物理 tick；不把 tick 或单张图像计作额外 transition。

## 调度和实现

- 环境结束后，通过 `/reset_envs` 恢复该环境的初始关节、按钮和控制历史，并改变其目标与面板偏移。第一次全局 reset 缓存经过 90 tick 稳定的状态；后续重置不推进物理时间。其他环境继续各自的 episode，不等整组结束。物理 step 仍使用共享 World 的同步批处理。
- 每个 transition 的真实 terminal observation 在 reset 前保存，SAC replay 不会把下一条 episode 的初始状态当成上一条的末状态。reward 和 discount 依照实际执行的物理 tick 计算。
- 全部环境的 FK、受限 IK、关节读取和按钮读取批量计算。仿真使用 PhysX tensor views；当前 NumPy 控制路径不宣称已经实现 Isaac Lab 的完整 GPU 原生任务。
- 仅在 action chunk 边界和准确的终止 tick 渲染；两个相机仍使用 tiled rendering。相机按 320×240（4:3）渲染，再按原模型入口使用的 BILINEAR 缩放为 224×224，PNG 无损传输；灯光变更后另作无物理推进的捕获。
- 推理节点一次批量执行视觉编码与 flow 解码，缓存每条观察的完整 action tokens；支持本机 5090 或远端 H200。actor 使用的 2048 维特征可用 little-endian float32 字节传输，减少 JSON 体积，数值不量化。context 释放合并进下次 `/encode`。
- HTTP 工作在线程中等待；actor 读取、replay 写入和 SAC 梯度更新仍在训练主线程执行。等待采样或远端推理时，训练端消耗已采集 transition 对应的更新额度。默认 warmup 后 UTD=1，不靠少训练来提高吞吐。

这些选择参考了本地 ZPRL 的独立 worker autoreset、批量编码、终止观测处理，以及 Isaac Lab 的 [DirectRLEnv 与控制 decimation](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/tutorials/03_envs/create_direct_rl_env.html)、[tiled camera](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/overview/core-concepts/sensors/camera.html) 和 [相机规模实测方法](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/how-to/estimate_how_many_cameras_can_run.html)。ZPRL 的 vector step 本身仍有批次屏障；网络与 learner 更新重叠是本项目的实现。

## 启动

本机已有环境与冻结资产的命令示例，输出目录必须新建。fast 仿真使用 `threadpoolctl>=3` 限制已加载 BLAS 的线程数；它已列入 `online-rl` 可选依赖，本机无需重新安装环境。

```bash
env -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .conda/envs/pressb/bin/python scripts/serve_rl_fast_simulation.py \
  --gpu 1 --port 19880 --num-envs 64 --max-seconds 15 --gpu-dynamics \
  --output outputs/online_rl/fast_sim_new
```

本机推理使用独立的 `.conda/envs/vlajepa-inference`，VLA-JEPA 克隆在 `src/VLA-JEPA`。依赖、最小权重文件及目录要求见该仓库的 `docs/piper_rtx5090_inference.md`（提交 `9c9dc71`）。该环境复用 Torch 2.7/cu128，单独安装与 H200 匹配的 Transformers 4.57 等模型依赖，不修改 Isaac 环境。

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .conda/envs/vlajepa-inference/bin/python scripts/serve_rl_inference.py \
  --vla-repo src/VLA-JEPA \
  --checkpoint .cache/vlajepa/piper_panel_stratified_press30hz_pose9_3epochs_20260929/checkpoints/step_010600 \
  --base-vlm .cache/vlajepa/Qwen3-VL-2B-Instruct \
  --base-encoder .cache/vlajepa/vjepa2-vitl-fpc64-256 \
  --device cuda:0 --port 19891 --batch-size 64 --torch-threads 4 \
  --image-preprocess-device cuda --cache-size 512 --cache-ttl 3600
```

这里 GPU1 由独立的 Isaac 和推理进程共享，GPU0 只运行 learner；容量与吞吐验证须包含两者同时运行的情况。若使用 H200，用 `scripts/sync_rl_inference.py --remote-root <独立目录>` 同步 addon，远端服务及 SSH 转发使用 19881，并将训练端 URL 改回对应端口。三个模块的 API 与跨机器能力保留。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .conda/envs/pressb/bin/python scripts/run_fast_online_rl.py \
  --config configs/online_rl_fast_action_residual.json --device cuda:0 \
  --simulation http://127.0.0.1:19880 --inference http://127.0.0.1:19891 \
  --port 19882 --output outputs/online_rl/fast_residual_new
```

噪声方法使用 `configs/online_rl_fast_initial_noise.json`，与 residual 顺序复用服务。两个配置各默认 1,000,000 transition；每 10,000 条覆盖 `last.pt`，每 100,000 条保留归档，replay 容量 100,000。停止时停止补充新 episode，排空正在执行的 episode，完成已挣得的梯度更新，再保存 checkpoint；因此最终计数可略超预算。

```bash
curl --fail http://127.0.0.1:19882/status
curl --fail -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:19882/stop
```

固定条件评估仍循环遍历 12 层 × 中心/四角，seed=20260930；64 环境建议 120 条（两轮），数量不必整除环境数，但初始评估条数须不少于环境数。基线与学到的策略必须使用同一 fast 仿真和推理约定比较。

## 与旧训练的关系

批量 DLS 控制器、320×240 渲染、降低渲染频率和 CUDA 图像预处理会改变闭环数值行为。这不是旧采集器的逐值等价加速。显式 `--gpu-dynamics` 还将物理切至 GPU（保留 CPU tensor 读取）；该模式关闭 CCD，须使用独立物理约定、正确配置 GPU 碰撞缓冲，并重新验证接触与基线。清单记录这些约定、源文件 SHA、模型权重 SHA 和 batch 配置；`--resume --checkpoint ...` 要求匹配的 fast 身份，不允许直接接入旧 replay。

需要转移已有 residual 时，用 `--warmstart-actor <旧last.pt>` 显式只加载 actor。critic、target critic、温度、优化器、replay、随机数与新实验计数重新初始化；原 checkpoint 与其计数保留在来源记录中。不能把迁移前后计数当作一次完整恢复。

## 性能与正确性证据

组件性能不是整个训练的吞吐。完整耗时必须包含 reset、渲染、图片传输、特征编码、动作执行、replay、梯度更新和检查点。`status.json` 的 `session_transitions / wall_seconds` 给出该次运行的整体速率；timing 中 RPC 与 updates 会重叠，不能直接相加。

当前组件证据位于 `outputs/online_rl_throughput_20261002/`、`outputs/online_rl_inference_perf_20261002/`：

- 真实历史观察 B64：H200 单机 CUDA 预处理编码约 96 observation/s；通过 SSH 的 packed 特征响应约 46.47 observation/s。通信是必须单独计入的成本。
- 100k 容量 replay 的计算基准：residual 约 255 SAC update/s、noise 约 233 update/s。这是合成 replay 上的计算速度，不是采样性能。
- 1,542 条已录制动作通过关节位置/速度约束检查。专家目标的最大 TCP 误差小于 0.000407 mm；513 条模型目标中有 49 条在近初始姿态阶段比旧 IK 多出超过 1 mm 误差，因此保留独立控制器约定。
- 两条专家动作在真实 fast Isaac 中首次按下即终止，tick=977/981，与旧诊断相同。按钮位移、接触力、奖励/discount 和终态图像通过；这是物理路径验收，不是策略成功率证据。

本机部署及最终 GPU 物理验收证据：

- VLA-JEPA 已在 H200 创建提交 [`9c9dc71`](https://github.com/pgq18/VLA-JEPA/commit/9c9dc71c8e199c7428b25d103ab666395832c001)，经本机中继推送到 GitHub，并克隆至 `src/VLA-JEPA`。10.27 GB 的 step10600 权重 SHA-256 已逐文件校验。
- 本机 B64 推理约 75.48 observation/s；与 H200 对照时 pooled 特征及状态逐值相同，基础动作 XYZ 最大分量差 3.58e-7 m。该结果是单独推理服务测试。
- 相机修正前的 64 环境 GPU 物理服务约 91.00 transition/s（完整 28 tick chunk），含首次 reset 约 81.09/s；仅为旧图像路径的组件诊断速度。62 条正确目标专家重放全部真实接触成功，错按、无接触超时、独立 reset 不改变同伴状态均通过。证据：`outputs/online_rl_fast_probe_20261003/final_simulation_source_qualification.json`。
- GPU 碰撞缓冲容量显式设置并读回，原生 PhysX 错误会中止运行。故意降低缓冲容量的真实负例退出码为 1，且未开放 ready 服务。此前含容量溢出错误的 96–98/s 数据已作废。
- 最终相机修正版完整 CPU 回归：648 passed、2 skipped；日志 `outputs/online_rl_throughput_20261002/pytest_final_camera.log`。

相机比例曾导致实际回归：直接原生 224×224 渲染时，RTX 按输出比例确定垂直视野，物体垂直尺寸缩成原来的 0.75；模型训练入口则将 640×480 图像压到 224×224。这一候选 base 只有 2/120 成功，已停止并保留为诊断。修正版维持原 4:3 投影再缩放，并重新评估基线。真实 8 环境、32 对 RGB 比较（home 与 24 tick 后）显示最佳垂直比例回到 0.995–1.005，state 最大差为 0，RGB MAE 中位数 2.50/255；见 `outputs/online_rl_fast_probe_20261003/camera4_3_qualification.json`。候选诊断的 37.47 transition/s 不视为修正版的完整速度。

整体吞吐、资源配置与正在运行的任务以 `outputs/online_rl_current.json` 和相应运行目录的状态、清单为准。

2026-10-03 修正版长程运行目录为 `outputs/online_rl_fast_20261003`。固定条件 base 评估 25/120（20.83%）；两种 RL 各 1M，训练后各 120 条同条件评估。下面几段为当时训练早期的速度观测；现已完成，最终成功率与耗时见本文开头。

相机修正版早期实测约24.6 transition/s（含持续独立重置、batch推理和warmup后UTD=1），1M外推约11.3小时；相比原3环境1.4/s约17.6倍。GPU0/GPU1每5秒采样峰分别1387/20504MiB。此为约5k transition短窗口的估算，不是1M完成时间；噪声阶段尚未开始，额外flow decode需另测。证据 `outputs/online_rl_fast_20261003/throughput_report.json`。

同次独立监控的较晚85秒窗口为22.59/s（1M约12.3小时），较长120秒更新后窗口为24.52/s（约11.3小时）；宜按11–13小时/1M作当前规划。较晚窗口约67%时间在仿真step与reset，32%在推理，SAC更新均与RPC重叠。

## 从训练日志生成曲线

本轮两种方法已于2026-10-04完成各1M训练及最终评估。可从原始日志离线复现曲线：

```bash
.conda/envs/pressb/bin/python scripts/plot_online_rl_training.py \
  --run outputs/online_rl_fast_20261003
```

默认在运行目录的 `plots/` 导出 recent SR、训练进度、SAC优化指标、终止原因与最终评估图，包含PNG/PDF/SVG、5页合并PDF `training_curves.pdf`、两个方法的CSV与源文件SHA清单。

主recent SR使用最近1000个已完成并记录的episode，浅线使用最近100个；不足窗口长度时使用全部已完成episode。绘图依据每行metrics的episode计数精确对应episode日志前缀，不插值补造零点或中途评估。末尾短窗口会受停止补充episode、排空最后64个环境影响。统计说明和图像核对结果见运行目录 `plots/README.md`、`plots/qa.json`。窗口可通过 `--recent-episodes` 和 `--throughput-transitions` 调整。
