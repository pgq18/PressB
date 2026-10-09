# RTX 5090 工作站运行说明

2026-10-01 在 `ssh 5090` 对应工作站完成部署与实测。项目目录是 `/home/pengguanqi/Workspace/Research/PressB`，系统为 Ubuntu 24.04.4 LTS，驱动为 580.173.02，GPU 为两张 RTX 5090（每卡约 32 GB）。

## 安装与启动

本次环境组合为 Python 3.11.16、Isaac Sim 5.0.0、Isaac Lab v2.2.0、PyTorch 2.7.0+cu128、torchvision 0.22.0+cu128；NumPy 1.26.0、SciPy 1.15.3、Pillow 11.2.1、Gymnasium 1.2.0、Warp 1.7.1、PyAV 15.1.0。Conda 环境保存在项目下的 `.conda/envs/pressb`，不依赖手动激活。PressB 已安装为 editable 包，完整依赖快照位于 `logs/install-5090-requirements.txt`，`pip check` 通过。

Isaac Lab 使用官方 tag v2.2.0，Git SHA 为 `46dff135f44683f031edf346e544fcfd8456b2bb`；核心 Python 包的元数据版本为 0.44.9，发布版本以源码 tag 为准。

```bash
cd /home/pengguanqi/Workspace/Research/PressB

CONDA_BIN=/home/pengguanqi/miniconda3/condabin/conda \
  bash scripts/setup_env_5090.sh

.conda/envs/pressb/bin/python scripts/fetch_assets.py
PYTHONPATH=.cache/usd-inspect \
  .conda/envs/pressb/bin/python scripts/prepare_wrist_asset.py

# 检查所有固定版本素材的 SHA-256。
.conda/envs/pressb/bin/python scripts/fetch_assets.py --check-only
```

`setup_env_5090.sh` 是本次部署提供的新安装入口。**不要在该 Python 3.11 环境上运行原来的 `scripts/setup_env.sh`**：旧入口固定 Python 3.10 / Isaac Sim 4.5 / Isaac Lab 2.0.2，属于另一套依赖配置。LeRobot 导出使用单独的数据处理环境，不要将 `requirements-dataset.txt` 安装进仿真环境。

## 12 个按钮与双相机验收

下面启动参数已按当前 `scripts/run.sh`、`scripts/run_sim.py` 和 `configs/scene.json` 核对。默认配置依次按压 24–35 层，每次按完归位，记录腕部和桌面两路 RGB-D。默认物理频率 120 Hz、相机采样 10 Hz，RGB 分辨率为 640×480。

```bash
cd /home/pengguanqi/Workspace/Research/PressB

# 使用尚未用于其他运行的输出目录，避免混合记录。
bash scripts/run.sh --headless --gpu 0 --video --output outputs/new_run

.conda/envs/pressb/bin/python scripts/audit_episode.py outputs/new_run
.conda/envs/pressb/bin/python scripts/audit_global_camera.py outputs/new_run
.conda/envs/pressb/bin/python scripts/check_results.py outputs/new_run
```

`check_results.py` 必须退出码为 0 并输出 `PASS: 12/12 physically pressed in order`，同时列出两路已校验的 RGB-D 帧数。它会重算按压顺序、物理接触、按钮回弹与归位、夹爪跟踪、相机帧完整性和桌面相机的完整面板投影。不要用 `--max-steps` 截断运行作为完整验收；不要仅凭 `run_status.json` 显示完成就判定通过。

验收产物位于 `outputs/new_run/`：

- `report.json`、`audit_summary.json`、`global_camera_audit.json`：运行与独立校验结果。
- `trajectory.npz`：实际机械臂轨迹和按钮物理测量。
- `scene.usda`：生成的仿真场景，可供后续并行采集引用。
- `episode.mp4`：场景视角录像。
- `wrist_camera/rgb.mp4`、`global_camera/rgb.mp4`：两路相机录像；各目录同时保存 RGB、深度、内参及时间戳。

需验证第二张 GPU 时，用独立输出目录重复运行：

```bash
bash scripts/run.sh --headless --gpu 1 --video --output outputs/new_run_gpu1
.conda/envs/pressb/bin/python scripts/check_results.py outputs/new_run_gpu1
```

必须显式传入 `--gpu 0` 或 `--gpu 1`；旧启动脚本的默认 GPU 编号为 4。该编号采用 Isaac/Vulkan 的设备枚举。无窗口运行仍使用 GPU 渲染；图形界面需在工作站桌面会话中运行，普通 SSH 不提供桌面显示：

```bash
bash scripts/run.sh --gpu 0 --hold --output outputs/gui_new_run
```

## 实际验收记录

- 35/35 个官方来源资产 SHA-256 校验通过，官方腕部支架与 D435 几何来源审计通过。
- 两张 GPU 均通过实际 PyTorch CUDA 运算、矩阵乘法和 Warp 自定义 kernel 检查。
- GPU 0 完整运行 23,206 个物理步，12 个按钮全部按压、回弹并归位，没有意外碰撞；双相机各 1,934 帧有效 RGB-D，物理、相机、面板投影与官方支架独立审计全部通过。
- 双相机渲染时间一致，渲染时间与物理采样时间的偏移没有漂移。
- GPU 1 完成两个并行环境、30 Hz 的两条测试轨迹，分别 517 / 512 帧；四个视角的几何最优延迟均为 0，首亮与灭灯均同步，灭灯后首帧橙色像素均为 0。
- Isaac Lab 的 AppLauncher / SimulationContext 在 GPU 1 成功 reset、步进 10 次并正常退出；同进程 PyTorch 与 Kit 内置 Warp 运算通过。
- 最终全套测试：414 passed，31.34 秒，退出码 0，无跳过或失败。

| 验收内容 | 项目内路径 |
| --- | --- |
| 完整场景、双路 RGB-D、录像及审计 | `outputs/edge_feedback/` |
| GPU 验证 | `logs/gpu-verification-5090.json` |
| 双相机渲染时间一致性 | `logs/verification-5090-camera-clocks.json` |
| 通过验收的 30 Hz 原始采集 | `datasets/5090_smoke_30hz_v2/` |
| 按钮边缘反馈 | `outputs/5090_smoke_feedback_v2.json` |
| 图像与状态几何同步 | `outputs/5090_smoke_camera_sync_v2/report.json` |
| Isaac Lab 启动与步进 | `logs/verification-5090-isaaclab.log` |
| 最终完整测试 | `logs/verification-5090-pytest-final.log` |

初次调试证据另行保留；`datasets/5090_smoke_30hz/` 有已定位的灭灯余色，应使用通过验收的 v2 版本。

## 30 Hz 并行采集预检

已生成的 `outputs/edge_feedback/scene.usda` 和同目录双相机内参可直接作为采集输入。以下仅采两条随机面板轨迹：

```bash
bash scripts/collect_dataset.sh --gpu 1 --num-envs 2 --fps 30 \
  --episodes-per-task 1 --max-episodes 2 \
  --config configs/dataset_panel_randomized.json \
  --snapshot outputs/edge_feedback/scene.usda \
  --output datasets/new_smoke_30hz
.conda/envs/pressb/bin/python scripts/audit_raw_dataset.py \
  datasets/new_smoke_30hz --episodes-per-task 1 --allow-partial
.conda/envs/pressb/bin/python scripts/audit_button_feedback.py \
  datasets/new_smoke_30hz --episodes-per-task 1 --allow-partial \
  --episode-ids 0 1 --output outputs/new_smoke_feedback.json
.conda/envs/pressb/bin/python scripts/audit_camera_sync.py \
  datasets/new_smoke_30hz --episode-ids 0 1 --output outputs/new_smoke_camera_sync
```

原始轨迹包含撤回和归位，训练数据仍按项目已有的首次亮灯前缀导出流程处理。完整 1,200 条分层采集的配置及 seed 要求见 [dataset_panel_stratified.md](dataset_panel_stratified.md)。此次只配置仿真环境并运行预检，未采集新的完整数据集，也未安装独立 LeRobot 导出环境。

## 版本适配

Sim 5.0 的 `rendering_frame` 返回 Fabric 参考时间的分子和分母。记录器保留该字典及类型说明，旧版整数帧号仍可读取；`rendering_time` 保持 API 提供的仿真秒数。分子不能被当成帧号。

灯光切换后增加到 16 次保持物理时间不变的渲染刷新，消除本次测试发现的边缘余色；刷新次数写入采集元信息和指纹。物理时间、机械臂状态不变的断言和原有图像校验阈值均保留。不要把不同指纹的采集续写到同一数据目录。

如需重新执行 GPU、官方资产、完整场景和双相机联合验收，请使用新的输出目录：

```bash
bash scripts/verify_simulation_5090.sh outputs/reverification_new
```

官方版本依据：[Isaac Lab v2.2 安装说明](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/setup/installation/pip_installation.html)、[Blackwell 渲染与依赖更新](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/refs/release_notes.html)。
