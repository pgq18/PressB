# PressB — PiPER 电梯按键仿真与数据采集

基于 **Isaac Sim 4.5 / Isaac Lab 2.0.2** 的机械臂按键场景，包含轨迹规划、真实接触反馈、双相机采集、LeRobot 导出、动作回放和 VLA 策略闭环评估。

- 黑色 PiPER 放在桌沿，夹爪保持 8 mm 总开度夹持按压杆；初始姿态为大臂、小臂在桌面上方折叠。
- 墙上面板为 2 列 × 6 行：左列从下到上 24–29，右列 30–35。按钮受接触力推动，边缘橙灯在按下时亮、回弹后灭。
- 腕部与桌面各一台 D435；腕部复用 AgileX 官方支架网格，双路 RGB 默认 640×480。
- 采集支持面板左右 ±25 mm、前后 ±10 mm 变化；10×10 分层覆盖，每层 100 条，共 1,200 条 episode。
- 训练数据只保留折叠初始姿态到首次成功亮灯的前缀。撤回、归位用于下一次采集，保留在原始诊断记录中。

仓库包含代码、配置、资产来源清单与文档。**数据集、模型权重、下载素材、Conda 环境和运行录像不存入 Git**；它们由以下流程在本地生成。详细场景与历史验证记录见 [scene_details.md](docs/scene_details.md)。

## 安装

需要 Linux x86_64、Conda、支持 Isaac Sim 的 NVIDIA RTX GPU/驱动与 Vulkan，以及系统 `git`、`ffmpeg`、`ffprobe`（FFmpeg 需包含 `libx264`）和 DejaVu 字体。无窗口运行仍需要 GPU 渲染。显存需求随并行环境数增加；先从一张卡、一个环境验证。

```bash
git clone https://github.com/pgq18/PressB.git
cd PressB

# Ubuntu / Debian 系统依赖，按机器已有安装情况执行
sudo apt-get install ffmpeg fonts-dejavu-core

# Isaac 环境：Python 3.10，默认 .conda/envs/pressb
# 可用 CONDA_BIN=/path/to/conda 显式指定 Conda
bash scripts/setup_env.sh

# 下载固定版本资产并核对 SHA-256；已有冲突文件不会被覆盖
.conda/envs/pressb/bin/python scripts/fetch_assets.py

# 使用隔离的 OpenUSD 24.11 原样提取官方支架与 D435 网格
PYTHONPATH=.cache/usd-inspect .conda/envs/pressb/bin/python scripts/prepare_wrist_asset.py

# LeRobot 转换环境：独立 Python 3.12 / CPU PyTorch
bash scripts/setup_dataset_env.sh
```

两个 Python 环境分开使用：仿真使用 `.conda/envs/pressb/bin/python`，LeRobot 转换与官方读取使用 `.conda/envs/lerobot/bin/python`。不要将 `requirements-dataset.txt` 安装到 Isaac 环境。安装脚本设置 `OMNI_KIT_ACCEPT_EULA=YES`；使用 NVIDIA 软件和素材需遵守其许可，来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 先生成场景并验证

```bash
# 12 个按钮依次按压；每次按完回到折叠初始位姿
bash scripts/run.sh --headless --gpu 0 --video --output outputs/edge_feedback

.conda/envs/pressb/bin/python scripts/audit_episode.py outputs/edge_feedback
.conda/envs/pressb/bin/python scripts/check_results.py outputs/edge_feedback
.conda/envs/pressb/bin/python scripts/audit_global_camera.py outputs/edge_feedback

# 需要显示器 / 远程桌面时
# bash scripts/run.sh --gpu 0 --hold --output outputs/gui
```

生成的 `outputs/edge_feedback/scene.usda` 和同目录 `wrist_camera/intrinsics.json`、`global_camera/intrinsics.json` 是后续采集的输入，干净克隆中尚不存在。`--gpu` 是 Isaac/Vulkan 设备编号，请选择有可用显存的设备。演示默认 10 Hz 图像、120 Hz 物理；批量采集通过 `--fps` 独立指定采样频率。

## 采集 12 × 100 条位置分层数据

```bash
# --gpus 0 表示一张卡；也可指定 0,1 等多个设备
.conda/envs/pressb/bin/python scripts/collect_parallel.py \
  --config configs/dataset_panel_stratified_1200.json \
  --snapshot outputs/edge_feedback/scene.usda \
  --output datasets/piper_elevator_raw_panel_stratified_30hz \
  --gpus 0 --num-envs 3 --fps 30 \
  --episodes-per-task 100 --seed 20260930

.conda/envs/pressb/bin/python scripts/audit_raw_dataset.py \
  datasets/piper_elevator_raw_panel_stratified_30hz --episodes-per-task 100

.conda/envs/pressb/bin/python scripts/audit_panel_coverage.py \
  datasets/piper_elevator_raw_panel_stratified_30hz \
  --report outputs/panel_stratified_1200/coverage.json
```

每层的 10×10 网格要求 `--episodes-per-task 100`，且采集 seed 必须与配置里的 `grid_seed` 一致。少量预检可用 `collect_dataset.sh --max-episodes 2`，仍保留这些配置参数并使用独立输出目录。每次换位后检查全局相机的完整面板投影、全部按钮可达性，以及实际按压、碰撞和归位记录。并行场景只保留一个有效 Dome 环境光。

采样频率须整除 120 Hz 物理频率，例如 10、20、30、60 Hz；更改频率须使用独立数据目录。下面的首次亮灯导出示例针对 30 Hz 数据。采集元信息与代码指纹阻止混用不同来源，不能通过改写元数据绕过续采检查。

## 导出首次亮灯前缀为 LeRobot v3

```bash
PYTHONPATH=src .conda/envs/pressb/bin/python -m pressb.press_prefix \
  --raw datasets/piper_elevator_raw_panel_stratified_30hz \
  --output outputs/panel_stratified_1200/cut_plan.json \
  --episodes-per-floor 100

.conda/envs/lerobot/bin/python scripts/export_press_dataset.py \
  --raw datasets/piper_elevator_raw_panel_stratified_30hz \
  --cut-plan outputs/panel_stratified_1200/cut_plan.json \
  --output datasets/piper_elevator_lerobot_panel_stratified_press_30hz \
  --parts datasets/piper_elevator_parts_panel_stratified_press_30hz \
  --episodes-per-task 100 --part-size 25 --workers 4

.conda/envs/lerobot/bin/python scripts/confirm_press_dataset_reader.py \
  datasets/piper_elevator_lerobot_panel_stratified_press_30hz \
  --expected-episodes 1200 \
  --report outputs/panel_stratified_1200/readback.json
```

导出器对分片与最终数据执行数值、视频解码和首次亮灯终点检查；官方读取检查是额外验证。输出路径已存在时，遵循脚本的恢复/一致性检查或换用新目录，不要覆盖已验证的批次。

数据约定：

| 字段 | 含义 |
|---|---|
| task | `Press 24 floor.` … `Press 35 floor.` |
| state | 实际夹爪 TCP 位姿与实测夹爪开度 |
| action | 下一采样时刻的绝对 TCP 目标位姿与固定夹爪开度 |
| pose | `[x, y, z, qw, qx, qy, qz, gripper_width]`，基座 `base_link` 坐标系，米 |
| cameras | `observation.images.global`、`observation.images.wrist`，同步 RGB |
| metadata | 面板 XY 偏移、随机 seed、来源与采集指纹 |

夹爪 TCP 位于 `link6` 局部 Z=0.1358 m；按压杆尖端为 Z=0.24 m，两者不是同一个点。VLA-JEPA 训练适配将旋转转换为旋转矩阵前两行展开的 6D，XYZ 保持原值，固定夹爪开度不参与学习。训练及推理服务代码位于 [pgq18/VLA-JEPA](https://github.com/pgq18/VLA-JEPA/tree/base)。

## 回放与策略评估

- [录制动作回放](docs/replay.md)：从 LeRobot action 经 IK 执行并检查实际接触。
- [策略闭环评估](docs/policy_eval.md)：输入当前双相机、实测 TCP 与任务文本，执行模型返回的绝对目标。
- [动作平滑](docs/policy_smoothing.md)：120 Hz 关节插值之后默认使用 3 点因果均值滤波。

评估须显式指定与训练集一致的配置、场景快照、checkpoint step 和 SHA-256。`--panel-layouts center_corners` 覆盖训练配置的中心与四个边界角点。成功依据真实按钮行程、接触力和无误按/异常碰撞，不以接近按钮或 IK 成功代替；远端推理期间仿真暂停，因此本测试不衡量真实时间部署延迟。

历史最佳 step 10600 的固定 60 条条件评估为 **5/60 成功、27 次误按、28 次超时**；误按均为高一层。每个条件仅运行一次，不能视为任意位置的泛化成功率。实验录像和完整报告保留在本地 `outputs/`，不随源码仓库发布。

## 检查与目录

```bash
.conda/envs/pressb/bin/python scripts/fetch_assets.py --check-only
.conda/envs/pressb/bin/python -m pytest -q

# 可显式指定用于 USD 场景集成测试的快照
PRESSB_TEST_SNAPSHOT=outputs/edge_feedback/scene.usda \
  .conda/envs/pressb/bin/python -m pytest -q tests/test_dataset_scene.py

# 数据/视频审计测试在独立 LeRobot 环境运行
.conda/envs/lerobot/bin/python -m pip install pytest
.conda/envs/lerobot/bin/python -m pytest -q \
  tests/test_press_dataset_audit.py tests/test_press_export.py \
  tests/test_dataset_provenance.py tests/test_export_fps.py \
  tests/test_camera_timing.py tests/test_collection_timing.py \
  tests/test_feedback_panel_offsets.py tests/test_replay_audit.py \
  tests/test_panel_metadata.py tests/test_panel_coverage_audit.py
```

没有场景快照的克隆会跳过依赖快照的 USD 集成测试；显式指定不存在的快照会报错。素材依赖的运动学测试须在下载资产后运行。Isaac 环境不包含 PyAV 时，相应视频测试会明确跳过；上面的 LeRobot 环境命令补充执行这部分检查，避免把数据依赖装进 Isaac 环境。

| 目录 | 内容 |
|---|---|
| `src/pressb/` | 场景、机器人运动学、规划、相机、数据语义与策略控制 |
| `scripts/` | 安装、资产准备、采集、转换、回放、评估和审计入口 |
| `configs/` | 固定场景、随机位置和 1,200 条分层采集配置 |
| `tests/` | CPU 单元测试与可选 USD 场景集成测试 |
| `assets/*.json` | 固定资产来源、校验和与安装配准元数据 |
| `docs/` | 当前流程说明及注明范围的历史实验记录 |
| `vendor/`, `assets/generated/`, `assets/isaac/` | 本地下载或生成的依赖与素材（忽略） |
| `datasets/`, `outputs/`, `logs/`, `.conda/`, `.cache/` | 本地数据、结果、环境与缓存（忽略） |

更多说明：[30 Hz 数据](docs/dataset_30hz.md) · [首次亮灯裁剪](docs/dataset_press_only.md) · [位置随机化](docs/dataset_panel_randomization.md) · [分层覆盖记录](docs/dataset_panel_stratified.md)。
