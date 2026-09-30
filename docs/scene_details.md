# 场景结构、相机与历史验证记录

本页保留项目的场景设计和历史本机验证说明。`outputs/`、`datasets/`、依赖和下载素材不随 Git 仓库发布；下列产物路径须在本机生成后查看。新机器请先按 [README](../README.md) 的安装与场景准备步骤运行。

## 本机完整验证结果

低位近距离机位已在 RTX 5880 Ada 上完整运行，进程正常退出（退出码 0）。运动、夹爪与腕部 RGB-D 独立审计 **88 项全部通过**，官方腕部资产审计 **115 项全部通过**，包含面板覆盖与桌面安装的全局相机审计 **28 项全部通过**。下表保留边缘灯带修改前 `outputs/latest` 的实测结果；新版独立输出到 `outputs/edge_feedback`。

| 实测项 | 结果 |
| --- | --- |
| 按压 / 回弹熄灯 / 收回折叠初始姿态 | 各 12/12，按 24–35 顺序完成 |
| 实际回位外壳倾角 | 大臂约 0.993°，小臂约 1.006°；均低于 1.2° |
| 上下主臂 X 方向重叠量 | 约 0.250 m |
| 物理时长 / 轨迹采样数 | 193.383 s / 23,206 条 |
| 最大回位关节误差 | 0.000732 rad，约 0.0420° |
| 逐键峰值位移 | 2.84–3.24 mm |
| 逐键峰值接触力 | 0.95–1.04 N |
| 最大关节跟踪误差 | 0.02433 rad |
| 两指目标位置 / 最大实测偏差 | +4 mm、−4 mm / 0.00073 mm |
| 场景视频 | 1280×720，10 fps，193.4 s |
| 腕部 RGB-D | 1,934 组，640×480，10 Hz |
| 固定全局 RGB-D | 1,934 组，640×480，10 Hz；逐帧时间戳与腕部相机一致 |
| 腕部深度有效像素比例 | 平均 100.00%，逐帧最低 100.00% |
| 全局深度有效像素比例 | 平均 100.00%，逐帧最低 100.00% |
| 相机与实测腕部变换的一致性 | 最大位置误差 0.00053 mm、姿态误差 0.00000141 rad |
| 固定相机位姿相对设定值的最大误差 | 0.000049 mm、0.000000012 rad；全程保持固定 |
| 全局视野覆盖 | 完整面板及 12 按钮边界在图像内，面板距画框至少 108.97 px；允许机械臂裁切 |
| 桌面安装 | 底座 18×18 cm，底面 z=0.76 m；距桌前沿 69.1 mm、左侧沿 197.9 mm |
| 超过 0.1 N 的异常外部接触 / 机械臂自接触 | 0 / 0，包含初始预热与运动过程 |

查看 低位相机布置（本地产物：`outputs/latest/global_camera_setup.png`）、固定 D435 视频（本地产物：`outputs/latest/global_camera/rgb.mp4`）、固定相机图像与安装审计（本地产物：`outputs/latest/global_camera_audit.json`）、腕部视频（本地产物：`outputs/latest/wrist_camera/rgb.mp4`）、实测轨迹（本地产物：`outputs/latest/trajectory.csv`）、运动及腕部审计（本地产物：`outputs/latest/audit_summary.json`） 和 官方资产审计（本地产物：`outputs/latest/mount_asset_audit.json`）。检查从实际关节角重算末端和相机位姿，并逐事件核对接触力、按钮行程、灯状态变化和归位，未用规划成功替代物理验证。

官方资产审计独立核对源 USD 的固定提交和 SHA-256、网格点与面的原样引用、URDF 安装变换，以及旧自制支架是否已从场景移除；`wrist_mount_closeup.png` 用于直观核对支架、相机和夹爪的装配关系。

## 环境与运行

本机已配置项目 Conda 环境 `.conda/envs/pressb`：Python 3.10、Isaac Sim 4.5.0、Isaac Lab v2.0.2、PyTorch 2.5.1+cu124。Isaac Lab 源码位于 `vendor/IsaacLab`；任务控制使用 Isaac Sim Core/PhysX。环境安装和版本记录见 `logs/install-versions.json`。

```bash
cd /path/to/PressB

# 本机已经安装；新机器或需要恢复时执行
bash scripts/setup_env.sh
.conda/envs/pressb/bin/python scripts/fetch_assets.py

# 原样提取官方网格，生成 Isaac Sim 4.5 可读的兼容资产
PYTHONPATH=.cache/usd-inspect .conda/envs/pressb/bin/python scripts/prepare_wrist_asset.py

# 完整物理仿真、场景视频及腕部/左侧固定相机两路 RGB-D
# 先写独立目录，保留原先已验证的 latest
bash scripts/run.sh --headless --gpu 4 --video --output outputs/compact_view

# 有显示器/远程桌面时打开交互窗口，结束后保持窗口
bash scripts/run.sh --gpu 4 --hold --output outputs/gui

# 独立复核实测轨迹、灯光、逐键归位以及腕部相机数据
.conda/envs/pressb/bin/python scripts/audit_episode.py outputs/compact_view
.conda/envs/pressb/bin/python scripts/check_results.py outputs/compact_view

# 独立复核固定相机图像、时间同步、完整面板覆盖与桌面安装
.conda/envs/pressb/bin/python scripts/audit_global_camera.py outputs/compact_view

# 不启动 Isaac Sim，独立复核官方支架/D435 的 USD 网格与安装变换
PYTHONPATH=.cache/usd-inspect .conda/envs/pressb/bin/python scripts/audit_mount_asset.py outputs/compact_view

# 官方 URDF 运动学、可达性、限位、速度和水平初始姿态测试
.conda/envs/pressb/bin/python -m pytest -q
```

默认使用 GPU 4，`--gpu` 是 Isaac/Vulkan 物理设备编号；可按空闲情况更改。无窗口运行也需要 NVIDIA 驱动与 Vulkan。启动关闭本任务不需要的 P2P/IOMMU 自检，不改动内核或驱动配置。使用 Isaac 默认快速退出并显式传递退出码，失败任务不会被误报为成功。`--max-steps` 是截断诊断，不能通过完整任务验证。

上述复现命令写入独立目录 `outputs/compact_view`；检查新版边缘灯带结果时，将审计命令的目录改为 `outputs/edge_feedback`；历史整面发光结果保留于 `outputs/latest`。

启动脚本设置 `OMNI_KIT_ACCEPT_EULA=YES`；许可见 [NVIDIA Omniverse 软件许可](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)。

## 初始姿态和布局

配置位于 `configs/scene.json`。桌面高 **0.76 m**，底座位于世界 **x=-0.24 m**，桌沿 **x=-0.14 m**，底座中心距桌沿 0.10 m。按钮正面 x=0.46 m，最低一排 z=0.98 m，行距 35 mm，列距 90 mm。

初始关节角 `home_q` 为 `[0, 0.069762054, 0, 0, 0, 0]` rad，即在官方折叠零位基础上将肩关节抬高约 3.997°。**大臂向桌内收回，小臂反向叠在上方**，两段沿垂直桌沿的方向布置，均位于桌面上方。末端和腕部相机朝向面板。每次按完一个按钮后，机械臂完整收回这一姿态并停稳，再开始下一次。

这里区分关节中心连线与真实外壳直段：Piper 外壳存在弯折和轴心偏置，不能用关节中心连线水平来代表外壳水平。根据官方 STL 的长直壳面计算，原厂零位下两段外壳分别倾斜约 3°、5°；当前姿态将两段各自与桌面的夹角降至约 **0.999°**。让两段同时精确达到 0° 需要肘关节约 +2°，超过官方上限 0°，因此未改动原厂限位来伪造严格水平。

`home_side.png` 是实际 PhysX 归位后的正交侧视图，可以直接核对上下折叠与桌面关系。审计从每次归位的**实际关节角**重新计算折叠重叠量和外壳倾角；逐次实测倾角要求不超过 1.2°。

世界坐标 +Z 向上，+X 指向墙，+Y 为面向面板时的左侧。长度为米，关节角为弧度，时间为秒，接触力为牛顿。按压杆尖端在 `link6` 的局部 +Z 方向 0.24 m。

`gripper_joint_positions_m` 配置两指的闭合位置。官方手指在两个关节都为 0 时指腹互相贴合，所以夹住直径 8 mm 的杆应使用 `[0.004, -0.004] m`。杆体仍作为腕部固定工具，其根部为 `link6` 局部 `z=0.125 m`，伸入指腹，末端按压位置不变。记录中额外保存两指目标和实测位置；独立审计要求全程偏差不超过 0.25 mm，并检查官方关节限位。

任务过程中只发送关节驱动目标，不逐帧传送机器人位置，也不脚本移动按钮。按键允许沿墙面法向移动 4 mm，由弹簧回弹。控制采样 **120 Hz**。`run.sh` 单场景演示默认每 12 个物理步同步保存两路图像，即 **10 Hz**；可通过 `render_stride`、`wrist_camera_capture_stride` 和 `global_camera_capture_stride` 调整。LeRobot 批量采集使用独立的 `--fps` 参数，当前默认 **30 Hz**、每 4 个物理步同步采图，详见[30 Hz 采集入口](../docs/dataset_30hz.md)。两种方式均不改变物理时间步。

## 输出文件

本节列出 `run.sh` 单场景演示的输出；当前 30 Hz LeRobot 批量数据的目录与内容见[数据集说明](../docs/dataset_30hz.md)。

| 文件 | 内容 |
| --- | --- |
| `episode.mp4` | 场景视角视频，默认 1280×720、10 fps，需 `--video` |
| `scene.png` / `completed.png` | 开始及结束的上下折叠初始姿态，按钮均熄灭 |
| `home_side.png` | 实际归位后的正交侧视图，展示上下折叠及桌面间隙 |
| `wrist_mount_closeup.png` / `gripper_closed.png` | 官方支架、D435 与夹爪闭合在按压杆两侧的近景 |
| `global_camera_setup.png` | 独立布置视角，展示机械臂左旁的实体固定 D435、0.18 m 桌面支架及面板的位置关系 |
| `press_24.png` … `press_35.png` | 对应按钮实际按住并亮灯的画面 |
| `scene.usda` / `completed_scene.usda` | 包含官方资源引用、相机、支架及物理属性的 USD 场景 |
| `trajectory.csv` / `trajectory.npz` | 120 Hz 的目标/实际关节角、速度、两指目标/实际位置、末端位置、目标楼层、阶段、12 按钮位移/接触力/灯状态 |
| `events.json` | 每层 `button_pressed`、`button_released`、`cycle_home` 三类实际事件 |
| `report.json` | 物理执行结果、逐键归位误差、异常接触及资产来源 |
| `audit_summary.json` | 独立复算的按压、释放、归位、相机完整性和腕部安装误差 |
| `mount_asset_audit.json` | 独立核对官方 USD 来源与哈希、网格点面、URDF 安装变换及旧自制支架的移除 |
| `global_camera_audit.json` | 固定相机完整帧、两路同步、固定世界位姿、完整面板覆盖及支架贴桌范围的独立审计；机械臂覆盖另行报告 |
| `trajectory.png` | 实测末端轨迹、关节角、按键位移及接触力图 |
| `wrist_camera/rgb.mp4` | 腕部相机视角视频 |
| `wrist_camera/rgb/*.png` | 原始 640×480 RGB 帧 |
| `wrist_camera/depth/*.npy` | 对应帧的 float32 深度，单位 m，无效位置为非有限值 |
| `wrist_camera/timestamps.jsonl` | 图像与深度路径、仿真时间、目标楼层、运动阶段、相机世界位姿 |
| `wrist_camera/intrinsics.json` | 分辨率、K 矩阵、坐标约定、光学安装变换 |
| `wrist_camera/floor_24.png` … `floor_35.png` | 各按钮按压时的腕部 RGB 示例，另有 `_depth.png` 深度预览 |
| `wrist_camera/camera_report.json` | 帧数与有效深度统计 |
| `global_camera/rgb.mp4` | 左侧固定 D435 全局视角视频，640×480、10 fps |
| `global_camera/rgb/*.png` / `depth/*.npy` | 固定相机逐帧 RGB 和对应的 float32 米制深度 |
| `global_camera/timestamps.jsonl` / `intrinsics.json` | 与腕部相机同步的帧时间戳、固定世界位姿及独立内参记录 |
| `global_camera/initial.png` / `completed.png` / `floor_24.png` … `floor_35.png` | 固定相机初始、结束及逐键按压图像；按压图另有 `_depth.png` 深度预览 |
| `global_camera/camera_report.json` | 固定相机帧数、深度有效率和资产来源 |

CSV 的按键向量后缀 `0…11` 对应楼层 `24…35`；`gripper_q_actual_0/1` 与 `gripper_q_target_0/1` 分别对应 `joint7/joint8`，单位 m。USD 场景可单独打开；逐键动作由 `scripts/run_sim.py` 执行，打开 USD 不会自动重放轨迹。

## 腕部相机与数据读取

相机固定在夹爪腕部 `link6` 上，复用 **AgileX 官方 `piper_isaac_sim` 的现成 RealSense 支架与配套 D435**。该仓库的 `piper_description_v100_realsense_camera_v2_base.usd` 提供 `/meshes/realsense_mid_stand` 和 `/meshes/d435` 两个子树，原始网格点和面保持不变，没有重绘支架。早先“官方模型没有现成支架”的判断不准确：原先选用的 `robot_lab` 资产没有附带这一装配，但独立的官方 Isaac 适配仓库已提供。

官方基础 USD 含中文 prim 名称，Isaac Sim 4.5 自带的 USD 版本无法直接读取。`scripts/prepare_wrist_asset.py` 在隔离的 OpenUSD 24.11 进程中原样复制上述两个网格，舍弃不兼容的中文 `Looks` 节点，生成 `assets/generated/official_wrist/assembly.usda` 和 `provenance.json`。场景引用这一兼容副本；这一步只解决文件格式兼容，不重建或简化原网格。独立资产审计对照固定来源核验点、面、安装变换和生成记录。

这套装配与当前 Piper 使用不同的腕部坐标。依据两版本真实夹爪几何配准，旧 `link6` 到当前 `link6` 的变换为 **绕 Z 轴 +90°、沿 Z 轴 −4 mm**，证据见 [官方支架配准记录](../assets/official_wrist_registration.json)。在这一坐标映射之后，官方 URDF 的支架、相机固定关节和相机外壳视觉变换完整保留；RGB 光学中心采用 Intel `_d435.urdf.xacro` 中相对 `camera_link` 的 `+0.015 m Y` 偏移。相机随真实物理腕部运动；独立审计由实际关节角、URDF 和安装外参逐帧复算相机位姿。

640×480 图像按 [D435 官方 RGB 标称视场 69°×42°](https://www.realsenseai.com/products/stereo-depth-camera-d435/) 作方形像素中心裁切：`fx=fy≈625.2214 px`、`cx=320`、`cy=240`，有效视场约 **54.2084°×42°**。这些是仿真内参，不是某台实机的标定结果。位姿四元数按 **wxyz** 保存，光学坐标采用 USD 的 **+X 向右、+Y 向上、−Z 向前**。深度是到成像平面的轴向距离，与 RGB 对齐。

腕部新增载荷约 **104.296 g**：D435 沿用上游 72 g 质量；官方支架 URDF 的质量为 0，因此由闭合网格体积和 ABS 密度 1050 kg/m³ 估算为 32.296 g。上游明确注明相机惯性数值不可靠，当前按相机尺寸作均匀盒体近似。质量与惯性建模依据保存在同一配准记录中。

RGB 来自 Isaac RTX 渲染；深度是**理想几何真值**，不模拟 D435 硬件的立体匹配、噪声或最小测距过程。本场景近距离按键所产生的有效几何深度，不能据此推断真实 D435 在同样距离下必然输出有效硬件深度。

```python
import json
from pathlib import Path
import numpy as np
from PIL import Image

root = Path("outputs/latest/wrist_camera")
frame = json.loads((root / "timestamps.jsonl").read_text().splitlines()[0])
rgb = np.asarray(Image.open(root / frame["rgb"]))
depth_m = np.load(root / frame["depth"])
valid = np.isfinite(depth_m) & (depth_m > 0)
calibration = json.loads((root / "intrinsics.json").read_text())
print(rgb.shape, depth_m.shape, frame["time"], valid.mean())
```

运行期间使用 `Camera.get_rgba()` 和 `Camera.get_depth()` 读取图像，后者启用了 `distance_to_image_plane` annotator。`*_depth.png` 仅用于显示，研究计算应使用保存的 `.npy` 深度。

## 左侧固定相机

固定 D435 放在机械臂左旁的**同一桌面**上（世界 +Y）。光学中心为 `global_camera_eye=[-0.28, 0.24, 1.05] m`，朝向 `global_camera_target=[0.43, 0, 1.0675] m`；光心高于桌面 **0.29 m（29 cm）**，与机械臂底座的水平距离约 **0.24331 m（24.331 cm）**。视角以完整按钮面板入画为优先，机械臂可以出现在画面边缘或被裁切。按压过程中，夹爪、腕部相机和杆尖可能自然遮挡面板局部；完整入画不代表所有像素全程无遮挡。固定相机在每次运行中保持世界位姿不变，图像由独立的 Isaac Camera 渲染与采集。

支架复用 NVIDIA **`Props/Mounts/Stand/stand.usd`** 的现成网格，水平缩放底座至 **0.18 m**，主支架高度约 **0.17727 m（17.727 cm）**，底面落在 **z=0.76 m** 的桌面上。上方黑色延长杆由原来的 **10 cm** 缩短为 **5 cm**，通过小型云台连接件接到 D435 官方底部螺孔坐标；相机外壳直接复用与腕部相同的兼容 D435 网格。`global_camera_support_surface="tabletop"` 选择桌面安装，`global_camera_stand_footprint_m=0.18` 控制底座尺寸，`global_camera_head_riser_m=0.05` 控制黑色延长杆长度。`global_camera_setup.png` 用于检查实体支架安装位置，`global_camera/rgb.mp4` 则是该固定镜头实际看到的操作画面。

固定相机装配记录位于 本地历史文件 `assets/global_camera_registration.json`，已核对低位机位的实际 USD 来源、网格、静态碰撞、螺孔安装和间隙。18 cm 底座距桌前沿约 **69.064 mm**、左侧沿约 **197.862 mm**，底面与桌面重合；完整运行未出现支架或相机与机械臂的异常接触。在 Isaac Sim 的视口相机列表中选择 `/World/GlobalCamera/CameraLink/ColorCamera` 可查看此镜头，选择 `/World/GlobalCameraSetupView` 可查看支架布置。

固定相机与腕部相机采用相同的 640×480 仿真内参和约 54.2084°×42° 有效视场，但独立保存 RGB、深度、标定和位姿。读取方式与前述示例一致，将 `root` 改为 `Path("outputs/latest/global_camera")` 即可；两路 `timestamps.jsonl` 按帧号和 `time` 对齐。

`global_camera_coverage="panel"` 要求完整面板外框和 12 个按钮边界在图像内，并保留至少 **5 px** 边距。`scripts/audit_global_camera.py` 仍根据全部实测关节轨迹计算机械臂关节和按压杆尖端的覆盖范围，但裁切机械臂不会使此模式失败；未配置该项的历史结果默认使用 `"all"`，要求所有这些点都在视野内。

桌面安装审计还要求支架记录 `support_surface="tabletop"`、`support_z=table_height` 和完整的 `world_bounds_m`。支架底面与桌面高度误差不得超过 **1 mm**，支架整体 XY 边界不得越过桌沿（数值容差 0.1 mm）。当前桌面范围为 **x∈[−0.89,−0.14] m、y∈[−0.50,0.50] m**。相机必须位于机械臂左侧、水平距离小于 **0.6 m**，且光心高于桌面。这些安装检查依据记录的支架世界包围盒和桌面配置；实际碰撞另由完整物理运行的接触记录核实。

独立审计同时检查全部图像与深度、双路时间同步、标称 D435 内参和相机世界位姿不变。`scripts/check_results.py` 会根据报告中的全局相机配置调用这一审计；新场景缺少相机文件会失败，历史无全局相机的场景保持兼容。投影覆盖与画面遮挡是不同检查，实际遮挡通过全局图像和视频核实。最终覆盖、有效深度统计和验收结果以运行后的 `global_camera_audit.json` 为准。

## 官方资源与来源

- [AgileX robot_lab](https://github.com/agilexrobotics/robot_lab)：Piper USD、URDF、官方几何/惯性/碰撞与关节限位，固定 commit `b868140eeb1459acefef24a865587ec39a5278c3`。
- [NVIDIA Simple Room 桌子](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Simple_Room/Props/table_low.usd)：复用桌架网格与材质纹理，上覆精确平整桌面以与碰撞面一致。
- [NVIDIA Stand 固定支架](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/Mounts/Stand/stand.usd)：左侧固定相机复用现成网格与静态碰撞，缩放为 0.18 m 底座的桌面支架并调整立柱高度。
- [AgileX piper_isaac_sim](https://github.com/agilexrobotics/piper_isaac_sim/tree/8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2)：复用现成 RealSense 支架和配套 D435 的原始 USD 网格，经兼容提取后供 Isaac Sim 4.5 引用，固定 commit `8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2`；安装关系来自其 `piper_description_v100_realsense_camera_v2.urdf`，名义光学外参来自 `realsense2_description/urdf/_d435.urdf.xacro`。
- [Isaac Lab v2.0.2](https://github.com/isaac-sim/IsaacLab/tree/v2.0.2)：已安装并保留源码，当前任务不需要训练强化学习策略。

`scripts/fetch_assets.py` 校验并获取 **35 个依赖文件**，包含新增的 NVIDIA Stand；来源、固定提交和 SHA-256 保存在 `assets/sources.lock.json`，`--check-only` 可离线复核。官方相机装配原文件保存在 `vendor/piper_isaac_sim`，包含基础 USD、URDF、两个 DAE、光学外参 Xacro、README 和两个包的原始 `package.xml`。旧 D455 资产不再属于当前版本的必需依赖。

环境安装脚本将 `usd-core==24.11` 以 `--no-deps --target .cache/usd-inspect` 安装到隔离目录，检测到同版可用时跳过；不会替换 Isaac Sim 主环境的 USD。兼容提取和独立 USD 审计均只在各自命令中设置 `PYTHONPATH=.cache/usd-inspect`，不要将它全局用于 Isaac Sim 运行。若单独恢复这一依赖，可执行 `.conda/envs/pressb/bin/python -m pip install --no-deps --target .cache/usd-inspect usd-core==24.11`。独立审计读取场景和源资产，不启动 Isaac Sim。

已检查的 Isaac 候选目录见 `assets/isaac/asset_inventory.json`；参考视频抽帧见 `assets/reference/contact_sheet.jpg`。墙面、面板与按钮按任务尺寸构建，腕部支架和左侧桌面支架均复用官方资源。`robot_lab` 的原始许可随资源保留，NVIDIA 资产遵循其对应许可。`piper_isaac_sim` 在所固定提交中没有独立根许可文件，`piper_description/package.xml` 仍为 `TODO: License declaration`；Intel `realsense2_description` 声明 Apache License 2.0。项目保留这些实际声明，没有为官方支架推定或补造许可。
