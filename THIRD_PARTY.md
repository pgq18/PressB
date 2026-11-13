# 第三方软件与资产来源

本仓库不打包下载的机器人网格、NVIDIA 场景资产或 Conda 环境。`scripts/fetch_assets.py` 按固定提交和 SHA-256 下载 35 个文件；`assets/sources.lock.json` 保存来源与许可声明。下载器保留既有冲突文件并报错，不覆盖用户修改。

| 来源 | 固定版本 / 提交 | 使用范围与许可说明 |
|---|---|---|
| NVIDIA Isaac Sim | 当前 5090 环境 5.0.0；原采集环境 4.5.0 | 物理仿真与渲染；遵守 NVIDIA 软件许可 |
| Isaac Lab | 当前 v2.2.0；历史 v2.0.2 | 独立下载到 `vendor/IsaacLab`；适用其上游许可 |
| AgileX `robot_lab` | `b868140eeb1459acefef24a865587ec39a5278c3` | PiPER URDF/USD/网格；下载时同时保存上游 Apache-2.0 LICENSE |
| AgileX `piper_isaac_sim` | `8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2` | 原厂相机支架与 D435；该提交无根 LICENSE，Piper 包声明不能视为完整授权，不对其几何另行推定许可 |
| Intel RealSense description | 上述固定源树中 `_d435.urdf.xacro`、D435 网格和包声明 | 保留源文件中的 Intel 许可声明与归属信息 |
| NVIDIA Isaac 4.5 内容库 | 逐文件 SHA-256 | 桌子、支架及材质；适用 NVIDIA 的资产使用条款 |
| ZPRL / ZPRL-PressB | 方法来源 `a34a1cba029702af82c0fed873be9e0e6bd274b3` | `src/pressb/online_rl/learner.py` 的 SAC 适配保留上游 MIT notice；参考仓库独立克隆，不打包其环境或数据 |
| VLA-JEPA PiPER 适配 | 来源版本 `9c9dc71c8e199c7428b25d103ab666395832c001`，另含 H200 的 joint／多任务工作区模块 | 推理、训练和数据代码已内置 `src/pressb/model_sources/vlajepa`，来源和逐文件 SHA 见其中清单；不含权重 |
| Wall-X / Wall-OSS | 来源和本地修改摘要见 `src/pressb/model_sources/walloss/source_manifest.json` | 模型、推理、训练、数据和 CUDA 源码已内置；原源码版权头保留；不含权重或编译二进制 |
| NVIDIA CUTLASS | `e51efbfe18fe4f4cbb66ab814c55bf4aa0185491` | 仅构建 Wall-OSS CUDA 扩展时获取到缓存；原许可文本保留于 Wall-OSS 源码包的 `licenses/` |
| DPPO / Diffusion Policy | `cc7234ad7ff39a8f32de3af903606723a16f0648` | MIT；`src/pressb/model_sources/diffusion_policy` 内置条件 UNet、DDPM/DDIM 公式及依赖模块，保留 LICENSE 和逐文件原始／适配后 SHA-256；PressB 添加视觉编码、LeRobot 训练和节点接口，不包含上游模型权重 |
| Hugging Face LeRobot / π₀.₅ | `d324ffe810d17264a0b1e628698aa1fa09aa639c`（包版本 0.4.5） | Apache-2.0；由 `scripts/install_pi05.py` 校验源码压缩包 SHA-256 后安装，PressB 提供推理、数据与训练适配器；不在仓库内打包模型权重或上游源码包 |
| Ascend PyTorch 扩展 / CANN | 已验证 `torch_npu 2.10.0`、PyTorch 2.10.0、CANN 9.0.0 | 运行时依赖单独安装；不在仓库分发驱动、工具包或二进制。环境配套及验证范围见 `docs/ascend_pi05.md` |

`prepare_wrist_asset.py` 在隔离的 OpenUSD 24.11 中提取官方支架与相机子树，以兼容 Isaac Sim 4.5。它保留网格点与拓扑，生成文件及来源记录位于 `assets/generated/official_wrist/`，不提交 Git。

本仓库尚未指定自有代码的开源许可证。公开可见不替代许可授权；第三方组件始终遵循各自的许可。

模型内化保留已有逐文件版权／SPDX 声明，不另行授予许可。所用 VLA-JEPA
源树未包含其 `pyproject.toml` 引用的 LICENSE 文件，原元数据的 MIT 声明单独保留；
Wall-X 源树也无顶层 LICENSE，原 `setup.py` 只有 BSD classifier。
未把这些元数据推定为全部文件的统一授权，也未补造缺失的许可原文。
各文件来源、原始摘要与本地改动见两份 `source_manifest.json`。

`docs/media/` 仅发布 PressB 仿真生成的精选压缩演示，并记录原始视频哈希与审核范围；
不包含机器人网格或场景资产，不改变上述第三方许可。仓库版本与用途见 `repos.lock.json`。

上游链接：

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/v2.0.2)
- [AgileX robot_lab](https://github.com/agilexrobotics/robot_lab/tree/b868140eeb1459acefef24a865587ec39a5278c3)
- [AgileX piper_isaac_sim](https://github.com/agilexrobotics/piper_isaac_sim/tree/8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2)
- [NVIDIA Omniverse 软件许可](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)
- [DPPO](https://github.com/irom-princeton/dppo/tree/cc7234ad7ff39a8f32de3af903606723a16f0648)
