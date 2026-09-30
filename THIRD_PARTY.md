# 第三方软件与资产来源

本仓库不打包下载的机器人网格、NVIDIA 场景资产或 Conda 环境。`scripts/fetch_assets.py` 按固定提交和 SHA-256 下载 35 个文件；`assets/sources.lock.json` 保存来源与许可声明。下载器保留既有冲突文件并报错，不覆盖用户修改。

| 来源 | 固定版本 / 提交 | 使用范围与许可说明 |
|---|---|---|
| NVIDIA Isaac Sim | 4.5.0 | 物理仿真与渲染；遵守 NVIDIA 软件许可 |
| Isaac Lab | v2.0.2 | 独立下载到 `vendor/IsaacLab`；适用其上游许可 |
| AgileX `robot_lab` | `b868140eeb1459acefef24a865587ec39a5278c3` | PiPER URDF/USD/网格；下载时同时保存上游 Apache-2.0 LICENSE |
| AgileX `piper_isaac_sim` | `8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2` | 原厂相机支架与 D435；该提交无根 LICENSE，Piper 包声明不能视为完整授权，不对其几何另行推定许可 |
| Intel RealSense description | 上述固定源树中 `_d435.urdf.xacro`、D435 网格和包声明 | 保留源文件中的 Intel 许可声明与归属信息 |
| NVIDIA Isaac 4.5 内容库 | 逐文件 SHA-256 | 桌子、支架及材质；适用 NVIDIA 的资产使用条款 |

`prepare_wrist_asset.py` 在隔离的 OpenUSD 24.11 中提取官方支架与相机子树，以兼容 Isaac Sim 4.5。它保留网格点与拓扑，生成文件及来源记录位于 `assets/generated/official_wrist/`，不提交 Git。

本仓库尚未指定自有代码的开源许可证。公开可见不替代许可授权；第三方组件始终遵循各自的许可。

上游链接：

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/v2.0.2)
- [AgileX robot_lab](https://github.com/agilexrobotics/robot_lab/tree/b868140eeb1459acefef24a865587ec39a5278c3)
- [AgileX piper_isaac_sim](https://github.com/agilexrobotics/piper_isaac_sim/tree/8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2)
- [NVIDIA Omniverse 软件许可](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)
