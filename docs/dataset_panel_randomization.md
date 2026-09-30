# 面板左右、前后随机偏移

> 文中的数据集、权重、图像和运行报告属于本地产物，不随 Git 仓库分发；新机器请先按 [README](../README.md) 生成场景。历史结果不代表本次安装已经完成验证。

批量采集默认使用 `configs/dataset_panel_randomized.json`。每个 episode 在机械臂折叠归位后，独立采样面板的位置；机械臂、桌子、两台相机的安装位置和面板高度固定。

| 参数 | 默认范围 | 正方向 |
| --- | --- | --- |
| `panel_offset_y_m` | −0.025 至 +0.025 m | 面向墙壁时向左 |
| `panel_offset_x_m` | −0.01 至 +0.01 m | 远离机械臂 |

两个方向分别均匀采样，再执行安全检查。不通过的组合被拒绝，最多尝试 8 次；全部失败会报错，不提交错误样本。随机种子由采集 seed、episode ID 和楼层确定，因此改变并行环境数量不会改变同一 episode 的布局。偏移量是相对原始居中布局的绝对值，不会逐次累加。

左右移动整块面板，包括外框、数字、灯带、12 个按钮和弹簧锚点。前后移动时，背墙及其饰板、扶手一同移动，保持面板安装在墙上。侧墙、地面、桌子和相机不随之移动。

每个采样布局都检查 12 层完整的归位—接近—按压—撤回轨迹、官方关节限位、速度、桌面/墙面/相机支架间隙，以及整块面板的相机投影。实际 episode 使用扰动后的时序和接近距离，再检查一次。默认要求相机画面边界至少留 24 像素；这保证完整面板在画面范围内，按压期间机械臂仍可能自然遮挡局部。

Isaac 每次重新布局后先物理静置 24 个步长，再核对全部按钮的实测静止位置、弹簧锚点、固定相机姿态及实际 USD 面板包围盒的八角投影。这些准备步骤不写入 episode。采集期间继续监测按钮横向漂移、异常接触、按压亮灯、释放熄灯和最终归位。

左右范围由 ±2 cm 扩大到 **±2.5 cm**，前后仍为 ±1 cm。扩大后的四个 XY 边界组合在接近距离 62、65、68 mm 下均通过全部 12 层完整轨迹检查；最小相机支架保守间隙为 **15.150 mm**，仍满足原有 15 mm 阈值。最紧组合是 X=−10 mm、Y=+25 mm、接近距离 68 mm。+3 cm 的部分轨迹会低于该阈值，因此当前相机位置下未进一步扩大到 ±3 cm。扩展范围时仍须通过逐布局检查，不能只修改边界后假定可达。

默认新原始目录为 `datasets/piper_elevator_raw_panel_randomized_30hz`。原始 schema 11 保存每条 episode 的 X/Y 偏移、实际几何与相机检查记录；LeRobot 导出清单和动作回放保留这些布局信息。`observation.state`、`action` 仍为基座坐标系下的 8 维夹爪绝对位姿，偏移不添加为模型输入。

小批验证或正式采集使用相同入口，按可用 GPU 调整参数：

```bash
# 每层一条，共12条，两个并行环境
bash scripts/collect_dataset.sh \
  --config configs/dataset_panel_randomized.json \
  --output datasets/piper_panel_randomized_preview \
  --num-envs 2 --gpu 6 --fps 30 --episodes-per-task 1

# 每层100条，共1200条；这是启动命令，本次修改不会自动运行此批次
.conda/envs/pressb/bin/python scripts/collect_parallel.py \
  --config configs/dataset_panel_randomized.json \
  --output datasets/piper_elevator_raw_panel_randomized_30hz \
  --gpus 4,6 --num-envs 2 --fps 30 --episodes-per-task 100
```

训练导出继续使用“初始状态到首次亮灯”规则。完整原始轨迹中的释放和归位仅用于物理验证及准备下一条，不进入最终 LeRobot 训练 episode：

```bash
PYTHONPATH=src .conda/envs/pressb/bin/python -m pressb.press_prefix \
  --raw datasets/piper_elevator_raw_panel_randomized_30hz \
  --output outputs/panel_randomized_press/cut_plan.json --episodes-per-floor 100

.conda/envs/lerobot/bin/python scripts/export_press_dataset.py \
  --raw datasets/piper_elevator_raw_panel_randomized_30hz \
  --cut-plan outputs/panel_randomized_press/cut_plan.json \
  --output datasets/piper_elevator_lerobot_panel_randomized_press_30hz \
  --parts datasets/piper_elevator_parts_panel_randomized_press_30hz \
  --episodes-per-task 100 --part-size 25 --workers 4
```

使用 `--config configs/scene.json` 可继续采集固定居中场景，但新代码须使用新的输出目录。已有数据、场景快照以及 H200 正在训练的固定面板数据集保持原样。

独立边界验证工具 `scripts/validate_panel_randomization.py` 依次测试 XY 四个边界组合，每个组合按遍 24–35 层。它保存 120 Hz 物理证据和指定时刻的双相机静态图，不是连续视频或训练数据；正式采集验证使用上面的普通采集器。

## 当前 ±2.5 cm 范围的验证

- CPU 检查覆盖四个 XY 边界组合和三种接近距离，共 12 个布局/轨迹参数组合、144 条完整按键轨迹；全部通过，最小完整面板画面余量 103.57 px。
- Isaac 使用间隙最紧的接近距离 **68 mm**，在四个 XY 边界位置执行全部 12 层：**48 次按压、48 次释放全部成功**，按键之间均归位，没有异常碰撞。进程正常退出（0）。
- 独立物理审计核对原始接触力、按钮行程、灯状态、归位及选定腕部橙灯图像，全部通过；按实际 USD 包围盒重新投影，完整面板画面边界余量至少 **103.59 px**。
- **33 项规划回归测试通过**。端点测试直接读取默认采集配置，同时检查接近距离扰动的两端，避免测试仍停留在历史 ±2 cm 范围。

证据见本轮汇总（本地产物：`outputs/panel_randomization/wider_25mm/summary.json`）、独立物理审计（本地产物：`outputs/panel_randomization/wider_25mm/corner_audit.json`）和靠近相机一侧的完整面板实拍（本地产物：`outputs/panel_randomization/wider_25mm/corners/corner_1/home_global.png`）。本轮只扩大默认范围并执行边界验证，没有启动新的 1200 条采集。

候选范围扫描结果见规划报告（本地产物：`outputs/panel_randomization_validation/wider_range_candidates.json`）。该报告保存扫描时的原始配置，并同时保留 ±2.4 cm 对照结果；实际测试偏移在每个 case 中记录，汇总按候选范围分别标记。

## 历史 ±2 cm 范围的验证

以下验证使用此前左右 ±2 cm、前后 ±1 cm 的配置，原始记录中保存了当时的范围；其中的采集/导出结果不代表重新采集了 ±2.5 cm 数据。

- CPU 检查覆盖 25 个 XY 网格位置、300 条全楼层轨迹，另检查接近距离极值和随机 episode。最小画面余量 103.84 px，接近距离扰动下最小相机支架保守间隙 19.61 mm。
- Isaac 四个边界位置共 **48 次按压、48 次释放全部通过**，每次按键之间均有实测归位。实际面板画面边界余量至少 103.85 px；独立核对原始接触力、按钮行程、灯状态与选定腕部图像。
- 普通采集器使用 **2 个并行环境、30 Hz** 采集每层一条，共 **12 条随机 XY episode**，全部物理审计通过。原始完整记录 6308 帧；首次亮灯裁剪后为 **2987 个训练帧**。
- 12 条均已实际导出 LeRobot 并通过完整视频解码和数值审计。官方读取器核对首/中/末共 36 个样本、72 张图像、36 个动作窗口和 12 次终点 padding，全部通过。每条 X/Y 布局元数据完整保留，state/action 维度不变。
- 双相机全运动同步抽查覆盖 episode 0、1，几何最佳滞后为 0 帧，亮灭反馈与物理标签同帧。此视觉同步结论只覆盖这两条抽查视频。

证据见验证汇总（本地产物：`outputs/panel_randomization/summary.json`）、四角实际相机图（本地产物：`outputs/panel_randomization/layouts.png`）、独立物理边界审计（本地产物：`outputs/panel_randomization/corner_audit.json`）、随机采集审计（本地产物：`outputs/panel_randomization/raw_audit.json`）及官方读取报告（本地产物：`outputs/panel_randomization/official_readback.json`）。小批训练数据位于 `outputs/panel_randomization/lerobot_preview_v1`；本次没有重新采集 1200 条正式数据。

回放会从导出清单恢复每条 X/Y 位置，并区分首次亮灯终止与历史完整动作终止。对应判定和来源检查已通过单元测试；本次物理验证使用采集规划控制，没有额外执行新数据的末端 action 回放。
