# PressB / VLA-JEPA 工作交接

## 2026-11-08 Playground 作为后续新任务的基础模板

- 用户明确：未来在此工作台添加其他物体形成新任务。沿用现有 `scene_config + scene_overrides`，新任务固定引用 `configs/calibration/playground_20261108_final.json`；桌面、PiPER、双相机标定、初始姿态、灯光和动力学统一继承，各任务自己定义物体、随机化、语言、奖励/终止。新标定应另存模板版本，再显式升级派生任务。
- 新增可运行节点示例 `configs/rl_nodes/playground_objects_example.json`：仅添加29mm红色ExampleCube及示例观测指令；仍为`playground`单环境、reward=0、无自动成功条件。完整新任务在`tasks/<task>/`实现，复用场景builder和共享runtime；没有复制新框架，也未修改已定版空桌配置。
- 文档说明dict递归合并/list整体替换、dynamic XY/table-Z与static base-XYZ差异、`run_playground.py`接收scene而非node JSON、活动episode需先`end_episode`再reset。README双语与框架说明已补入口。
- **40项既有CPU/USD测试通过**；真实Isaac读取该node示例，方块移动约72mm XY并物理落桌后，结束episode再reset恢复原位置（误差0）；双相机K/外参一致，模板/default SHA仍为`446f8eb0...34b21`，正常清理退出。验证目录 `outputs/playground_template_20261108/`。首次harness未先退休episode而被协议拒绝，日志保留，按正常API时序重跑通过；无生产代码修复、无真机操作。

## 2026-11-08 Playground 综合定版（仅离线与仿真）

- 最终预设 `configs/calibration/playground_20261108_final.json` 与当前默认 `configs/piper_playground.json` 完全相同，SHA256 `446f8eb0f264d15c37c7050e91319472b7f097d890c29ef8dc5fb3a963a34b21`。相较本轮前的配置只增加 `reconstruction.final_revision`，全部运行参数保留；不继续按单张图漂移参数。
- 综合选择：有效安装高度 **6 mm**、当前已配准global、上一轮等开度wrist外参修正、实机两路K/640×480、29 mm绿块、原几何与驱动摩擦；默认仍空桌面。6 mm是经验仿真修正，不是重新测得真机垫板厚度。
- 新完成6/8/10/14 mm四档联合外参/cube XY-yaw拟合及各自 **2146条原命令**完整PhysX回放。加上已核验SHA的当前baseline，共5组物理门槛全部通过，无非预期外部接触/绑定/瞬移。五组图像/深度等权工程分数baseline **0.64362**，新fit **0.68253/0.66751/0.65901/0.66200**，均无明确综合优势；mount-only8mm仅改善0.37%，低于10%选择容差。候选及所有失败/退化指标保留。
- 当前闭合双指竖直覆盖 **17.89/18.49 mm**，稳定抬升 **49.97 mm**，双指接触率1，放回XY偏移7.24mm。覆盖不是实际接触斑，真机35.6mm vs 仿真约30mm开度差仍在，不能称绝对标定完成。深度采用工作物体XY位置；改用base原点的外推深度会改变排名，已在独立审计披露。
- 唯一桌面调色候选双视角平均色差下降，但两处wrist局部退化+2.27/+2.02超冻结阈值+2.0，未采用；不应描述为颜色全面变差或夹爪轮廓变差。灯光、粗糙度、夹爪形状和动力学未因这轮图像拟合改变。
- **102项CPU/USD通过**；最终默认重新做空桌渲染及真实Isaac backend smoke，reset/step、时限、幂等重试、相机内参/FK与退出通过。未SSH/CAN、未真机运动、未commit/push。
- [便携结果](results/playground_final_20261108.json)，全部证据 `outputs/playground_final_20261108/`。最终比较 `media/real_final_1x.mp4` 为左真机/右最终仿真、上global下wrist，34–59s原速，复用已完成录像并明确自然开度差；`media/final_empty_preview.png` 是新的无参考物默认场景。历史所有记录保留。

## 2026-11-07 Wrist 夹爪轮廓对齐（仅相机外参、无真机操作）

- 用户指出腕部图像内夹爪位置不一致。已将默认 Playground wrist 光学外参微调 **0.707° / 1.633 mm**，保留实机 `fx/fy/cx/cy`、640×480、底座6mm、global相机和所有物理参数/动作。旧默认归档 `configs/calibration/playground_20261107_before_wrist_refine.json`；PAP 未改。
- 先分离开度问题：real闭合35.6mm vs sim自然接触30mm可导致约15px横移。本轮用相同实测35/40.1/50/59.4mm开度配准，未用camera掩盖该开度差；CPU审计排除double光心偏移、开度轴符号和STL/USD不一致。
- 原指身外缘最大偏19px，独立Isaac等宽渲染的四宽度轮廓RMSE **7.44→4.64px**，未入优化的40/50mm **6.06→4.91px**，最大偏差降到12px。仍有形状/材质差；20s静止cube验证RMSE5.70→7.62px有退化，不能声称全画面对齐。验证帧已用于候选筛选，并非盲测。
- 新脚本 `scripts/refine_playground_wrist.py` 使用原STL三角形逐行边缘（不能用凸包跨越指身凹轮廓），默认sigma5px、prior0.3°/0.5mm；版本化标注 `configs/calibration/playground_20261107_wrist_contours.json`。10项新增回归；综合 **102 passed**。只变camera+metadata，reproduced_final_v2与采用的数值逐位相同。
- 同2146动作完整物理重放前后 `physics.jsonl` **逐字相同**，49.974mm抬升/放回保持，无非预期外部碰撞。该调整只改变成像，不改变夹取物理过程。
- 便携结果 [wrist refinement](results/playground_wrist_refinement_20261107.json)。证据 `outputs/playground_wrist_refine_20261107/`；最终等宽图 `evidence/renderer_comparison_selected.png`，旧before_after是v1历史不要误用；视频 `video/real_wrist_before_after_1x.mp4` 左真机/中before/右after，原速34–59s，上global下wrist。

## 2026-11-07 按用户授权降低 Playground 仿真底座高度（无真机操作）

- 默认 `configs/piper_playground.json` 的有效安装高度 **14→6 mm**，改善同一原始轨迹的夹入深度。原 14 mm 完整存档于 `configs/calibration/playground_20261107_before_mount_depth.json`，10 mm 候选保留。此值是经验仿真修正，不能称真机垫板实测 6 mm；原用户估测 14 mm 仍作为历史依据。
- 14/10/6 mm 三组完整 2146 条原始关节/夹爪命令、阶段、帧率保持逐值相同。桌面、物体、全局/overview 相机、灯光/地板世界位置不动；仅对固定物的 base 坐标重表达，wrist link-local 外参及两相机实机内参不改。默认仍为空桌面、旧初始姿态，其他任务配置不改。
- 闭合双指竖直覆盖分别 9.90/10.51、13.89/14.49、17.89/18.49 mm。6 mm 时最低指尖离桌 10.51 mm，稳定抬升 49.974 mm、双指接触率 100%，放回偏移 7.24 mm，无非预期外部碰撞。覆盖是几何包络，不是精确接触斑。
- 图像存在权衡：6 mm 改善接触前 green 侧面 RMSE 20.42→12.72 px；10 mm 的多帧总体图像误差更低。横向偏心约 13.46 mm、模拟约 30 mm 与真机 35.6 mm 夹持开度差仍未解决。不能宣称完整标定完成或所有图像都改善。
- 新增纯 CPU `scripts/adjust_playground_mount.py` 保持房间世界坐标生成候选，9 项新增语义测试；综合 **92 passed/0 skipped**，新默认真实 Isaac reset/step/相机/退出 smoke 通过。两组新物理回放完整通过。
- 便携结果 [mount depth](results/playground_mount_depth_20261107.json)。证据 `outputs/playground_mount_height_20261107/`；四列原速视频 `video/real_mount_14_10_6mm_1x.mp4`，左真机/14/10/6，上global下wrist。本轮未 SSH、未发送真实机械臂动作。
- 注意只用原 plan 做高度对照，不叠加上一轮 deeper_8mm 新动作。旧 plan 的高度元数据描述原14mm场景，以本次config计算桌上高度；新生成 table-relative IK 会补偿mount变化。

## 2026-11-07 夹持深度复核与下移 8 mm 仿真对照（无真机执行）

- 用户指出仍夹得浅。实际 STL＋物体姿态审计确认：fit04 原轨迹闭合时两指最低离桌 19.11/18.50 mm，与 29 mm 方块竖直包络只重叠 9.90/10.51 mm，约 34–36%。此前优化的是视觉对齐，并未修正该物理夹入深度。包络重叠不是接触斑面积，日志缺少独立两指位置及接触点，采用对称手指假设。
- 仅仿真做下移 8 mm 对照：桌上 TCP 20→12 mm，保持 fit04 场景、XY/姿态目标、夹爪命令、2146 帧和阶段时序，改变 999 条手臂命令，未真机执行。闭合重叠增至 17.89/18.49 mm（62–64%），最低离桌 10.51 mm；自然 PhysX 稳定抬升 49.978 mm、放回，无非预期外部接触。仍约 30 mm 夹持开度，且旧冻结路径与校准后的物体存在约 14.16 mm 工具 X 方向偏心。该实验不是孪生标定完成，默认场景不改。
- 修复未来新生成轨迹的坐标语义：新增 `--grasp-height-above-table`（默认 20 mm），正确减去安装垫高；显式 `--grasp-height/--grasp-height-base` 保留旧 base-Z 语义，`--plan-input` 不改历史动作。原 retry08 轨迹已正确使用 baseZ=6 mm，不能把新发现的生成器问题解释为旧视频差异。83 项相关 CPU/USD 测试通过。
- 8 顶点投影审计纠正视觉报告中的面标签：绿色可见四边形是近侧竖直面，不能直接称“顶面”；原像素误差仍有效，但旧面语义撤回。默认相机、垫高和几何未因这一诊断继续拟合。
- 便携记录 [夹持深度诊断](results/playground_contact_depth_20261107.json)。原始几何审计在 `retry_08/contact_depth_review/`；新仿真、计划及 25 秒左真机/中20mm/右12mm视频在 `retry_08/calibration_optimization/contact_depth_review/`。右列持续注明新动作未真机执行；执行计划继承旧高度/速度元信息的勘误有单独记录，原始证据未覆盖。

## 2026-11-07 最新录像驱动的 Playground 校准修订（仅离线）

- 根据 retry_08 的完整真机录像、实测关节反馈和双 RGB-D，采用联合图像校准 `fit04`，更新默认 `configs/piper_playground.json` 的全局/腕部外参与参考方块位置，正式采用用户估计的 14 mm 安装垫高。保持实机内参、29 mm 绿块、旧默认起始姿态、机器人模型、摩擦与驱动不变；默认仍 `objects=[]`，其他任务配置不变。旧默认完整存档于 `configs/calibration/playground_20261107_before_retry08_video.json`。
- 相对本轮原仿真，绿块参考 XY 修正约 `[-15.55,+1.47]` mm。优化器外的同次采集验证：20 秒腕部角点坐标 RMSE **30.96→8.71 px**，20/36 秒全局指尖 **13.38→4.10 px**；41 秒绿块边缘相对指尖误差约 **63→21 px**。这些画面开发时已查看，不是盲测；全局静态方块点约 0.98→1.15 px 有小幅退化。全局裸桌深度参与拟合，腕部深度仅验证，近场仍有残差。
- 原 **2146 条动作**逐值保持，新的完整自然 PhysX 回放稳定抬升 **50.019 mm**、保持双指接触后放回，无绑定/瞬移或非预期外部碰撞。新夹持开度约 **30.018 mm**，真机 **35.6 mm**，仍差 **5.58 mm**；不能称接触几何已准确对齐。方块自然侧移后回桌距初始点约 **7.20 mm**。物理成功不是优化目标，也不是新真机实验。
- 独立 CLI 可重现全部 18 个参数（最大差 0），内置纯 NumPy STL 读取、输入指纹、输出防覆盖；`scripts/calibrate_playground_video.py` 配合 `configs/calibration/playground_20261107_retry08_video_manifest_v04.json`。63 项相关 CPU/USD 测试无跳过通过；最终默认空桌 Isaac 接口 smoke 通过 reset/step、相机/FK、超时、重试幂等和退出清理。
- 本轮**没有 SSH/CAN、复位、使能或真机运动**；实机状态仍以下方最后一次真实记录为准。旧录像、日志、轨迹均保留。便携结果见 [录像校准记录](results/playground_video_calibration_20261107.json)，三列原速对比与全部验证在 `outputs/playground_grasp_transfer_20261107/retry_08/calibration_optimization/`。

## 2026-11-07 Reference 驱动对齐：真机完整抓取与放回成功

- 按用户要求对齐原 PiPER client 的执行规则。`reference` 队列与硬件策略不加逐点到位等待、5 秒软到位超时、接触阶段门控、实测关节角启动拦截或额外软件 slew；使用合法目标和设备 MOVEJ 插补。原目标/单步限幅、反馈故障与人工停止保留；旧 `tracked/projected` 行为可显式复现。reference 正常结束不发新目标或失能，不把 dispatch 完成当任务成功。
- 475 项相关测试通过（含 27 项实际 adapter＋fake SDK 集成），源码备份后已部署到 ssh piper。本轮未发送复位。用户确认现场未压桌/卡住、人员离开后执行。原 J5=76.276° 的 2500 帧恢复候选因现场摆放后姿态已改变，在使能前起点核对被拒绝，0 动作；没有使用旧轨迹。读取新姿态后改用 900 帧 PAP home 目标、设备 1%，30.812 秒完成，最终最大关节误差 0.070°。
- 回位后重新定位绿色 29 mm 方块：底座 XY `[0.329556,0.084877]` m，朝向 25.75°。沿用估计 14 mm 安装高度、TCP 桌面上 20 mm（base Z=6 mm）、实测相机内参及已有外参；全局/腕部 XY 仍有 22.57 mm 差异，不能宣称标定完成。
- 新 PhysX 完整轨迹 2146 帧、71.533 秒，方块稳定抬升约 49.970 mm 后放回，未检测到非预期外部机器人接触。真机采用同一目标序列、30 Hz、设备 2%、effort 1000；2146/2146 动作、72.378 秒完成，包含张爪和撤退，无执行错误或保护停止。
- **真实抓取结果由连续双相机画面核实：绿色方块离开桌面、随夹爪抬升并下降，松开后留在桌面。** 固定全局画面绿色底缘上移约 32 px、再回到桌面位置，蓝色方块保持不动；不把这一像素位移转换为未实测的 5 cm 真实升高。执行器本身仍记录 `task_success=null`，视觉结论单独保存；本轮仅一次成功试验，不是成功率评测。
- 最新只读反馈 `after_grasp_feedback.json`：`arm_status=0,error_code=0`，六电机 enabled，关节 `[14.365,102.489,-64.352,0,51.890,-11.285]`°、夹爪 59.4 mm；后置 11.48 秒采集关节变化范围全为 0。**机械臂当前保持撤退后的姿态，未急停、未再回位或追加抓取。** 本轮运动/双相机/CAN/只读采集进程均已退出。下方急停及 J5 超界记录均为历史。
- 便携记录 [reference 抓取结果](results/piper_reference_grasp_retry_20261107.json)；原始 `outputs/playground_grasp_transfer_20261107/retry_08/`，含独立视觉审核、执行审核、双相机原帧、PhysX、localization、原速对照视频。SDK 禁用本地 loopback，passive CAN 仅捕获反馈，不能用未见命令帧证明未发送命令。

## 2026-11-07 再次抓取请求：单次复位完成，等待现场 J5 回限位

- 用户要求重新测试抓取。只读100条反馈和双 D435i RGB-D确认仍急停、J5=76.258°，绿色方块被夹爪遮挡；探针CAN TX增量0。用户随后明确现场支撑、授权仅复位，已执行 `reset_once.py` 一次：`0x150/0200000000000000`，send_attempts=1、send_bytes=16，无重试/使能/运动。
- 复位后100条新鲜只读反馈（TX增量0）：`arm_status=0, ctrl_mode=0, error_code=0`、六电机均disabled、无故障位，关节 `[12.371,136.323,-107.140,0.367,76.276,7.565]`°、夹爪35.1mm。**急停已清除，不需要再次复位；J5仍超过已读设备70°上限。** 已请求现场支撑并使用熟悉方式恢复J5到界内、稳定放置且人员离开后告知。收到回复后必须先读新姿态，尚未执行回位/抓取。
- 用户质疑为何仍要求手动J5恢复。已再次核对：原client只做目标限幅（J5±75°）和单步delta，原bus不因实测角超界拒绝启动；当前阻止的是PressB适配器额外的 `_check_joint_feedback_bounds`。查询到70°仅证明配置范围，**不能证明固件拒绝向内回位**。此前“因此必须手动调整”的推断撤回，尚未修改控制代码或评估/执行新的向内恢复路径；不能将本次解释视为运动授权。当前仍以复位后disabled状态为最近反馈。
- 新执行策略为下节已部署的 `reference`。恢复后需要新鲜姿态和物体定位，再生成/仿真验证抓取；不要直接运行旧1188帧计划。原始证据 `outputs/playground_grasp_transfer_20261107/retry_07/`，远端 `playground_retry_09_before/`、`playground_retry_09_reset/`。旧轮次急停/姿态记录均为历史。

## 2026-11-07 按原 PiPER 客户端对齐动作发送：已部署，仅软件验证

- 用户明确要求参考 `lerobot-piper/piper_scripts/piper_inference`，不要自加执行规则。有限回放新增 `reference` 调度：按配置频率逐条发送，使用原目标限幅和相对实测值的单步限幅，块末重复末目标并定时等待 0.5 秒；不增加逐点到位、5 秒软到位超时或接触阶段门控。之前拟议的自定义 streaming/阶段等待方案已撤回归档。已生成轨迹不额外平滑。
- CLI 未显式指定模式时默认 `reference`；尊重计划中的显式模式，`--execution-mode` 可覆盖。库层旧计划默认仍为 `tracked`，供复现旧运行。`reference_chunk_size=null` 表示整条有限轨迹作为一块；显式 32 对应原在线客户端默认块长。此次只对齐执行队列规则，不声称完整推理/UI/SDK 生命周期等价；底层限位、速度约束、反馈/故障、起点核对、人工停止和 watchdog 保留。
- 429 项相关测试通过；原脚本完整 `control_loop` 的四组队列对照，发送时刻差 0、目标最大差 `1.11e-16` SI，包含块末最后一帧。实际驱动＋动态假 MCU 完成 1188/1188 条发送，无停止；`success_meaning=dispatch_only`，`target_reached/task_success=null`，不能据此宣称真机到位或抓取成功。
- 四个文件已同步到 `ssh piper` 的现有工作目录，旧源码备份于 `software_updates/reference_dispatch_20261107_220846/before_source/`，硬件适配器源码未改。远端 dry run 为 `reference`、1188 actions、`executed=false`；CAN TX 前后均 84935。**本轮没有连接硬件执行、复位、使能或运动。** 归档计划仅用于检查，不可当作当前起点有效的运动计划。
- 最近实际抓取仍是下方的失败试验，最近只读状态为急停、J5=76.258°，本轮未重新读关节或解除急停。下方“释放修复未部署”描述的是该历史时点；当前 tracked 修复随本次代码同步，但未真机验证，reference 本身不使用这些门控。
- [便携验证记录](results/piper_reference_execution_20261107.json)；原始证据 `outputs/reference_execution_20261107/`（原脚本来源、独立对照、假 MCU、部署和远端 dry run）。

## 2026-11-07 Playground 14 mm 修正后真机重试：夹住但未抬起，释放超时停止（历史失败试验）

- 用户授权现场准备完成后，仅执行一次回位与抓取。回位 605/605 动作、57.268 秒成功；重新采集双 RGB-D，绿色 29 mm 方块 XY 为 `[0.421168,0.093492]` m。沿用 14 mm 安装高度候选、实测双相机内参及原 wrist 外参。跨视角 XY 仍相差约 18–28 mm，不能称绝对标定完成。
- 新 1,188 帧完整 PhysX 轨迹成功，稳定方块抬升 49.965 mm，无非预期外部接触。规划抓取 TCP 为桌面上 20 mm，即 base Z=6 mm。**真机仍失败：用户现场确认“夹住了，但没夹起来”。** 日志计算的约 49.217 mm 仅是 URDF/FK 末端位移，夹持负载与 35.1 mm 开度均不能替代物体升空证据。
- 真机设备速度 2%、effort 1000 未增加，完成 939/1188 动作，用时 55.828 秒。首次 release 目标 28.167 mm 小于实测接触开度 35.1 mm，6.933 mm 误差超过 5 mm 软到位阈值，程序不推进到后续张开目标，约 5 秒后请求软件急停。未完成释放／撤退；这是 PressB 回放层的控制问题，非已报告的电机故障，与未夹起的任务失败分开记录。
- **最近只读状态仍为急停：** `arm_status=1, ctrl_mode=0, error_code=0`，关节 `[12.518,136.298,-107.047,0.301,76.258,7.556]`°，夹爪 35.1 mm。停止后 CAN 显示姿态变化，J5 约 12.61 秒后超过 70°，最终 76.258°。使能位不证明保持力；未确定姿态变化的物理原因。停止后未复位或追加运动，所有采集／回放进程结束。下方无急停／home 状态均为历史。
- 释放死锁已在本机修复，**未部署、未真机验证**：仅对已获接触的初始张开段允许有界软到位过渡，锚定最近可信接触开度，越过开度即永久退出，跨动作累计原 5 秒限时；硬限制、故障／时效、arm 跟踪等保持。401 项测试通过；同一真实驱动＋动态假 MCU、相同原计划，归档旧代码精确复现 939/1188 超时，新版 1188/1188 完成。此软件修复不代表物理抓取已解决。
- [便携记录](results/playground_mount14_grasp_retry_20261107.json)；原始证据 `outputs/playground_grasp_transfer_20261107/retry_06/`。`video/close_lift_release_stop_detail_1x.mp4` 为 35.83 秒关键过程；完整真机和真机／仿真对照均标明失败，并保留停止后 15 秒实际画面，各自原速播放。已执行源码原样存于 `executed_source/`，不得用修复后源码覆盖。

## 2026-11-07 Playground 安装高度诊断：14 mm 候选，已完成两轮仿真

- 用户补充桌面到机械臂安装底面约 **14 mm**，不含机械臂底座自身高度。旧代码把 `base_link` 和桌面设为同高，URDF/USD 没有隐藏安装偏移；可见安装板 RGB-D 高程约 11–13 mm，仅支持非零垫高，不是精确基座标定。
- 共用 workbench 新增 `robot_base_height_above_table_m`，默认 0；Playground 动态物体仍落在桌面。新增可直接加载的空桌候选 `configs/calibration/playground_20261107_mount14mm_candidate.json`，补偿固定相机、灯光、地板的 base-relative Z 以保持世界位置；腕部实测焦距和当前 link6 外参均不改。默认配置仍保留，历史抓取记录不改写。
- 相同 29 mm 方块、XY、姿态和物理参数下，14 mm 场景重放原 1177 帧关节计划：两指接触 0、抬升 0。将抓取 TCP 桌上 20 mm 正确换算为 base Z=6 mm 后重解 IK，1180 帧完整 PhysX 成功，稳定抬升 **49.961 mm**，放回偏差 **0.117 mm**，无非预期外部接触，最大 J5 **63.572°**。这些是仿真结果，未执行新真机抓取。
- 在回位后同一张腕部图像上，仅加入 14 mm 的投影敏感性检查：纵向误差 20.55→6.44 px，横向仍约 25.23 px。此前 21–28 mm 是跨视角 XY 重建差，不是已测定的腕部外参平移；安装高度可解释夹浅和部分投影偏差，手眼/FK/指爪几何仍需独立校核。
- Playground 37 项、独立场景回归 88 项、根代理 USD/相机/物体回归 51 项、轨迹与序列化 11 项通过，套件有重叠不相加。独立检查 37 个固定场景世界变换。顺带修正仿真失败结果 `numpy.bool_` 无法写 JSON；第一轮失败报告由完整物理日志重建并注明来源，第二轮正常落盘。不能把 Isaac 退出码 0 当成成功。
- 便携记录 [安装高度诊断](results/playground_base_height_20261107.json)；原始 `outputs/playground_grasp_transfer_20261107/retry_05/base_height_14mm/`。本轮无 SSH/CAN 操作、无新运动/复位，最近真机状态仍以下方完整抓取停止后的记录为准。安装板/夹具仍为视觉件，不模拟其碰撞。

## 2026-11-07 Playground 完整抓取实测：起抬失载，无急停（21:25 后最新状态）

- 低速回位已完成，新全局 RGB-D 定位绿色 29 mm 方块 XY `[0.419486,0.091522]` m。新位置使用固定径向倾斜 5°工具姿态、60 mm预抓、20 mm抓取、50 mm抬升，维持现有限位。完整1177帧PhysX通过，稳定抬升49.964 mm，无桌面/蓝块非预期接触。
- 已将完整轨迹提交一次真机执行，设备2%速度、gripper effort仍1000。39.66秒在34.4 mm开度/raw effort约−1040取得显式接触判定；41.65秒开始lift后负载和开度快速下降，41.964秒以`grasp_lost`保持中止，完成726/1177。用户现场确认方块在桌面，录像显示相对指尖转动/偏移；本次仍未完成稳定抓取。
- TCP按实际关节与URDF计算仅抬升约5.63 mm，不等于方块升空高度。无急停、无复位、无后续自动运动；目前保持early-lift姿态，六电机enabled，status/error均0，关节`[12.305,120.488,-88.342,0,57.723,16.336]`°，夹爪27.8 mm。停止后CAN及100条只读反馈均未见关节下垂。
- 新可选contact-aware replay保留默认position模式；close/carry需新鲜负载和开度证据，任务失载hold保留原clamp目标，hardware/control故障仍stop。独立actual-adapter/fake-MCU验证通过；本机相关402+独立helper67项测试通过。新遥测接口已真机只读20样本/发送增量0验证。
- 便携结果 [完整抓取试验](results/playground_complete_grasp_20261107.json)；原始 `outputs/playground_grasp_transfer_20261107/retry_05/`。`video/real_vs_sim_grasp_1x.mp4`为左右真机/仿真、上下global/wrist独立1×，仿真先结束明确冻结；`close_lift_detail_1x.mp4`保留失载后5秒。
- 用户随后提出真实 base 可能垫高及夹取过浅；离线诊断及 14 mm 候选仿真已完成，见上方最新节。约 21–28 mm 跨相机位置差尚不能归因于单一 wrist 外参。未改已测相机内参、未追加真机运动。下方急停状态或 home 状态均是历史。


## 2026-11-07 Playground 完整抓取准备：复位与回位已完成（21:18 后）

- 用户现场确认上一轮 20 mm 下移试验实际夹住了方块，但没有抬升；此为现场观察，独立于录像和遥测结论。此前“闭合失败”应理解为位置到位判定超时，不能据此否定现场夹持。
- 用户支撑机械臂并授权仅复位后，已发送一次复位，随后六电机未使能、status/error 均为 0，J5 回到 42.682°。用户再次确认稳定放置且人员离开后，执行先竖直抬升约 30 mm、再回 PAP 起始位姿的低速回位；646/646 动作完成，用时 61.268 秒，无急停或错误。
- 最新只读状态：`[-1.771,25.553,-25.260,-4.769,40.462,5.038]`°，夹爪 29.7 mm，六电机均 enabled，status/error 均为 0，机械臂停在初始位姿。原始记录 `outputs/playground_grasp_transfer_20261107/retry_05/`；下方“当前急停仍锁存”均为历史状态。
- 已采集回位后的双 D435i RGB-D，绿色方块位置已发生变化，不能沿用旧目标。正在准备显式有界接触夹持控制及新完整轨迹；尚未执行本轮抓取。默认空桌 Playground 与相机实测内参保持。


## 2026-11-07 Playground 下移 5 mm 抓取：闭合超时，急停后姿态变化（19:38 最新状态）

- 按用户“夹的时候下移多一点”，`retry_04` 将计划抓取 TCP 高度从 25 mm 降至 20 mm，绿色方块采用实测 29 mm；XY 与最终闭合目标 28 mm 保持。833 帧 PhysX 完整计划成功，稳定抬升约 49.9768 mm。真实低速分段下降至计划 25／22.85／20 mm 均完成，20 mm 处新鲜关节反馈经 URDF 计算 TCP 为 20.059 mm；这些是运动学高度，不是实测指尖桌面间隙。
- **真实闭合阶段失败，尚未进入抬升。** `playground_retry_06_grasp` 用时 9.46865 秒，仅完成 90/656 条动作，在局部索引 90（原计划帧 267）等待：目标夹爪开度 29.6667 mm，反馈停在 34.8 mm，误差 5.1333 mm 超过既有 5 mm 到位阈值，约 5 秒后触发 tracking timeout，软件发送急停。夹爪反馈阻塞和较高 effort 只支持受阻判断，不证明双指正确夹住方块或完成抓取。
- **当前急停仍锁存，停止后机械臂姿态发生变化，未保持原抓取姿态、未撤退。** `after_stop_feedback.json`（19:38:16，100 样本、TX 增量 0）为 `[-1.666,125.664,-84.364,-0.342,76.284,-1.528]`°，夹爪 34.8 mm，`ctrl_mode=0, arm_status=1, error_code=0`，六电机 enable 位仍 true、无驱动故障位。J5 已超过 70°上限；使能位不能解释为保持力已验证。停止后没有复位或追加运动，后续运动前需要现场处理、确认复位准备并重新读取姿态。下方 `retry_03` 的“无急停／已撤退”是上一轮历史状态。
- 新记录 [下移抓取结果](results/playground_deeper_grasp_20261107.json)，原始证据 `outputs/playground_grasp_transfer_20261107/retry_04/`。独立闭合及停止后变化分析、含停止后画面的视频随该记录保存。此前配置与试验结果保持原样；默认 Playground 和腕部实测内参不变。
- 闭合审计 `retry_06_gripper_contact_audit.json`：5–8 秒区间 600 条新鲜 CAN 反馈的开度恒为 34.8 mm，effort raw 中位数 −1043，支持明显闭合阻力；该值未经力／力矩标定，不能区分方块接触、卡阻或正确双指抓取。主视频 `retry_04/video/real_deeper20_with_poststop_1x.mp4` 共 51.433 秒，各阶段真实 1×，包含停止后 15 秒实际姿态变化；阶段之间等待省略，不宣称相机硬同步。

## 2026-11-07 Playground 重新抓取：执行完成，真实方块未抬起（19:28 后更新）

- 用户确认现场清空并授权低速回位及抓取，已完成 `retry_03` 的四个真实阶段：回位 1/1 动作、16.694 秒；预抓取 615/615、52.039 秒；下降到 52.5 mm 计划 TCP 高度 99/99、5.511 秒；连续下降／闭合／抬升／放下／张开／撤退 710/710、40.631 秒。四个执行器均 `success=true`，无错误或保护停止。回位中 `native_startup_recovery.check_inward_progress=false` 已实际运行并完成；下方“仅部署、未运动”是此前状态。
- **执行器成功不等于抓取成功：本次真实抓取失败。** `grasp_live_03.jpg`、`grasp_live_04.jpg` 可见绿色方块仍在桌上，未随夹爪抬起。PhysX `sim_trial_02` 使用同一规划抓取高度 25 mm、闭合目标 28 mm，方块抬升约 49.975 mm并稳定保持 1.5 秒；真机控制器会按反馈重新定时，不能声称两次已执行命令的时间轴相同。具体偏差来源仍待定位，不能仅由执行器通过推断接触几何正确。
- 最新只读反馈 `after_grasp_feedback.json`（19:28:51，100 样本、TX 增量 0）：关节 `[-1.666, 101.206, -71.174, 0.000, 65.043, -1.777]`°、夹爪 59.4 mm；status0/error0，六电机 enabled，无驱动故障。机械臂保持在撤退后的计划 TCP 高度 80 mm 姿态，无急停；没有随后复位或追加运动。80 mm 为运动学计划值，不是实测桌面间隙。
- 默认 Playground 仍为空桌面，腕部相机实测内参保持不变；本次方块及抓取计划为独立试验配置。证据位于 `outputs/playground_grasp_transfer_20261107/retry_03/`，便携记录见 [抓取重试结果](results/playground_grasp_retry_20261107.json)，说明见 [playground.md](playground.md#authorized-grasp-retry-2026-11-07)。此前失败、复位和标定证据保持原样。
- 视频 `retry_03/video/real_full_retry_1x.mp4` 为 131.933 秒：各动作阶段按真实 1× 播放，省略阶段之间的长等待；`real_vs_sim_grasp_1x.mp4` 为 43.633 秒，真实 40.631 秒／仿真对应段 23.667 秒各按自己的 1× 播放，较短侧明确标注冻结。两者不是硬件同步或同时间轴执行；关键帧已复核，真实方块仍在桌面。回放、相机及 CAN 采集进程均已结束。
- 初步离线核对：相同动作索引的实测关节经同一 URDF 计算，闭合末端真／仿 TCP 差约 1.10 mm、抬升保持差约 0.30 mm；腕部画面中方块相对指尖仍有约 30–40 px 的目测差异（非三维测量）。证据不支持把明显关节跟踪错误当作主要解释，但无法区分物体前后／高度、指爪几何、零位或相机标定等因素，也不能据此归因摩擦。此前二维间隙检查不足以验证三维抓取接触。见 `retry_03/geometric_failure_audit/report.json`。
- 用户随后提供方块边长 **29.0 mm**；本次 `retry_03/sim_trial_02` 实际配置为 **29.0734578 mm**，差 0.07346 mm（相对实测约 0.253%）。较早原始采集的 30.16 mm 拟合值不适用于本次配置。原配置保留；独立 `sim_trial_measured_29mm` 仅改绿色方块三轴边长，809 条动作、初始状态、阶段时序及物理参数保持一致，仍成功抬升 49.9755 mm、稳定双指接触 100%、回放落点偏移 0.0072 mm，无非预期外部接触。该微小尺寸修正不能解释当前仿真中的成败差异，但不验证真实接触几何；受控复测未连续录制视频，物理步进和静态截图仍保留。见其 `comparison.json`。

## 2026-11-07 PiPER 启动单调检查可选关闭：已部署，未执行真机运动

- 用户要求停用启动检查。新增 `native_startup_recovery.check_inward_progress`（默认true），当前准备的回位方案显式false。只关闭native阶段“恢复反馈边界持续收缩”和“反馈反向超过slack停止”两项；初始有效J2/J3改用已有配置、最高3°的固定反馈边界。起点匹配、使能阶段检查、合法目标、1%设备速度、4°偏移、5°/秒观测速度、其他轴/夹爪漂移、故障/超时/急停/稳定到位均保留。退出即恢复普通检查，不重新开放已退休的恢复例外。
- **371项相关测试通过**，包含记录的J2 −0.633→−1.676°反馈点以及关闭策略后的各类保留保护。记录点测试只验证位置规则接受后主动中止，不能说真实回位成功，也未解释反馈跳变；独立速度检查仍可能触发。
- 两个源码文件已同步到 `ssh piper` 的现有部署，hash一致。新方案 `playground_home_bounded_policy_01.json`（仅回位、夹爪保持27.4mm）已在本机和远端通过相同dry run，`executed=false`，CAN TX增量0。本轮未使能/复位/运动；执行前仍须读取当前实际状态并刷新方案起点。旧方案与源码备份保留。
- 证据 `outputs/playground_grasp_transfer_20261107/retry_02/startup_policy_update_01/`；便携记录 `docs/results/piper_startup_policy_20261107.json`；说明 `docs/real_robot.md`。下方现场状态为最近一次只读采集结果，不能代替后续新鲜反馈。

## 2026-11-07 Playground 最新恢复状态：急停已解除，J2/J3 仍未确认回到界内

- 用户再次明确支撑并仅授权复位，已执行 `retry_02/playground_retry_03_reset` 一次；随后全部操作只有只读反馈和双相机采集。当前 status0/error0、六电机未使能，不需要再次复位。下方“急停再次锁存”为复位前历史状态。
- 用户确认调整后两次补读，最新角度 `[0.784,-1.672,2.428,0.000,21.620,-2.563]°`，夹爪27.4mm；J1/J4/J5/J6已有变化，J2/J3仍与复位后相同。最新100样本probe TX增量0；被动CAN也记录了0x251–256电机反馈，未发送命令。尚未重新使能、回位或抓取，不能重复相同边界启动。
- SDK时间戳仅为CAN接收时间，重复位置广播也会使其增长，不能据此断定操作者没有移动或编码器必定刷新。J2是肩部俯仰、J3是长臂末端肘关节；已请求确认实际调整部位/方式。需先获得与实际姿态一致且明确在界内的反馈，再规划普通低速回位；未放宽保护阈值。
- 证据根目录 `outputs/playground_grasp_transfer_20261107/retry_02/`：`latest_recovery_status.json`、`recovered_feedback_05.json`、`disabled_raw_feedback_01.json`、`playground_retry_04_before/`。采集已退出，无本轮运动进程。

## 2026-11-07 Playground 再次回位启动受保护停止：未进入抓取

- 现场再次调整后，fresh反馈为 `[4.633,-0.633,2.336,-0.353,20.723,2.725]°`，夹爪27.4mm，stop已解除、六电机未使能。J5已在界内；J2/J3在既有3°启动恢复范围。只采用已验证native J2/J3恢复参数，目标+1°/-1°，然后PAP原位，夹爪保持不动；没有扩大限位/保护阈值。
- 实际执行 `retry_02/playground_retry_02_home` 在 **0.3023秒** 保护停止：原始CAN在+.289741s单包J2从−0.633→−1.676°，超过0.25°允许反向变化；J5同周期20.952→21.499°。只收到1条native snapshot，普通回位actions=0，未抓取。驱动使能反馈+.1606至+.2508s出现；被动CAN没有记录TX，软件target正确不能替代线缆命令证明。10Hz图像未见明显视角变化，但不能判定是编码器更新还是机械动态，原因未定。
- **当前急停再次锁存**：停止后100样本 `[4.633,-1.676,2.336,-0.353,21.499,2.725]°`，status1/error0/六电机enabled仍true；只读probe TX增量0。没有再复位/重试。不要反复执行相同边界起步；下一步需现场恢复到J2/J3明显在限位内的工作姿态，fresh读数验证后再规划。复位会失能，仍需新的现场支撑授权。
- 证据在 `outputs/playground_grasp_transfer_20261107/retry_02/`：`home_failure_audit.json`、`playground_retry_02_home_can/`、`playground_retry_02_home_capture/`、`home_after_stop_feedback.json`。录像和被动采集已正常关闭。下方“已仅复位”状态已被本次停止覆盖。

## 2026-11-07 Playground 抓取重试前恢复：已仅复位，等待现场 J5 回限位

- 用户要求回原位后再抓，并确认现场已支撑、仅授权一次复位。已发送一次 `0x150/0200000000000000`；后续只读100样本TX增量0，急停状态0、错误0，六电机均未使能。未执行任何回位/抓取或使能命令。证据 `outputs/playground_grasp_transfer_20261107/retry_02/playground_retry_02_reset/report.json`、`after_reset_feedback.json`。
- 最新J5 **76.307°**，仍高于本次验证上限70°；已请求现场通过熟悉的示教/维护方式恢复到限位内、稳定放置并离开运动范围后告知。现有native恢复只覆盖J2/J3小越界，不能放宽J5限位绕过。收到确认后先读取新鲜姿态，再建回位计划；不要自动再复位。绿色方块已移位，重抓前必须重新定位，旧抓取计划不能直接重放。此前“急停仍锁存”记录均为历史状态。

## 2026-11-07 Playground 根据真实失败轨迹校准：视觉改善，接触仍未闭合

- 当前默认 `configs/piper_playground.json` 已采用保守多姿态视觉修正。旧配置完整保存于 `configs/calibration/playground_20261107_before_multiview.json`。默认仍为空桌；蓝绿方块只作显式历史标定参考，真实绿色方块后来已移动。PAP、电梯、table_buttons 配置未修改。
- 保留两台真实 D435i 的 `fx/fy/cx/cy`。只优化腕部外参、历史参考方块几何并相应重新注册全局相机。腕部光心改变约6.82mm、旋转约0.69°；绿色方块中心改动X+10.37/Y−6.91mm，拟合边长30.16mm。没有改URDF、关节零偏/限位、TCP、桌面高度、摩擦或驱动，也没有复位/使能/真实动作。
- 每角点二维欧氏RMSE：连续预留5姿态21.47→5.72px；独立标注6姿态29.07→6.82px；冻结候选后新标下降3姿态/11角点51.35→19.07px。前两组用于候选审阅，不是完全盲测。个别早期帧变差，指尖纵向误差约1→8px，下降越接近方块剩余误差越大；绝不能声称毫米级绝对精度或抓取已可靠。
- 已做同 `applied` 指令/原始时间轴的两版真实PhysX复验，共各3200帧/106.667秒，无物体绑定/瞬移；下降阶段两版均零物体接触。真实首次可辨物体位移在下降开始后6.867秒，仍张开59.4mm；仿真旧11.057秒/新9.724秒才首次接触，均已闭夹。**校准仍未解释下降碰边。** 具体实际接触体未从视频确认；需继续实测真实指套/夹爪开度几何、绝对关节零位、底座与桌面关系。
- 同实测q的零物理步截图单独用于视觉对照；与 `applied` 物理回放严格区分。新增工具 `scripts/calibrate_playground_multiview.py`、`scripts/render_playground_alignment.py` 及实际指令重定时支持；47项相关CPU/USD测试通过。原/新PhysX进程均退出0，GPU已释放。
- 新只读证据 `playground_post_contact_calibration_01/` 显示姿态和绿色方块位置已变化（J5约76.228°，超出配置+70°上界）。独立SDK/控制器FK接收探针TX增量0，SDK offset=1与URDF一致，不支持简单DH模式搞错的解释，也不证明真实绝对零位准确。后续不要沿用下方旧状态、不要自动复位或直接继续旧动作；仍需新鲜状态核对。近距离腕部深度失效，未用于拟合。
- 便携结果：`docs/results/playground_calibration_20261107.json`；详细证据：`outputs/playground_grasp_transfer_20261107/calibration_revision_01/selected_partial/`、`calibration_diagnosis/applied_physics_comparison_01.json`；对比图：`calibration_comparison/geometry_comparison_pregrasp.jpg` 和 `geometry_comparison_three_poses.jpg`。完整方法/复现见 `docs/playground.md`。以下初始构建与实测记录保留为历史。

## 2026-11-07 Playground 抓取迁移实测：真机闭夹超时，急停仍锁存

- **用户随后现场纠正：夹爪向下移动时先碰到了方块边缘，真实没有夹起。** 因此必须区分“下降碰边”的物理失败与稍后“闭夹到位超时”的控制停止原因；之前将接触描述为已经夹住不准确。当前在离线核对录像与标定，未再操作硬件。不能仅放宽闭夹容差后重试，需先解决下降对齐与实际夹爪几何/坐标标定。
- 用户授权低速抓绿色方块、抬约5cm再放回，并要求同步对比录像。只执行了一次；未在停止后复位或重试。
- `scripts/run_playground_grasp.py` 的真实PhysX验证成功：双指接触抬升49.965mm，稳定1.5秒，放回偏移0.035mm，无活动机械臂与桌面等非预期外部接触；无物体绑定/瞬移。原速49.767秒视频及完整物理记录在 `outputs/playground_grasp_20261107/sim_trial_03/`。
- 真机 `playground_pregrasp_run_01` 完成733动作/58.026秒。随后 `playground_grasp_run_01` 在闭夹局部207/原计划940帧停下：目标26.990mm，反馈32.1–32.2mm，比5mm到位阈值大约0.11mm，等待5秒触发stop；等待时六关节最大误差0.091°。尚未到989帧的抬升段，不能说真机抓取成功或抬起后掉落。
- **最新物理状态优先于下方旧回放记录**：停止后只读probe 100样本、TX增量0；arm_status=1、error_code=0，六电机enabled反馈仍为true。最终角度 `[-16.3,115.18,-76.453,-0.692,58.698,-13.561]°`，夹爪30.3mm，停止前后姿态已有变化。不要自动复位、直接resume旧目标，也不要把32.1mm直接当作方块尺寸。恢复需要现场准备及新鲜状态/物体位置核对。
- 相机录制已关闭。真实两段日志、双相机1392对原图及 `grasp_transfer_audit.json` 位于 `outputs/playground_grasp_transfer_20261107/`，便携记录 `docs/results/playground_grasp_20261107.json`。对比视频按各自原速播放，明确标注悬停检查、结束冻结和真机未抬升。
- 原通用replay是位置到位语义；真实接触抓取需要显式有界接触模式，不能通过统一放宽/忽略夹爪跟踪来宣称修复。本轮没有修改实机控制器或放宽停止后的阈值。新增重定时工具6项测试通过，仅接受applied和真实墙钟，不用observed强制设置仿真物理状态；本次视频使用原始成功仿真，未运行重定时物理重放。

## 2026-11-07 Playground 真机工作台孪生环境

- 新增独立 `tasks/playground`、`configs/piper_playground.json` 和 `configs/rl_nodes/playground_simulation.json`。默认空桌，保留 PiPER、工作台、安装件和相机三脚架；蓝绿方块只用于显式 `--reference-objects` 标定预览。PAP、table_buttons、电梯配置未改。
- 在 `ssh piper` 对当前静止姿态只读采集 RGB-D、设备内参和 CAN 反馈；本轮未发运动／使能／复位命令。新全局光心约 `[0.71710, 0.04654, 0.49431]m`（相对机器人底座）。按用户最新要求，双相机均使用实机读出的 `fx/fy/cx/cy`，腕部 `link6_from_camera_usd` 保留 PAP；继承外参相对实测桌面仍有约 11.7mm 高度残差，不能将拟合残差当成绝对标定精度。
- `scripts/run_playground.py` 提供干净场景三视角 PNG/USD、可选真实反馈离线重放；`capture_real_workspace.py` 和 `fit_playground_camera.py` 保存采集与拟合流程。默认对应点只适用于本次 capture_01，换相机／方块位置需重新标注。
- 插件复用公共 reset/step 服务、PiPER运行时、动作平滑。当前单环境、reward=0、没有任务成功检测、时限截断；后续需任务奖励/人工信号才能开展有意义的RL。不要称已有多环境 playground 或任务成功率。
- 修复公共PiPER运行时在Isaac相机初始化后被覆盖的非等值 fx/fy，并从USD核对主点；原配置fx=fy的任务保持光学参数。渲染与后端均检查实际USD姿态和URDF FK，解决仅set_joint_positions+render可能画面滞后的问题。
- 最终130项CPU/USD回归无跳过；实际Isaac的none和joint_ema两种模式通过关节/夹爪运动、双相机、幂等重试、超时和再次reset。标准fast shutdown由父进程验证退出码0和资源释放；slow shutdown的Isaac退出GC段错仅保留为失败诊断。所有测试进程已退出，远端相机采集已关闭。
- 最终预览 `outputs/playground_twin_20261107/playground_preview.jpg`；clean/calibration最终渲染分别为 `render_clean_03`、`render_calibration_03`。可移植摘要见 `docs/results/playground_20261107.json`，说明见 `docs/playground.md`。旧01/02渲染为修复前诊断，不作最终结果。未commit/push。


## 2026-11-07 PiPER 真机回位与完整单条轨迹回放（17:18 后更新）

- 已连接 `ssh piper` 的 `can0`（1 Mbps），双 D435i 已确认：全局 `261622078356`、腕部 `243322073170`。下方较早的“无 CAN／运动未测试”记录是历史状态。
- **已成功低速回到 PAP episode 144 的起始位姿，并在后续独立授权下完整回放该 episode 的 342 条动作。** 回位阶段仅执行 action 0。 在前三次失败和三次单独授权复位后，用户确认人员离开，`home_native_run_02` 第四次回位成功：实际 16.237 秒，native 恢复 1.951 秒／93 次反馈，普通控制 408 ticks／1 条动作完成，最大到位误差 0.454°（阈值 0.5°）。设备速度 1%，六电机均在 native 阶段开始前完成使能，证据 `home_native_success_audit.json`。
- 使能顺序已用实际反馈核对：16:59:21.937–16:59:22.017，六个电机的 `driver_enable_status` 均从 false 变为 true，随后才进入 native 回位阶段。参考驱动和 PressB 均先调用 `EnableArm(7)`；PressB 还等待六关节和夹爪使能反馈。不能用肉眼是否看到移动来判断使能。证据 `outputs/piper_replay_20261107/native_enable_audit.json`。
- 第三次触发原因：J2 首次达到 0° 后恢复标记立即退休，随后 CAN 测得 −0.052°，被普通严格限位拒绝。停止前无硬件故障反馈。native-only 延迟退休修复已完成：原始 J2/J3 恢复标记保留至稳定到达边界内目标或退出 native 阶段，并在发送前拒绝反向恢复目标；0.25° 容差及普通严格限位不变。最终 339 项相关测试通过；独立 fake-MCU 复现旧版失败并验证新版，−0.251° 回跳仍拒绝。修复部署后已完成上述一次真实回位验证；离线最大误差 0.451°，真实到位最大误差 0.454°，勿混用。
- **完整单条回放成功：** `full_episode144_run_01` 完成 342/342 actions、3,944 ticks，实际 135.524 秒。设备速度 1%，关节参考变化率 5°/s、夹爪参考变化率 10 mm/s；逐动作到位最大关节误差 0.499°、夹爪误差 1 mm。本次从合法起始姿态开始，startup allowance 全零，没有新增复位或 native 恢复。用户要求的是 episode 144 完整一条，不是 500 条全部回放。
- 数据集原时间跨度 11.367 秒，本次为低速硬件跟踪重新定时；`full_episode144_d435i.mp4` 为 139.52 秒实际执行原速录像（左全局／右腕部，含前后各约 2 秒），不能按原采集速度解读。**桌面为空，本次只验证完整轨迹执行，不代表物体抓取放置成功率、模型策略执行或真机在线 RL 验证。**
- 最近一次只读检查为完整回放后的 17:18：关节角 `[-10.621, 27.116, -26.308, -6.519, 42.856, 4.695]`°，夹爪闭合，`arm_status=0`、`error_code=0`，六电机 enabled，无驱动故障；100 次采样、TX delta=0。**机械臂保持使能并停在数据集最终位姿，无急停锁存；该位姿不同于起始位姿。** 所有回放、相机／CAN 录制和只读 probe 进程均已结束。累计保留三次失败接近、三次单独授权复位、一次成功回位和一次完整单条回放的历史。旧 dry-run 和部署证据描述当时状态，不改写原始证据。
- 完整记录见 [real_robot.md](real_robot.md)、[控制实现核对](results/piper_control_reference_audit_20261107.md)、[回放证据](results/piper_replay_20261107.json)。原始 CAN、双相机录像、执行时源码和计划保留在 `outputs/piper_replay_20261107/`；SDK 关闭本地 CAN 回环，独立监听只捕获反馈，不能据此声称记录了发送命令。

## 2026-11-07 真机端与训练端人工结果覆盖

- 新增 `pressb real_robot`，通过 `piper_real` 接入同一环境服务，固定 `num_envs=1`；保留历史 `node_type=simulation`、`nodes.simulation`、`endpoints.simulation` 通信键，增加 `environment_type=real_robot`。PiPER CAN 控制、SI 单位换算、逐帧反馈检查与双 RealSense RGB 采集已内化，不依赖外部 LeRobot 仓库。
- 控制驱动在 `robots/piper/real.py`，相机在 `sensors/realsense.py`，通用执行器在 `runtime/real_robot.py`；复位／奖励通过任务工厂替换。当前动作接口为六个绝对关节角 rad＋全夹爪宽度 m，其他模型表示需要显式适配。提供可选 EMA、限位／指令速度限制、独立超时看门狗、缺失 CAN／陈旧反馈拒绝执行，默认 `actuation_enabled=false`。
- 训练端新增 `run.human_feedback.mode=environment|human`，默认沿用环境；人工模式可用于原生仿真或真机，奖励默认为成功 1／失败 0。原始环境结果保存在 info；人工提前终止必须收到 `/end_episode` 的 episode／step 身份确认，然后再存 replay 和重置。PAP、原生电梯、桌面按键和真机支持该能力，旧远程电梯桥不支持。未标注自然终止时等待人工；取消等待不会写入假标签。
- 默认真机任务的人工复位在机械臂主机上，训练标签在训练主机上，两者 token／episode ID 不同。模板 `pap_real_walloss_dsrl.json`、`pap_real_walloss_eval.json` 已提供；相机序列号仍是待确认占位符，硬件控制默认关闭。使用与扩展说明见 [real_robot.md](real_robot.md)。
- **验证：2,516 项独立 CPU／依赖环境测试通过，无未解决依赖跳过。** 包括 HTTP 下 base／DSRL／残差完整小流程（明确使用假硬件和假推理）、人工邮箱与训练器集成、检查点语义、停止等待、生命周期及旧仿真回归；干净 wheel 安装后可独立导入／构造真机后端。真实模型／Isaac 长程实验未重跑，真实机械臂运动未测试。
- `ssh piper` 实测两台 D435i：`243322073170`、`261622078356`，640×480 RGB、配置 30 FPS，采集 30 组双路数据；关闭／重开／再关闭通过，无遗留 video 占用。画面较暗，**全局／腕部对应关系尚未确认**。远端没有 CAN，已验证缺失 CAN 报错、禁用动作的 reset 拒绝、节点 SIGTERM 正常退出及端口释放；不能据此宣称已完成机械臂控制联调。
- 本地证据 `outputs/real_robot_20261107/`；便携汇总 [results/real_robot_20261107.json](results/real_robot_20261107.json)。远端测试使用独立 `/home/metachip/Workspace/pressb_probe_20261107_cameras_01/.venv`，没有修改已有 Conda 环境。原始数据、权重未更改，正式 PAP 残差仍暂停；本轮没有自动 commit／push。
- 新增任务生命周期与公共控制接口改变了源码身份；下节旧恢复配置的验收结论属于旧快照，本轮未迁移训练检查点。以后恢复正式训练时需重新核验身份，不能绕过严格检查。

## 2026-11-07 任务／共享／历史实现归属整理完成，验收与迁移通过

- 本轮继续按用户要求拆清代码归属：电梯场景、规划、楼层语义、采集与仿真移入 `tasks/elevator`；PAP／桌面按键仍由各自任务包负责。共享服务与学习流程在 `runtime`，通用滤波／位姿转换在 `control`，标定／RGB-D 记录／图像拼接在 `sensors`，PiPER 运动学、控制与硬件在 `robots/piper`。根目录 Python 文件只保留 `__init__.py`、`registry.py`。
- `online_rl`、`rl_nodes` 以及原顶层任务／辅助模块不再保留转发占位。历史电梯 runner、专用推理桥接和日志在 `compat/elevator_rl`；冻结 VLA-JEPA 后端归入 `models/vlajepa_backend.py`。`configs/rl_nodes/` 仍是配置模板目录，不是旧 Python 包。规范路径和扩展规则见 [project_framework.md](project_framework.md#code-ownership)。
- 混合模块按实际假设拆分：通用 RGB-D 写入与电梯楼层／阶段预览分离；D435 硬件与电梯相机默认位置分离；通用有界 IK 与固定 8 mm 夹爪的历史 HTTP 协议分离。 `control/pose_protocol.py` 仅保留有限数组校验与位姿数学，`pose9_to_pose8` 要求显式传入 `gripper_width_m`；固定 640×480 PNG 辅助函数归入 `tasks/elevator/wire_protocol.py`。共享层不得反向依赖具体任务、`models`、`compat` 或 `model_sources`；模型集成由 `models` 负责，上游模型／训练源码与来源记录保留。
- 最终源码快照 `627ce9acdae9a6bf8fc27eeebe610728a58f71ef3d079939f5cd4cbeba6f96f6` 的 **2,210 项独立测试通过，无未解决跳过**：主环境 2,135、USD／碰撞 71、LeRobot 1、VLA 4，其中 1 项重复计数已扣除。干净安装包通过 **68 项规范导入和 50 项 CLI／配置检查**；旧包不存在，169 个内置模型／训练文件逐字保持。25 种场景的完整 USD 与元信息、29 个 RGB-D 输出文件完全一致；持久化位姿／图像协议对照覆盖 163 个位姿和 3 组 PNG，显式 0.008 m 时保持旧输出一致。
- 真实物理专家重放已通过：PAP **2/2**，桌面按键近／中／远×三色 **9/9**，电梯最终重跑 **2/2**；误触、超时及 joint7 夹爪检查也通过。已确认范围见本地 `outputs/component_ownership_20261107/validation_ready.json`。这些是功能回归，尚不代表新的策略成功率。
- 真实 Wall 两环境完整流程通过：DSRL40→恢复48、冻结 DSRL 后残差40、基础／残差评估及 2 个完整 episode 采集（26 transitions／104 PNG）；最终源码检查确认 112 个记录依赖不变。原生 **batch64** 新实测通过 **192 transitions／39 updates**，实际 encode 为 4×64、decode 为 3×64；64 个独立物体位置、128 张相机图像及单环境重置后其余 63 环境不变均已核验。64 环境仅覆盖 3.2 秒仿真，终止流程由两环境完整 episode 覆盖；各服务退出码 0、端口释放，不作新吞吐或学习效果结论。
- **最新恢复入口：** `outputs/component_ownership_20261107/checkpoint_migration/resume_experiment.json`（本地产物），已验证、未启动，仍为 64 环境／batch64／总预算 200000，剩余 183296。DSRL100022 与残差16704 的新副本通过严格 CPU 恢复，参数、优化器、回放和 RNG 精确保持；旧身份仍被严格拒绝。原件及历次派生文件不变，没有正式训练更新。验证报告 SHA `497ca28fed78bd3558e93894f71f74869e9966fabdad1834d10fac2eb6350ae1`。
- 新 DSRL SHA `f4fa61b544fd41ac6c7303a2578983607e08813a9db0c18c45bef5eefdea56c0`；新残差 SHA `08bbe6d04a90c32eff00b90411fcee606095afdc931b8dcdae16c830cf962188`。**正式 PAP 残差继续暂停在 16,704 transitions／14,673 updates，未恢复。**
- 本轮完整证据在 `outputs/component_ownership_20261107/`；便携摘要 [results/component_ownership_20261107.json](results/component_ownership_20261107.json)，SHA `73fd39b4c123d95c2e3b0d64fe44dcfb770585dd4ab7b44217a663b1cee5648e`。下方历史交接、报告和 SHA 记录保持原样，恢复配置以本节为准。

## 2026-11-07 顶层旧任务目录彻底移除

- 用户确认彻底归并目录结构。已删除 `src/pressb/pick_place/`、`src/pressb/table_buttons/` 的全部 20 个兼容文件及残留字节码目录；PAP／桌面按键唯一实现位于 `tasks/pap`、`tasks/table_buttons`。历史策略客户端继续放在 `compat`。旧 Python 导入不再支持，外部脚本按 [路径迁移表](project_framework.md#task-package-paths) 更新；含旧任务类路径的自定义 pickle 需在旧版本中转换。数据 schema 字符串保持不变。
- 当前脚本、22 个测试文件和文档已更新。PAP 源码身份清单移除三个已删除路径，普通检查点身份校验没有放宽。除源码身份清单外，可执行 AST 与上一提交相同；模型／训练源码、场景、物理、动作及任务语义未改。
- 本轮 **2,127 项独立测试全部通过，无未解决跳过**：主环境 2052，USD／碰撞 71，LeRobot 1，VLA 补充 3。相对上一轮少 10 项，是将 25 项旧别名／pickle 测试替换为 15 项规范路径、目录不存在、依赖边界及采集归档检查。干净构建的 wheel 不包含旧包，隔离安装、双任务 USD 场景、9 个插件及 24 项 CLI／配置检查通过。
- 新的真实 Wall＋Isaac **2 环境**短链通过：基础评估、DSRL40→恢复48、冻结 DSRL 后残差40、残差评估、2 个完整 episode 采集（27 transitions／108 PNG）。服务正常退出并释放 GPU。本轮未重跑 64 环境，上一轮证据作为历史记录保留，不作为本轮新实测。
- **最新恢复入口：** `outputs/task_directory_cleanup_20261106/checkpoint_migration/resume_experiment.json`，已校验、未启动，仍为 64 环境／batch64／总预算 200000。DSRL100022 与残差16704 的新副本通过严格 CPU 恢复；参数、优化器、回放及 RNG 保持一致，冻结 actor 不变。原权重及此前派生副本均未修改。正式 PAP 残差仍暂停在 **16,704 transitions／14,673 updates**。
- 最终检查点 SHA：DSRL `1bac6cb0ab21486cc1f265280890b038090ae55e6c1824439d19de1c473ac78d`；残差 `8344485abb5584a93e67a6feb21c4d2fc615ac8be5d6555418864c662c600a42`。离线迁移记录保留了元信息更正后的中断重启，以及修正验证脚本 PyTorch guard 后仅继续恢复验证的过程；最终均通过，产品代码没有因此变化。
- 证据目录 `outputs/task_directory_cleanup_20261106/`；便携摘要 [results/task_directory_cleanup_20261106.json](results/task_directory_cleanup_20261106.json)。下方“保留旧任务别名”和较早恢复配置均属于历史阶段，当前以本节为准。

## 2026-11-06 PAP／桌面按键任务目录归并与完整回归

- 用户要求归并 `src/pressb/pick_place`、`src/pressb/table_buttons` 并测试。PAP 场景、物理结果、采样／统计已归入 `tasks/pap`；桌面按键场景、布局、运动模板、碰撞、采集和评估已归入 `tasks/table_buttons`。原两个目录的 20 个文件仅保留说明或模块别名，不再有函数／类实现；旧导入、monkeypatch 和历史 pickle 类路径仍兼容。
- 共用 USD 几何／材质在 `runtime/scene.py`，桌面／房间／夹具／灯光／三脚架在 `runtime/workbench.py`。桌面按键场景不再依赖 PAP 场景。历史 Wall-X WebSocket 和桌面按键 HTTP 客户端移至 `compat`；现代任务后端不加载它们。PAP 从观测加载初始姿态独立为 `tasks/pap/initialization.py`。
- 采集／评估脚本已改为规范导入；采集源码归档和任务 source fingerprint 绑定真实实现及共享依赖。数据 schema、动作单位、相机、随机化、成功判定和轨迹时序不变。内置模型及监督训练源码未修改。
- 主环境 2,060 项通过，18 个依赖跳过在对应环境补齐：USD／碰撞 73、LeRobot 1、VLA flow 3，合计 **2,137 项独立测试通过**。与上一提交比较 17 种场景／三脚架配置，完整 USD 文本及元信息逐字一致。安装包验证、18 组旧／新别名、5 个实验配置与 14 个脚本入口通过。
- Isaac 实测：PAP 专家抓放 2/2，桌面按键远中近×红黄蓝 9/9、误触／超时／请求幂等／joint7 夹爪开闭通过，电梯专家重放 2/2。实际内置 Wall 两环境完成基础评估、DSRL40→恢复48、固定 DSRL 后残差40、评估和2个完整episode采集；64环境完成原生batch64、192 transitions／39 updates、单环境重置隔离及检查点验证。均是功能回归，不是学习效果或吞吐评测；服务已退出。
- 原暂停训练仍是 **16,704 transitions**，没有恢复。最新兼容派生权重及已校验、未启动的恢复配置在 `outputs/task_consolidation_20261106/checkpoint_migration/`。优先使用此处 `resume_experiment.json`；前两轮迁移配置仅对应历史源码身份。仅更新仿真源码身份，模型／优化器／回放／RNG 精确保持；原始和先前派生权重均未修改，普通检查点校验没有放宽。
- 完整证据：`outputs/task_consolidation_20261106/`；便携摘要：`docs/results/task_consolidation_20261106.json`。Isaac fast shutdown 的部分原有报告在 `operations_status=passed` 后停留 `status=closing`，另已核对进程退出码0；桌面验证日志保留了按既有环境配置补齐 EULA 变量后重跑的记录，以及此前已有的 Warp 启动诊断。

## 2026-11-06 模型推理与监督训练代码内化，统一提交版本

- 用户在要求整理 commit/push 后追加“将 wall-x 和 vla-jepa 中框架需要的推理和训练代码内化”。默认推理现在使用 `src/pressb/model_sources/walloss` 与 `vlajepa`，不依赖原独立源码克隆；保留显式 `repo/vla_repo` 旧源覆盖。旧克隆未删除。模型权重、数据集、基础模型文件、Conda 依赖和 CUDA toolkit 仍是外部输入。
- Wall-OSS 包含原生批推理、外部噪声、编码／前缀缓存、数据读取、监督训练、协议兼容和 CUDA 源码；VLA-JEPA 共用原模型主体，内化 pose9 及 H200 原本未提交的 joint6/joint7 单／多任务数据、训练和推理代码。原模块名和权重键保持一致；来源版本、原始／内置逐文件摘要和版权信息随源码保留，没有推定缺失的上游 LICENSE。
- 统一监督训练入口 `scripts/train_model.py` / `python -m pressb.training`；Wall-OSS 一个模板，VLA-JEPA 五个 profile：pose9、joint6、joint7、joint6_multitask、joint7_multitask，均在 `configs/model_training/`。在线 RL 继续用 `scripts/run_experiment.py train`。PAP 历史 checkpoint 配对配置逐字保存在 `configs/model_metadata/walloss_pap_20260606.yaml`，不要拿其历史 resume 路径启动新训练。
- Wall CUDA 扩展已从内置 csrc 重建，二进制位于用户缓存（不入 Git），构建脚本可获取固定版本 CUTLASS；loader 校验 Python/Torch ABI、CUDA 源码摘要和二进制摘要。默认安装包包含必要源码、配置、CUDA 和来源清单，隔离 Python／目录下的六种训练入口解析均通过。
- 新主回归 **2031 passed**；16个缺依赖跳过项在专用环境展开补测75项，主项目合计 **2106 项通过**。额外原源测试：Wall模型29项、旧launcher3项、VLA202项（含6项真实数据读取）；VLA的5项历史pose数据测试缺原输入路径仍跳过。真实小型flow head进行了反向传播、AdamW更新及优化器／权重恢复；未启动完整监督长训。
- 完整GPU权重对照：Wall BF16 batch2/8使用内置源码＋新CUDA扩展，与旧源码＋旧扩展动作／特征最大差0.0，200次CUDA窗口索引检查通过；VLA pose9/joint6/joint7与原策略动作最大差均0.0，pose9严格加载1617个张量。全程未恢复PAP残差训练。报告 [results/model_internalization_20261106.json](results/model_internalization_20261106.json)，使用说明 [bundled_models.md](bundled_models.md)。
- Wall-X 独立仓库的批推理改动已提交并推送 `base`：`cc4dd1e0579869c3e782205be0339c84c78f1d41`。PressB 内化清单仍保留导入时的源 HEAD＋工作区补丁摘要，不篡改历史来源记录。VLA-JEPA / ZPRL 的本地独立克隆未修改。
- **当前可恢复副本以新目录为准：** `outputs/model_internalization_20261106/migrated_checkpoints/`。内置模型来源身份变化后，再次生成 DSRL100022／残差16704 兼容副本，CPU严格恢复、完整学习状态逐字节一致验证通过，原件和上次副本均不变。新 DSRL SHA `856500b9b5f509cceb5378d943fcf53c7cc9821ca5546db14c737bf79e3f385a`，残差 SHA `29196769fb55752742b122cd699d4e6432e3d2e6cea4c8bf22eba2784ee5ef97`。同目录 `resume_experiment.json` 已校验但未启动，预算仍是总200000。下方旧重构目录的恢复配置仅对应上一版源码，不用于当前默认内置模型。

## 2026-11-06 通用框架重构已完成，PAP 残差训练按要求暂停

- 用户要求“通用运行核心＋任务插件＋模型适配器＋实验配置”，并要求重构后验证功能。统一入口为 `scripts/run_experiment.py`，支持 `list / validate / prepare / inference / simulation / train / eval / collect`。配置在 `configs/experiments/`，完整说明见 [project_framework.md](project_framework.md)。后续模型内化及提交状态见上方最新记录。
- `src/pressb/runtime/` 负责三端协议、批推理缓存、reset/step 调度、SAC、回放、异步更新、评估、记录、检查点及共享 PiPER/PhysX/相机工具；`tasks/` 实现 PAP、桌面按键和电梯的物理规则；`models/` 实现 Wall-OSS、pose9 VLA-JEPA、joint6/joint7 VLA-JEPA；`experiments/` 负责配置继承、校验、展开和统一启动。旧导入、旧 CLI、旧节点配置及逐帧 LeRobot 专家采集保留兼容。通用 `collect` 输出 chunk 轨迹，不能当作逐帧 LeRobot 数据。
- 全量回归 **1977 passed**，缺依赖的 13 个跳过项在 USD/碰撞与 LeRobot 环境中展开补测 **72 passed**，合计 **2049 项通过，无未解决跳过/失败**。真实 Isaac 专家回放 PAP 2/2、近中远×三色桌面按键 9/9、电梯 2/2 成功；同时验证错误目标、超时、独立 reset、夹爪与采样复现。真实 joint6/joint7 模型同输入与原实现输出差值为 0。以上是功能验证，不是新模型成功率评估。
- 真实三端短测覆盖 base eval、DSRL 更新/恢复、冻结 DSRL 后残差更新、残差 eval 和轨迹记录；**64 环境、128 路相机、Wall-OSS 原生 batch64** 短测完成 192 transitions/39 次更新。终止与融合重置另在 2 环境测试中验证。桌面按键插件目前每节点仅 1 个环境；未声称所有任务都已向量化。证据汇总 [results/framework_refactor_20261106.json](results/framework_refactor_20261106.json)，原始日志位于 `outputs/framework_refactor_20261106/`。
- 用户明确要求 PAP 残差先停：原 `outputs/pap_residual_on_dsrl100k_200k_20261106/train/checkpoints/step_000000016704.pt` 已完整保存 **16704 transitions / 14673 updates / 16704 replay**。原 trainer、monitor 及两服务已退出；全部重构验收服务也已停止。**不要自动恢复，不要按下面历史“已启动”记录重复启动。**
- 为兼容源码迁移，新增 `scripts/migrate_rl_checkpoint.py`，仅允许有回归证据的显式源码版本迁移，普通加载身份检查未放宽。原 DSRL100022 与残差16704 检查点不变；兼容副本位于 `outputs/framework_refactor_20261106/migrated_checkpoints/`：`dsrl_step_000000100022.pt`、`residual_step_000000016704.pt`。`verification.json` 记录严格加载、全部模型/优化器/回放/RNG 数据一致和原 SHA 不变；冻结 actor SHA 仍为 `aaaf59ebb33287876bcb52ae144b09071e3719361caa0819d62a5cae32074050`。仅做 CPU 恢复校验，没有训练更新或仿真请求。
- 已准备并校验同目录 `resume_experiment.json`，但未启动。它沿用原训练语义，恢复至总预算 200000（不是额外 200000），新日志目录为 `outputs/pap_residual_on_dsrl100k_200k_after_refactor/`，原输出保留。以后启动前仍核对 GPU/端口资源。检查点默认每 100k 永久保存，正常停止额外保存；恢复优化器/回放/RNG 后从新 episode 开始，不保证中途物理状态续接。

## 2026-11-06 固定100k DSRL，新增200k动作残差训练启动记录（现已暂停，见上）

- 用户要求“用这个100k的dsrl权重，继续训练200k的动作残差”。**14:51:03 已启动** `outputs/pap_residual_on_dsrl100k_200k_20261106/`，先读 `train/status.json`，不要重复启动。冻结Wall-OSS和100k DSRL；新残差actor/critic/optimizer/replay从零开始，未resume旧400k残差。本轮预算是新增200000，不包含源DSRL步数。
- 冻结源 `outputs/pap_dsrl_local_100k_20261105/train/checkpoints/step_000000100022.pt`，实际100022条/98001更新；SHA `d5c11f892b0dc899f9f21c8d0d294d2770a96503609f0985b53d0f3d37c444a8`，冻结actor SHA `aaaf59ebb33287876bcb52ae144b09071e3719361caa0819d62a5cae32074050`。只读actor，原optimizer/replay/RNG不加载；deterministic、noise32×20/scale1.5从源继承。
- 沿用上一轮残差配置：64env、32步chunk、全7维scale `[.05,.05,.05,.05,.05,.05,.005]`（六关节rad、夹爪全宽m）、gamma.995按30Hz时间折扣；warmup2000零残差+50k渐进启用，batch256/UTD1/replay100k，timeout不bootstrap、Q不设上下界、无额外动作平滑。新配置 `checkpoint.interval=100000, keep=null` 保留全部；首次100k跨界可能保存100032，最终200000可整除64。保留冻结源用于组合评估。
- GPU0复用推理PID2419135/start47112225（WS18131）和64环境仿真PID2559979/start50881195（HTTP18101）；GPU1新trainer PID2864548/start60722307，monitor PID2864549/start60722307。receipt、实际配置副本、源码快照/SHAs、资源预检在新run目录，操作前核对PID/start_ticks/命令。与源DSRL模型/任务/观测身份13项一致；6个关键实现文件与已验证100k快照逐字一致，未改训练代码。
- [实时W&B](https://wandb.ai/penggq2025-southern-university-of-science-technology/pressb-online-rl/runs/b3676d48406448a29665ea37e6bb424c) 已连接，远端API已确认学习指标上传。`initialization.json`记录0更新/0回放/空optimizer，正确冻结源及0可训练参数。14:55启动验收达到3136条/1105更新，数值有限、drop0、四进程身份一致；独立32项检查通过。1024→3136窗口13.27条/s，预计总耗时4.19h。证据 `startup_verified.json` / `startup_independent_audit.json` / `wandb_learning_check.json`；后续看实时status，recent SR仍是训练统计。
- 按历史约13条/s预计4h10–20m，约19:00–19:15完成，属于估计。原DSRL100k/400k及残差400k均保留。本轮未commit/push。

## 2026-11-06 PAP DSRL 100k 已完成，训练曲线已生成

- `outputs/pap_dsrl_local_100k_20261105/` 于北京时间11月5日23:10:19正常完成，耗时7515.35秒（2h05m15s），100022 transitions / 98001 updates。预算100000，因最后完整并行批次多22条，status准确记录；drop0。
- 全部9915条episode、1563条metrics核对一致：7499成功，累计训练SR75.6329%，recent100=91%、recent1000=90.8%。曲线PNG/PDF/SVG、CSV、逐episode数据及日志SHA摘要在该目录 `plots/`。这是训练探索统计，尚未独立评估。
- 最终检查点 `train/checkpoints/step_000000100022.pt` 已保存，包含回放和优化器；配置 `interval=100000, keep=null` 全部保留。原400k DSRL与400k残差权重不变，后续不得退回keep2默认。
- trainer和资源monitor正常结束。W&B上传子进程在收尾时超时，`wandb_status.json` 为 `local_fallback/shutdown_timeout`，远端最后指标可能缺失；本地status、metrics和权重完整，曲线按本地证据生成。本轮未启动额外训练/评估，未commit/push。

## 2026-11-05 重新训练 PAP DSRL 100k，后续检查点每100k全部保留

- **用户新的持久要求：后续强化学习每100k保存一次权重，保留全部，不再默认只留最新两份。** 通用三端trainer默认及6份训练模板改为 `checkpoint.interval=100000, keep=null`；旧普通runner默认及两份通用模板同步100k，原本不删除编号权重。旧fast runner继续100k永久编号权重+10k覆盖last.pt，永久文件不删除。历史outputs配置/已有权重不改；不要从旧配置复制回keep2。
- 通用checkpoint按绝对区间跨界保存，避免批次越界导致后续保存时间持续漂移；final/正常停止仍保存。`keep`正整数仅用于显式选择轮转的兼容配置。96项trainer/end-to-end与34项旧runner测试通过，涵盖全部保留、旧keep2兼容、跨界/提前final/resume、非法配置和尾批计数。
- 应用户要求从头重训PAP DSRL到100k。**21:05:03 已启动** `outputs/pap_dsrl_local_100k_20261105/`；先读取 `train/status.json`，不要重复启动。新actor/critic/optimizer/replay全部从零，无resume、无动作残差、无冻结旧DSRL；只冻结Wall-OSS。原DSRL和残差的400k权重保留。
- 配置与原DSRL400k一致，只改预算/保存策略/输出日志：64环境、noise32×20/scale1.5、gamma.995按30Hz控制步、warmup2000、batch256、UTD1、replay100k；相同红方块位置随机任务、20秒episode和原相机、无动作平滑。`training_config_comparison.json`和14项身份检查证明契约一致。64环境必须执行全部活跃槽，100k尾批可能最多超32条，文件名/status按真实计数，不能假称精确100000。
- GPU0复用推理PID2419135/start47112225和64环境仿真PID2559979/start50881195；GPU1新trainer PID2712306/start54326306，resource monitor PID2712307/start54326306。receipt均在新run目录，操作前核对PID/start_ticks/命令；源快照、配置和资源预检同目录。预计约2小时10分钟，约23:15完成，属于估计。
- [本轮W&B实时日志](https://wandb.ai/penggq2025-southern-university-of-science-technology/pressb-online-rl/runs/c79e43a6291746e5b2aa073fd5ea8c0f)。21:08启动验收达到2816条/785更新，数值有限、drop0、四进程身份正常；远端W&B已收到真实学习指标。CPU独立按seed42重建actor哈希与新初始化完全相同，22项独立审计通过。证据 `startup_verified.json` / `startup_independent_audit.json` / `wandb_learning_check.json`；当前值请读实时status。本轮未commit/push。
- 上轮固定DSRL+动作残差400k已于20:14完成，397969更新，52576个episode中46102成功，recent100=94%、recent1000=92.2%；冻结DSRL哈希全程不变。曲线在 `outputs/pap_residual_on_dsrl_400k_20261105/plots/`，尚未独立评估组合策略。原DSRL约100k checkpoint（100480）曾保存后被旧keep2轮转删除，只剩391872/400000，因此此次从头重训，不是恢复丢失权重。

## 2026-11-05 PAP 固定 400k DSRL，重新训练 400k 动作残差

- 上轮本机 PAP DSRL 已完成：400,000 transitions / 397,969 更新，耗时8h27m52；累计训练SR90.45%、recent100=95%。随后独立确定性评估128条、122成功（95.31%）。这两个SR口径不同。源checkpoint位于 `outputs/pap_dsrl_local_400k_20261105/train/checkpoints/step_000000400000.pt`，SHA256 `3fe807dc0ba36bbcf4f3b8a66d134d145266f915d9c78ce02a1842b55b0654cf`。
- 用户要求在此基础上再训练400k动作残差。**11:39:15 已正式启动** `outputs/pap_residual_on_dsrl_400k_20261105/`，先读 `train/status.json`，不要重复启动。原Wall-OSS和DSRL actor冻结；新残差actor/critic、优化器、replay从零开始，未resume预检或任何旧残差。`initialization.json`和`frozen_dsrl.json`保存证据。
- GPU0复用原BF16 Wall-OSS batch64推理PID2419135/start47112225（WS18131）及新64环境仿真PID2559979/start50881195（HTTP18101）；GPU1残差训练PID2561762/start50931471。资源monitor PID2561763/start50931471。实际receipt、配置、源码快照均在新run目录；操作前核对PID/start_ticks/命令。
- PAP执行绝对关节/夹爪动作；残差全7维scale `[0.05,0.05,0.05,0.05,0.05,0.05,0.005]`，单位前6项rad、末项全夹爪宽m。gamma0.995按30Hz实际控制时长折扣；warmup2000零残差，随后50k渐进探索；batch256、UTD1、replay100k，timeout不bootstrap、critic不做Q值裁剪；每10k保存完整checkpoint，保留最近2份。原冻结DSRL使用确定性actor、noise32×20、scale1.5。保留源checkpoint供组合策略加载。
- 同一观测编码一次→冻结DSRL产生噪声→缓存解码base动作→新残差→统一物理边界裁剪。新 `frozen_dsrl` 配置严格绑定源权重SHA、原模型/任务/观测契约；不恢复旧优化器、replay或RNG，保存时核对冻结actor张量哈希。实现见 `src/pressb/rl_nodes/trainer.py`，公开模板 `configs/rl_nodes/pap_residual_on_dsrl_local_fast.json`，说明 `docs/configurable_rl_nodes.md`。
- 真实512-transition短预检已完成277更新，新actor确实变化、冻结actor不变，replay与指标有限，缓存/仿真所有权已释放，独立结果 `preflight_verified.json`；77项trainer/end-to-end测试通过。正式启动验收11:43达到2,880 transitions / 849更新，数值有限、drop0、四进程身份正常；W&B远端API已收到学习指标。证据 `startup_verified.json`、`wandb_learning_check.json`。1024→2880启动窗口13.66条/s，预计总耗时约8–9小时；后续以实时status为准。
- [W&B实时日志](https://wandb.ai/penggq2025-southern-university-of-science-technology/pressb-online-rl/runs/119bc01936f440feb876eb69ab8a16d5)，recent SR为训练探索最近100个完成episode。仅统计本轮新增400k transition，不包含源DSRL的400k。未commit/push。
- 上轮执行视频及频闪修正已完成：`outputs/pap_dsrl_400k_video_20261105/stable_render/pap_dsrl_400k_execution_1x.mp4`。同3条实际轨迹、30fps原速；离线重渲染提高到2×8子帧，冻结画面波动第三视角下降37.2%、腕部33.7%。**正式训练仍保持原2×1渲染和无动作平滑**，未把视频效果调整带入训练。

## 2026-11-05 本机 PAP DSRL 400k 正式训练已启动

- 用户要求“先在本机启动400k训练”。北京时间 **02:16:50** 从头启动 **400,000 chunk transitions**，无 resume，未使用两轮10k基准的RL权重/回放。运行目录 `outputs/pap_dsrl_local_400k_20261105/`；先读取实际 `train/status.json`，不要重复启动。
- GPU0：复用健康的本机BF16 Wall-OSS推理，原生batch64/H32/10flow/full_attention，PID2419135/start47112225，WS18131；新64环境PAP仿真PID2449684/start47551707，HTTP18101。GPU1：DSRL训练PID2450028/start47556959，cuda:1、两卡可见、CPU线程1。实际receipt/配置/源码SHA快照均在运行目录；管理进程前核对PID/start_ticks/命令。
- 配置沿用本机较快方案：红方块、四条绿标记边围成矩形内随机位置；640×480原生相机、20秒episode、不平滑、动作裁剪；noise32×20、scale1.5、warmup2000、batch256、UTD1、replay100k，gamma0.995按30Hz控制步，timeout不bootstrap。只学习噪声策略，Wall-OSS冻结；每10k保存含replay/优化器/RNG完整检查点，保留最近2份。原仿真与模型元数据逐项核对通过。
- W&B已启用且远端API收到真实指标：[实时日志](https://wandb.ai/penggq2025-southern-university-of-science-technology/pressb-online-rl/runs/a8dab53dbad144a7b69042c9f2919485)。recent SR是训练探索最近100个完成episode统计，不是独立评估。`train/metrics.jsonl`、`episodes.jsonl`为本地证据，`wandb_status.json`为上传状态；启动后核验写入 `startup_verified.json`。
- 磁盘剩余约13.7TiB，仿真加载后主机available约39.5GiB，100k replay原始约1.776GiB，keep2+临时写入峰值有充分余量；预检 `resource_preflight.json`。资源采样每60秒、上限48h，trainer退出即结束，PID2450240/start47560499；无RPC/GPU分配，不干扰采样。
- 启动验收：2,560 transitions / 529 SAC更新 / 189.68秒（13.50条/s），3端身份正常、指标有限、drop0；W&B远端已收到学习指标。`startup_verified.json`、`wandb_learning_check.json`保存证据。该计数为启动快照，后续直接看实时status。
- 按上一轮实测13.3434条/s预计约 **8.33小时**，对应11月5日约 **10:37**，属于估计。运行说明与安全停止步骤在该目录 `README.md`。本轮未commit/push。下方“仅保留推理/未启动400k”是此前基准结束时的历史记录。

## 2026-11-05 Wall-OSS 已迁回本机，三种部署耗时对照完成

- 用户要求本机两张 5090 上运行推理并比较耗时。复用独立 `.conda/envs/wallx-inference`（Torch2.7.0+cu128），将已有修正的 Wall CUDA 扩展重编为原生 sm120；200项GPU窗口索引检查通过。模型保持BF16、batch64、H32、10步flow、full_attention前缀缓存。新公开配置 `configs/rl_nodes/pap_walloss_local_fast_inference.json` 与 `pap_dsrl_local_fast.json`，配合已有 `pap_vector_fast_simulation.json`。
- 真实三组同口径：均10,048条，主窗口2048→10048。H200推理 **12.5489条/s**；本机GPU1推理+SAC共卡 **12.9107条/s**；本机GPU0仿真+推理、GPU1独占SAC **13.3434条/s**。最后一种比H200快 **6.33%**，400k/1M外推 **8.33/20.82h**，并非已完成长程训练。15条/s仍未达到，未更改门槛。第二轮总耗时755.006秒，两轮本机均8017更新，窗口8000更新、零drop、47未消费credit保存在完整checkpoint内。checkpoint/replay/episode审计全部通过。
- 同64组观测的5次稳定纯推理对照：H200 encode/decode **1.812/1.179秒**，本机 **0.964/1.743秒**。通信减少而本地flow更慢，净收益有限。完整训练中，推理与SAC共卡decode1.933秒，改为与仿真共卡后1.770秒；SAC窗口累计耗时188→50秒（与RPC重叠不可直接相加）。共卡GPU0采样峰值29,852MiB，最低空闲2,235MiB，无OOM；64环境/128路640×480及任务物理配置均未降低。
- 输出差异：两张本机5090同输入特征/动作完全一致；H200与本机虽权重/源码/归一化一致，Torch/CUDA栈不同，BF16输出不完全一致。三组固定噪声手臂RMS约0.16度、最大2.625–5度，夹爪最大5.16–14.92mm。这是执行前命令差异，不能用训练recent SR证明迁移后的成功率等价。纯推理配对与三种部署均只做有限次数测量，不是硬件单因素或统计显著性结论。
- 当前仅保留本机GPU0推理 `ws://127.0.0.1:18131`，PID2419135/start_ticks47112225，cache0；管理前必须核对 `outputs/pap_dsrl_local_5090_20261105/sim_inference_colocated/inference_process.json`。两组测试trainer/simulation已退出，临时H200 GPU2服务也已停。GPU0约14726MiB已用（含桌面），GPU1约17MiB。下次仿真GPU0+推理GPU0、SAC GPU1；推理启动用 `CUDA_VISIBLE_DEVICES=0` 配JSON逻辑`cuda:0`。未启动新的400k训练。
- 原始证据 `outputs/pap_dsrl_local_5090_20261105/`，第二组 `sim_inference_colocated/`；汇总 `three_deployment_comparison.json`、最终服务状态 `final_deployment_state.json`。文档 [本机推理对照](pap_dsrl_local_inference.md)，可跟踪报告 `docs/results/pap_dsrl_local_5090_20261105.json`。W&B两组已结束：`748e5ea0bc1d4ae88d38ff4166090f5f`、`3f047d6e915742fcb3a98e57c1d545b4`。本轮未commit/push。

## 2026-11-04 PAP DSRL 提速完成，10k 持续验证已结束

- 用户要求停止原训练并优化。旧运行 `outputs/pap_dsrl_h200_bf16_64env_400k_20261104/` 安全停止于 **3,520 transitions / 1,532 updates**，完整checkpoint/replay/优化器/RNG保留。不要自动重启旧run或把其replay恢复到新profile；渲染、调度和模型运行身份不同。下方“正式训练已启动”是历史记录。
- 新配置：`configs/rl_nodes/pap_vector_fast_simulation.json`、`pap_walloss_h200_fast_inference.json`、`pap_dsrl_fast.json`。保留64环境、128路原生640×480、最多32动作/chunk、10步BF16 flow、20秒episode、UTD1与物理成功标准。已实现DLSS历史显式清除+两次零物理tick渲染，step/indexed-reset共用观测边界，零bootstrap终止态省略视觉编码，精确PIL预缩放/二进制PNG，以及单次decode内 `full_attention` 前缀隐藏态复用。默认旧接口和电梯入口保留。
- 最终验证 `outputs/pap_dsrl_optimization_20261104/fused_native_train/` 完成 **10,048 transitions / 8,018 updates / 799.935秒**。warmup后2048→10048窗口 **12.5489条/s**，比旧1.4406快 **8.71倍**；含检查点保存。8000条对应8001次更新（消耗1份已有credit），无drop，580次实际环境重置。15条/s目标尚未达到，其余验收条件通过。400k／1M外推 **8.85／22.14小时**，不等于已跑完；更大replay保存、启动及网络变化会增加时间。历史电梯400k7.43h、1M15.16–19.81h，且电梯每chunk最多7动作，PAP32动作。
- 当前耗时：仿真step+reset38.2%，encode整条RPC35.7%，decode22.9%；SAC更新与等待重叠。旧权重备份共享网络，经85秒恢复watchdog保护，仅暂停接收rsync10.87秒做A/B，encode中位1.841→1.718→恢复1.827秒；备份已恢复写入，15次特征完全一致。不能把这项短测当整体训练达标。
- 正确性：原生640的64环境专家回放64/64成功；64个真实初始布局、三种噪声的full_attention与原batch64 BF16特征/动作样本精确一致（不代表BF16 batch1等价）；CPU回归295+电梯Python3.11回归7，共302通过，Wall前缀另14通过。完整checkpoint audit通过，保留10048回放/8018更新与46未消费credit，参数确有更新且全部有限。训练SR不等于独立评估。
- 训练及仿真进程已退出，H200 GPU2推理服务也已停止释放GPU；未启动新的400k正式训练。SSH端口转发仍可复用。部署H200时必须 `CUDA_VISIBLE_DEVICES=2` 配合配置内`cuda:0`，不要占用他人的GPU0任务。默认示例W&B关闭，本次验证显式开启并已finished：[W&B](https://wandb.ai/penggq2025-southern-university-of-science-technology/pressb-online-rl/runs/1781eecaceed49f08508cd3f1b8a301c)。
- 完整说明、命令与限制：[PAP提速](pap_dsrl_pipeline_optimization.md)；可跟踪结果 `docs/results/pap_dsrl_pipeline_optimization_20261104.json`。原始证据在 `outputs/pap_dsrl_optimization_20261104/`，渲染在 `outputs/pap_dsrl_optimize_20261104/`。本轮未commit/push。

## 2026-11-04 PAP DSRL 正式训练已启动

- 用户确认使用 H200 BF16 推理；北京时间 **22:08:57** 已从头启动 **400,000 chunk transitions**，无 resume。运行目录 `outputs/pap_dsrl_h200_bf16_64env_400k_20261104/`；参数、命令、实时状态和安全停机说明见该目录 `README.md`。正式 `train/` 与短预检 `preflight/` 独立。不要根据本段重复启动；先读 `train/status.json` 并核对进程 receipt。
- H200 GPU0：冻结 Wall-OSS，原生 batch64、BF16、10 步 flow、cache256；服务 PID 历史记录 **3106521**，已替代下节旧 PID 3084098。SSH 转发 `ws://127.0.0.1:18130`；本机 GPU0：64 PAP 环境、128 路原生 640×480 相机、48 次 reset 渲染；本机 GPU1：DSRL SAC（`zprl` Conda，Torch 2.7.0+cu128）。各进程 PID/start_ticks/命令见 `inference_process.json`、`simulation_process.json`、`training_process.json`，源码及 SHA 已冻结保存。
- 固定红色方块、绿色标记长轴围成的矩形内随机位置，20 秒 episode；不加动作平滑，执行绝对关节/夹爪动作并显式裁剪至物理边界。SAC 观测 2062 维、噪声动作 `32×20`、scale1.5、warmup2000、minibatch256、UTD1、replay100k；gamma0.995按30Hz实际控制时间折扣。只训练噪声策略，不更新 Wall-OSS 参数，不叠加动作残差。每10k保存，保留2份含回放、优化器和 RNG 的完整检查点。
- 为保留完整 batch64，本次 WS 客户端和服务端消息上限显式设置 **192 MiB**；接口默认仍64MiB，HTTP旧请求限制不变。无需降低相机分辨率或拆批。可选 W&B 使用独立上传子进程，不阻塞采样/学习；训练退出有界清理、父进程意外死亡检测、失败回退本地日志，上传配置不包含端点 URL 或 token。
- 真实短预检完成 **256 transitions / 211 SAC updates**；保存检查点所有数值有限、actor相对初始参数最大变化0.00988，actor/critic优化器和完整回放可读，清理后推理cache0、仿真run已释放。此短测没有完整episode，不能作为成功率或收敛证据。审计见 `preflight_audit.json`；本次logger/transport/端到端相关测试通过，最终logger父进程退出回归也通过。
- 正式运行的 [W&B](https://wandb.ai/penggq2025-southern-university-of-science-technology/pressb-online-rl/runs/2cd7fd1fcd9f4b00b2b1e3c5a11bc295) 已由远端 API 核验收到真实指标（`wandb_remote_check.json`）。以 `train/status.json`、`metrics.jsonl`、`episodes.jsonl` 为本地完整证据；`wandb_status.json` 给出连接状态及 URL。recent SR 是训练探索期间最近100个完成episode的成功率，不是独立评测。以下“尚未正式训练”段落为此前阶段记录。本次未 commit/push。
- **22:29 启动验收：** 正式运行达到2,048 transitions / 50次SAC更新，actor和critic均实际更新，所有最后更新指标有限、dropped_update_credit=0；154个已完成episode中122成功，recent100 SR=68%。这是几乎全部来自warmup的探索统计，不能作为RL提升结论。`startup_verified.json`保存随后22:30的验收快照（2,112 transitions / 116 updates），后续进度读取实时status。此时累计端到端仅约1.68 transition/s，主要累计耗时为reset615.8秒、encode整条RPC376.0秒；完整400k按早期速度约66小时，未达到小时级目标。encode包含PNG处理/网络/模型，不能全归为GPU计算；reset全场48帧与终止时截图是后续优化点，未为提速而更改本次图像契约。不要将此前16.11 chunks/s的纯模型基准当成训练采样吞吐。

## 2026-11-04 H200 Wall-OSS 真正 batch 推理

- 已部署到 H200 `/home/pengguanqi/Worksapce/Research/PressB-walloss`，使用独立 `.venv` 覆盖依赖，继承现有 PyTorch 2.6.0+cu124；原训练环境未改。PAP 权重位于持久目录 `/home/pengguanqi/Models/piper_pap_20260606`（实体在 data 分区），文件 SHA256 与本机一致；无优化器文件上传。只清理了可重建的包管理缓存，未删除既有训练权重。未 commit/push。
- 推理 policy、预处理、模型、缓存拆分/重组和 PressB adapter 均支持 native batch；不同长度语言指令左 padding，并修正多模态 RoPE offset。每条缓存独立持有 embeddings/位置编码与基线噪声，支持重排、子集和重复引用；并非 Transformer KV cache。视觉按观测总 token 宽度分组，保留内部图像/窗口 mask，避免所有图像拼接后的平方级 attention。B1 旧路径及原 RTC 接口保留。
- 修复 `ref/wall-x/csrc/window_index.cu` 超过 256 窗口时的跨 CUDA block 前缀和竞争；H200 已用 CUDA 12.8、sm90 重新编译。**其他机器部署也必须重新编译 extension，不能仅同步 Python；本机旧 extension 尚未重编。** GPU 窗口索引测试 200 次通过（含 B64、混合尺寸、默认/非默认 stream；不代表所有模型 CUDA 算子均支持任意 stream）。Wall batch/flow CPU 测试 23 项通过，PressB 相关测试 30 项通过；本轮较早的完整 PressB 回归 1,521 passed / 13 skipped。
- 完整权重 BF16 实测 B1→B64：2.54→16.11 observation chunks/s（6.35×），B64 编码 1.657s、10 步 flow 解码 2.315s，峰值 allocated/reserved 11.64/13.42 GiB。forward hooks 证明 1 次视觉调用含 128 图像、10 次 Transformer 调用均 B64；缓存解码不再编码图像。基准使用 3 组真实 Isaac 观测循环扩展、4 种长度指令和独立外部噪声，不是 64 个独立布局的策略评测，不含网络/仿真/学习。
- **精度限制：** BF16 B64 对同噪声逐条基线，最大关节差 0.135263 rad（7.75°）、夹爪差 8.125mm；RMS 分别 0.004083 rad / 0.620mm。没有将这些误差判为可接受，也不能保证闭环 SR 不变。几何/索引对照完全一致；FP32 同精度 B64 对逐条最大误差 3.74e-5rad / 2.25e-6m，但仅 3.11 chunks/s。关闭 BF16 reduced-precision reduction 未解决偏差，未更改 production 默认。FP32 也不等价于原 BF16 单条基线。
- 配置 `configs/rl_nodes/pap_walloss_h200_inference.json` 当前 BF16、batch64、cache256、TTL600。H200 GPU0 服务及本机 SSH 转发保持运行，可访问 `ws://127.0.0.1:18130`；本次启动 PID H200=3084098、本机 tunnel=2272260，仅作记录，操作前核对 receipt 中 start_ticks。网络64条 encode/decode/infer/release 通过，重复 decode 与同 batch cached-vs-infer exact0，缓存释放后0。encode6.235s、decode2.445s、infer8.203s；最大请求29.00MiB，仅代表此3组观测样本，不覆盖原生64环境最坏消息尺寸。尚未开始64环境正式DSRL训练或批量策略成功率评估。
- [部署与性能说明](walloss_batch_inference.md)、[可跟踪结果](results/walloss_batch_h200_20261104.json)；原始证据、日志、配置、精度诊断和进程 receipt 在 `outputs/walloss_batch_h200_20261104/`。以下段落的 microbatch=1 是先前阶段记录，已被本节更新。此前63.12MiB是仿真reset响应的直接JSON序列化尺寸；HTTP64MiB限制接收请求，WS限制双向接收，不能把该数值直接当成已经超限的网络请求。

## 2026-11-04 PAP 64 环境共享 World 验证

- 用户要求尝试 PAP 64 并行；新增 `src/pressb/rl_nodes/pap_vector.py`、`configs/rl_nodes/pap_vector_simulation.json` 和 `scripts/benchmark_pap_parallel.py`。旧进程式后端保留。共享单 Isaac World、CPU PhysX tensor 批量读写及 128 路 tiled RGB，相机保留原生 640×480 标定；不宣称 GPU 原生物理或 Wall-OSS batch64 推理。
- 真实 64 环境容量测试完成 640 transitions、320 条超时 episode、10 次独立 reset 隔离检查；64 个随机物体位置和 128 路有效相机均已核对。显存监测峰值 15.07 GiB、进程 RSS 12.46 GiB。测试设置 reset_render_frames=8，纯 step 35.82 transition/s，包含高频两秒超时重置后 9.27/s；8 次渲染仍可见原物体位置的灰色残影，默认配置保持 48 次，不作为正式图像配置推荐。
- 使用历史 `wallx_pap_eval_20261023/run/episodes/red_000` 中已经平滑的绝对动作，不再次平滑，在 64 环境以默认 48 次 reset 渲染重放，**64/64 物理成功**。双指接触、抬升、入盒、释放和稳定落底均通过；这是同一记录动作的物理验证，不是策略成功率。监测峰值显存 15.10 GiB、RSS 12.65 GiB。
- 新 reset 在启动时统一稳定后缓存关节，随后按索引零物理 tick 恢复，任务指纹与旧进程后端不同；每次 step 要求全部活跃环境且前缀等长。终止观测在准确 tick 缓存；已结束物体随后可能继续自然沉降，不再产生 transition。当前仅支持每个服务一种配置物体几何，位置和颜色可随机。
- 修复了 USD 角度／PhysX 弧度的运行时 PD gains 对齐，并读回核对。服务新增等长前缀的提前校验，不因无效请求污染健康状态。回归 **1,508 passed／13 skipped**。详细命令、报告和限制见 [PAP 并行容量](pap_parallel_capacity.md)，原始证据位于 `outputs/pap_parallel_capacity_20261104/`。本轮未启动正式 DSRL 长训练，未 commit/push。
- 最后默认 48 次渲染的随机位置测试也通过：128 transitions／64 次超时、两次独立 reset，128 路图像无前述灰色残影。初次 reset 26.90 秒，纯 step 39.10 transition/s，包含人为高频且拆成两次的 reset 后 2.27/s；不外推正常 20 秒任务。最大 JSON 响应 **63.12 MiB**，距现有 64 MiB 上限不足 1 MiB；仅做了直接序列化尺寸检查，尚未验证 64 路网络＋推理＋训练，正式训练前需处理通信余量和逐样本推理瓶颈。所有本轮 GPU 测试进程均已退出。

## 2026-11-04 可配置三端 RL 框架

- 新包 `src/pressb/rl_nodes`：schema-v2 能力协商、HTTP／Wall-X 风格 MessagePack WS 通信、推理上下文缓存、带身份和幂等保护的仿真服务、通用 SAC 调度。保留所有旧电梯实验入口。
- 入口：`serve_inference_node.py`、`serve_simulation_node.py`、`run_rl_training.py`；8 份配置在 `configs/rl_nodes`，完整职责、API、跨机器部署和限制见 [可配置三端说明](configurable_rl_nodes.md)。公开方法名 `dsrl` 对应底层旧 `initial_noise`，支持动作残差和原模型评估。
- PAP Wall-OSS：外部噪声 `32×20`、最终绝对关节／夹爪动作 `32×7`，新通信边界统一 rad/m；使用 checkpoint 自带归一化。缓存多模态 embeddings/RoPE，非 Transformer KV；当前真实模型 microbatch=1，不宣称 Wall-OSS 已批量加速。`ref/wall-x` 独立仓库内的模型／policy／flow_context 改动必须与 PressB 一起保留；尚未 commit/push。
- PAP 仿真：每个环境一个 Isaac 进程，支持并发命令、独立 reset、真实物理成功判定、位置／颜色／物体随机化和可选 joint EMA。此方案每环境独占渲染器，显存成本高于电梯 tensor-vectorized 方案；仍在请求批次内等待所选 workers，非完全异步环境调度。
- 训练端：维度、单位、相机／控制率启动校验；真实 terminal next_obs 入 replay；按实际 tick 时长折扣；RPC 等待期间 SAC 更新；checkpoint 数量限制和完整 replay/optimizer/RNG 恢复。resume 重新 reset 仿真，不承诺 Isaac 状态逐位续接。显式 action_bounds=clip 时，执行端与残差 critic 使用一致边界，记录裁剪统计。
- 验证覆盖 CPU 真实 HTTP／跨 Conda WS、独立子进程 IPC、两种动作／噪声几何、实际 SAC 更新和恢复。完整 Wall-OSS 权重＋真实 Isaac 图像的缓存／原生解码最大误差为 0，改变噪声会改变动作，特征保持不变。随后使用两个真实 PAP Isaac worker，在 GPU0 运行模型及仿真、GPU1 运行学习，DSRL 与动作残差各完成 **24 条 transition／22 次 SAC 更新／2 条超时 episode**，保存检查点。测试人为将 episode 限制为 2 秒，仅验证功能，未进行 PAP 长程训练或成功率评估。报告、配置、日志及权重见 `outputs/rl_nodes_smoke_20261104`。H200 原监督训练未干预。
- 电梯兼容 adapter 连接一个全新、未被占用的旧 fast simulation 服务，保留全 7 步／全部活跃环境 step 约束；中途关闭活跃 episode 后需重启旧仿真和通用 simulation 节点。桌面按键 joint6 尚无新 schema-v2 adapter，原功能不变。
- 最终 PressB 回归 **1,491 passed／13 skipped**，Wall-X flow 测试 **3 passed**。短测结束后检查缓存为空、仿真所有权释放，再正常关闭本次推理／仿真服务及两个 worker；`service_cleanup.json` 已确认进程均退出。

## 2026-11-03 新三色路径多任务模型：H200 训练队列已启动

- 用户要求用刚合并的 `piper_table_buttons_target_relative_multitask_3000_20261103` 近／中／远三份数据各训练一个模型。新监督入口为 `outputs/table_buttons_target_relative_multitask_joint6_3000_20261103`；**当前记录时正在传数据，尚未进入正式 GPU 训练**。以后读 `workflow_status.json`、`storage/transfer_status.json` 和 `preparation_worker_status.json` 判断实时状态，不要重复启动。
- 三卡各一个模型，近／中／远 GPU0／1／2；官方预训练独立初始化、不 resume，state7/action6、夹爪固定；3 epochs、batch64、905 warmup，更新数 **18,096／18,096／18,093**。每档 900／100 布局→2,700／300 episodes，三色同布局不跨集合；验证各色500 anchors，以三色等权六关节首步MAE选best。此处MAE不是任务成功率。
- 数据传输、真实三组reader检查＋batch64三步GPU短测、配额核验及正式训练监督四个后台进程已启动；`startup_receipt.json` 记录身份。全部前置检查通过后自动训练、验证best推理并将原精度best/last归档到本机 `/home/pengguanqi/Models/VLA-JEPA/table_buttons_target_relative_multitask_joint6_3000_20261103`。只保存best/final，不保存optimizer/RNG，无法精确续训。
- 新数据H200公开路径与本机同名，实体在 `/home/pengguanqi/persistent_datasets/piper_table_buttons_target_relative_multitask_3000_20261103`；模型近／中放home下 `vlajepa_target_relative_multitask_joint6_3000_20261103/runs`，远放data下 `VLA-JEPA-target-relative-multitask-joint6-3000-20261103/runs`。均为持久目录，不放 `/tmp`。
- 用户已明确授权删除**旧20261027多任务**六个best/last远端model.pt；两端原精度SHA与归档回执匹配后已执行，释放约57.38GiB。本机完整备份及两端配置／日志保留，回执在 `storage/checkpoint_cleanup_receipt.json`。另清理约9.05GiB可再建pip缓存与遗留uv安装临时目录。原始单任务／多任务数据保留。
- 三个训练配置及独立新监督脚本已准备，8项本机监督回归通过；GPU短测颜色记录使用实际 `train.jsonl` 的 `batch_color_counts`（旧trainer的probe contract没有该字段），不能据缺失字段误判短测失败。详细路径、自动流程和日志见 [本轮训练记录](../outputs/table_buttons_target_relative_multitask_joint6_3000_20261103/README.md)。

## 2026-11-03 新三色路径数据：三档各 3,000 条多任务合并完成

- 使用最新 `target_relative_v2` 九份单任务数据 `/home/pengguanqi/Datasets/piper_table_buttons_target_relative_1000_20261030`，按近／中／远分别合并红黄蓝各 1,000 条。新目录为 `/home/pengguanqi/Datasets/piper_table_buttons_target_relative_multitask_3000_20261103/{near,medium,far}_multitask`；原九份数据及旧版多任务数据均保留。
- 三档各 **3,000 episodes**，帧数依次为 **428,893／428,885／428,813**，合计 **9,000 episodes、1,286,591 帧**。保持 LeRobot v3、30 Hz、双相机 640×480、原 joint7 状态／动作及单位；六关节输出仍应在训练端选择。任务指令保持 `Press the red/yellow/blue button.`，跨间距一致。
- 全量数值逐行比对、索引／任务映射、视频原字节复制、原数据未改动检查通过；官方 reader 抽查源／合并图像一致。独立核对全部 9,000 条 episode 的源模板／位置／时序元数据，三个最终发布路径均可用官方 reader 读取。
- 各档恰好 1,000 个布局、每布局三种任务；后续训练／验证按 `layout_group_id` 分组，避免同布局跨集合。入口 `merge_summary.json` 与每组 `meta/audit.json` 均 success=true；命令、日志及独立验收见 [本轮合并记录](../outputs/table_buttons_target_relative_multitask_3000_20261103/README.md)。本轮仅合并，未启动训练或传往 H200。

## 2026-11-03 新三色路径数据的全量轨迹相似度

- 已分析 `piper_table_buttons_target_relative_1000_20261030` 全部 9,000 条／1,286,591 帧；每档 1,000 个同布局三色组，共 3,000 布局、9,000 对。排除固定夹爪与开头 0.4 秒静止，六关节实测状态减初始值，活动段按进度对齐 101 点。近／中／远总体关节位移余弦 **0.87064／0.80064／0.60576**，FK 末端平均距离 **7.592／10.159／15.455 cm**，六关节 RMSE **10.214／12.968／19.127°**。余弦不是相似百分比。
- 新数据三对红黄／红蓝／黄蓝余弦：近 **0.83054／0.86092／0.92048**，中 **0.75656／0.79009／0.85527**，远 **0.58484／0.56846／0.66398**。近距黄蓝仍最相似；新路径差异主要发生在接近阶段（0.4–4.0秒），共同下压从4.0秒起。不得沿用旧2.2／3.1秒阶段边界。
- 旧新布局完全相同子集为近965／中943／远917。关节余弦旧→新为 **0.95991→0.87468／0.89031→0.80734／0.70283→0.60696**；末端距离变化 **+33.05%／+2.77%／−8.63%**。因此不能声称所有档位的空间分离都增加。同布局对照仍同时改变路径、时序及部分可行接近姿态；再限定所有三色接近姿态与IK目标一致的880／841／750布局后，距离变化仍为+33.22%／+2.61%／−8.64%，结论不变。
- 27条原始／导出数据对照与独立XML/Rodrigues FK的135条配对阶段记录复算均通过；七项指标最大差1.07e−14。全部逐配对CSV共54,000行；CI按布局bootstrap 2,000次，seed42。公共物理时钟敏感性结论一致。未重跑仿真、未修改数据或训练模型。
- 报告、全量指标、同布局对照、独立核验及三套PNG/SVG/PDF图在 [本轮分析](../outputs/table_buttons_target_relative_similarity_20261103/README.md)。包括预先指定seed42的三档实际三色末端轨迹。上述是示范轨迹差异，不是模型语言敏感性或闭环成功率。

## 2026-11-03 核验：三色朝目标模板九组全新 1,000 条已全部完成

- 本轮已于北京时间 **2026-11-01 05:22:49** 完成：`run/suite_status.json` 为 complete，`run/completion_summary.json` success=true。九组各 **1,000 条，共全新 9,000 条、1,286,591 帧**，九份 LeRobot 导出及全量物理／动作／视频审核全部通过，无错误；3,000 个三色配对布局核对无差异。监督器已无活动子进程，无需重新启动。
- 10 月 30 日晚至 31 日晚主机曾挂起约 22 小时 26 分钟（内核 suspend 计数、BOOTTIME 与 MONOTONIC 差值及日志空档一致）；恢复后正常续采。以下为本轮启动配置记录，数据已完成，不再处于采集中。

- 用户要求使用修订后的 `target_relative_v2`，近／中／远 × 红／黄／蓝，每组 **1,000 条成功示范，共全新 9,000 条**。本次不复用旧数据或 smoke 样本；原有所有数据集保留。同颜色跨间距仍使用相同指令 `Press the red/yellow/blue button.`。
- 正式入口：`outputs/table_buttons_target_relative_1000_20261030`；北京时间 **2026-10-30 21:19:26** 启动后台监督器，历史 PID **254533**，身份记录见 `launch_record.json`。`run/suite_status.json` 为实时进度，`supervisor.log` 和 `run/logs/` 为日志。不要仅凭本文 PID 重复启动；核对当前状态与 start_ticks。
- 正式数据输出：`/home/pengguanqi/Datasets/piper_table_buttons_target_relative_1000_20261030/<near|medium|far>_<red|yellow|blue>`，每组满额后自动导出 LeRobot 并校验。原始数据、冻结配置／代码、布局及尝试记录保留在 `run/`。每档按红→黄→蓝依次采集，初始种子 0–999；物理失败会独立重试，三色实际布局配对差异写入 `pairing_report.json`。
- 两张 5090 上三个 Isaac 进程并行：near GPU0，medium／far GPU1；每进程每完成 50 条轮换一次，限制仿真缓存累积。最多两个 CPU 导出进程。采集保持固定夹爪、30 Hz 控制／图像及 120 Hz 物理；保留原七维数据格式及单位转换。
- 正式启动前，九个组合各 1 条 **9/9 首次物理按压成功**，无规划／物理重试；九份 LeRobot 导出、官方 reader、动作／接触及 18 路完整视频解码审核全部通过。证据为 `smoke/completion_summary.json` 和 `smoke/audit.json`。相关采集／导出／新模板测试 **90 passed, 1 skipped**。
- 完成证据为 `run/completion_summary.json` 与 `run/audit.json`，均 success=true；原始记录与 `run/source_archive/` 保留用于复现。此次未安排模型训练。

## 2026-10-30 三色模板修订：持续朝目标推进

- 用户认为 v1 固定基座左右中间点导致绕远，要求整体朝目标移动。当前 `configs/piper_table_buttons_color_templates.json` 改为 **`target_relative_v2`**：红色从初始末端直线到 near；黄色水平前进40%同时抬高6cm，再直线到 near；蓝色先到目标上方（基座高度至少18cm），再垂直下降到 near。三色 XY 都沿初始点到目标的线段推进，初始末端高约16.34cm。旧 shared 默认不变，v1仅保留历史复现。
- 新增 `target_relative_motion.py`，逐30Hz笛卡尔位置用有界IK求解，姿态采用随位置变化的径向yaw及局部旋转插值；生成的关节前缀同时用于碰撞检查和执行。前缀3.6秒，按压从4.0秒开始，采集成功于4.733秒；包含松开、返回的完整demo为8.7秒。新配置路径平均间距阈值3cm，不再沿用v1的10cm门槛。
- 固定历史 near seed0–19、保留原按压姿态，**19/20布局的57条路径通过全部稠密预检查**；seed9黄色靠近基座的抬升不可行，尝试其他按压姿态也未找到清晰路线，此类布局会被拒绝。有效布局平均前缀末端间距7.95cm；中、远seed0新采样布局各三色CPU预检查通过（本轮未做这两档物理执行）。
- 近距seed0–2共9次采集，**9/9首次成功、无重试**；独立重算120Hz动作／接触与18路视频解码全部通过。实测FK水平横向偏差最大0.247mm、单步后退最大0.022mm，平均三色前缀间距7.80cm。seed0三个完整demo也通过按压／回弹／返回检查。相关CPU测试160 passed、1 skipped。
- 产物：`outputs/table_buttons_target_relative_20261030/`，其中 `near_seed0_<red|yellow|blue>_third_person.mp4` 为三个独立原速全景视频，每条261帧／30fps／8.7秒。配置说明见 [三色模板文档](table_buttons_color_templates.md)。未重采正式全量数据、未训练新模型；本轮仿真已退出。

## 2026-10-30 三色独立运动模板已实现并验证

- 新增可选配置 `configs/piper_table_buttons_color_templates.json`，模式为 `color_detours_v1`。红／蓝经过基座两侧的高位中间点，黄经过更高的中间点，然后进入原有 hover／approach／press；两段各 2.4 秒五次插值，最终下压整体后移 3 秒。原默认 shared 和已有数据保持不变。
- 布局采样检查三色完整绕行及按钮间碰撞，并要求每对三色移动段的平均 FK 末端间距 ≥10 cm；这是进度对齐的平均间距，不是每一时刻的间距下限。缓存路线需匹配颜色和配置，不可混用旧布局。执行端接触保持、回撤改用规划阶段时间，数据 metadata 保留模板／时间信息。
- 近距历史 seed0–19 的相同布局、60 条新路径全部通过稠密几何预检查；规划移动段平均间距 4.08→15.52 cm，关节位移余弦 0.945→0.647。中／远 seed0 新采样布局各三条路径也通过 CPU 预检查。
- 本机 GPU0 实际运行近距 seed0–2、三色共 9 条脚本示范，全部首次按压成功，无错键或意外接触；独立 120Hz 物理／动作检查和 18 路视频完整解码通过，实际布局与旧数据完全匹配。实测 FK 移动段平均间距 **3.55→14.86 cm（4.18倍）**，整段活动轨迹 **5.65→13.11 cm**；episode **3.833→6.833秒**。这是生成器小样本验证，尚未重采正式全量数据或训练新模型。
- 新红色 3 条示范经原 LeRobot 导出器与 reader 验证通过；136 项相关 CPU 测试通过、1 项依赖测试跳过。旧 shared 的 seed0 三色完整命令序列与改动前逐元素完全一致。
- 说明和命令见 [三色模板文档](table_buttons_color_templates.md)；数值、独立审核、代码和原速三色视频见 `outputs/table_buttons_color_templates_20261030/`。采集及验证进程均已完成，GPU1 上他人的任务未改动。后续配对统计须继续核对实际 seed＋positions，因为物理失败仍按颜色独立重试。

## 2026-10-30 生成示范的三色轨迹相似度分析

- 对九份 1,000 条原始示范对应的 LeRobot 数值数据进行全量分析，确认近／中／远各 **1,000 个完整匹配三色布局**，共 9,000 条轨迹、9,000 颜色配对。原始数据与训练权重均未修改，也未重新仿真。
- 剔除开头 0.4 秒共同静止段，每条活动段按时间进度对齐 101 点。近／中／远六关节位移余弦均值为 **0.9478／0.8666／0.6700**；实测关节 FK 末端平均距离 **5.7236／9.8976／16.9687 cm**；六关节 RMSE **7.0839／12.0174／19.9618°**。余弦不是百分比，也不等于空间重合率。末端参考点使用本任务规划器的 link6 局部 +Z 0.1358 m。
- 接近／下压阶段的局部位移形状余弦接近 1，但绝对末端间距约 **8.05／13.94／23.93 cm**。所有任务共享五次插值及阶段时间表，不能仅据逐关节相关性声称轨迹相同。跨档布局不同，结果为生成数据中的配置差异，不能单独解释模型语言利用或闭环成功率。
- 27 条原始轨迹与导出表数值对照、独立 FK 抽查及全部结果键／数值检查通过；共同物理时钟对齐的平均距离变化最大仅 0.000694 cm。完整报告、逐配对 CSV、图与脚本在 [轨迹相似度分析](../outputs/table_buttons_trajectory_similarity_20261030/README.md)。

## 2026-10-29 核验：三档多任务 joint6 900 条评测已完成

- 本轮评测及审核于北京时间 **2026-10-28 16:35:44** 全部结束，用时约 **2 小时 34 分钟**。`outputs/table_buttons_multitask_joint6_3000_eval_20261028/run/workflow_status.json` 为 `complete, success=true`；近／中／远各 300 条独立动作、物理、接触与视频审核全部通过，合计 **900/900、0 missing、0 invalid**。专用推理服务和 SSH 转发已自动释放；不要重启历史监督器 PID 1878788。
- 主任务成功率：近距离红／黄／蓝 **4%／3%／4%**（11/300，3.67%）；中距离 **8%／5%／5%**（18/300，6.00%）；远距离 **17%／22%／17%**（56/300，18.67%）。整体 **85/900（9.44%）**；严格采集口径 **78/900（8.67%）**，差额 7 条均包含已声明允许的指间闭合接触。
- 815 条主指标失败中 **800 条超时、15 条非目标按键手指接触**，无错键完整按压、其他阻断碰撞或非法模型动作。仅凭超时记录不能确定训练不足、语言混淆等根因。
- 原 300 条 joint7 多任务模型合计 23/900（2.56%），本轮增至 85/900，但该比较同时改变了数据量与学习动作维度。与上一轮红色 1,000 条 joint6 单任务相比，近／中／远红色为 **3%→4%、6%→8%、44%→17%**；远距离红色多任务表现明显较低，不能笼统称三档均优于单任务。
- 最终明细见 [本轮结果](../outputs/table_buttons_multitask_joint6_3000_eval_20261028/run/results/README.md)、同目录 `summary.json` 与 `per_episode.csv`。配置仍为匹配采集间距、固定原 100 个测试布局、12 秒时限、无额外平滑和固定夹爪；以下“评测启动”段落为历史记录。

## 2026-10-28 三档多任务 joint6 3,000 条模型：并行闭环评测启动

- 用户要求在权重备份期间同时评测。本轮各模型使用对应近／中／远配置，红／黄／蓝各 **100 条**，每模型 300 条，合计 **900 条**。best step 为近／中 14,400、远 14,100；训练验证选出的 best 在仿真评测前冻结。
- 逐字复用 `outputs/table_buttons_policy_eval_20261024/run/plan.json` 对应多任务各色的 100 条布局、场景及策略种子，布局 seed10000–10099。保持 **12 秒仿真时限、30 Hz 控制／120 Hz 物理、每次执行 7 步、无额外平滑**，原生六关节预测并追加约 0.2 mm 固定夹爪全宽。保留 `task_success` 和 `strict_collector_success` 双指标及两路执行录像。
- 专用 H200 GPU0／1／2 推理服务及本机 SSH 转发端口 **18130／18131／18132** 已就绪。三色各 3 观测、每模型 9 条，全部 27 条 HTTP 推理与严格重载的离线结果逐值相同；全部三色现有客户端调用通过。模型信息、契约和验证证据在 `outputs/table_buttons_multitask_joint6_3000_eval_20261028/service/`。
- 本机后台监督器于北京时间 **14:01:29** 启动，PID **1878788**，入口 `outputs/table_buttons_multitask_joint6_3000_eval_20261028/run_evaluation.py --smoke --run`。本机 **GPU1 三路 Isaac worker**，各负责一个间距，三色顺序执行，每 10 条轮换进程。先每档一条独立 smoke 并审核，再自动正式评测；smoke 不计入 900 条。GPU0 现有 PAP 服务及独立权重备份保持运行。
- 评测、计分审核及 suite 已兼容每档三色 joint6 多任务计划，183 项相关 CPU 测试通过；独立工作流和实际计划检查通过。**三档 smoke 均通过独立物理、动作及视频审核，正式评测已开始产生有效结果**，当前权重备份仍正常进行。完成后自动全量审核物理／动作／接触及视频，生成逐颜色结果、原 300 条 joint7 多任务比较和红色 1,000 条 joint6 单任务比较，再清理本轮专用服务。
- **当前是否完成以实时文件为准，不要据历史 PID 重复启动**：`run/workflow_status.json` 汇总进展；`run/<spacing>/summary.json` 为各色实时计数；最终 `run/<spacing>/audit/summary.json` 必须均 `audit_passed=true, complete=true`，并有 `run/results/summary.json`。完整说明见 [本轮评测说明](../outputs/table_buttons_multitask_joint6_3000_eval_20261028/README.md)。

## 2026-10-28 三档 3,000 条多任务 joint6 训练完成，恢复本机备份

- **三组均完成 3 epochs，训练正常退出，最佳权重严格重载及红／黄／蓝实际指令推理全部通过。** 近／中各 14,682 updates，远 14,685 updates；北京时间 10 月 28 日训练结束时间分别为 **13:33:08／13:16:19／12:55:39**。训练用时约 17.21／16.92／16.59 小时。H200 三张训练 GPU 已释放。
- 近／中／远 best step 分别为 14,400／14,400／14,100，验证六关节 first-action MAE 分别为 **0.521472°／0.462080°／0.473927°**；这是离线动作误差，**尚未进行这三个新模型的闭环成功率评估**。每个模型验证了 9 个观测及全部三色指令，原生六维动作和追加固定夹爪接口均正常。
- 本轮 live SSH 验收证据为 `outputs/table_buttons_multitask_joint6_3000_20261027/reports/training_completion_verified_20261028.json`，三组 best／last 模型均完整存在于上一节所列 H200 持久目录。远端监督器正常终态为 `ready_for_archive, success=true`，无需重训。
- 原本机监督器 PID 1573808 因 SSH banner 超时、30 次重试耗尽而退出，**远端训练独立完成，未受影响**。故障状态和日志保存在 `recovery_history/20261028_134047/`。修复本机控制器的 SSH 暂时断连重试及恢复状态清理，8 项流程测试通过；只修改本机监督代码，没有改动已完成训练。
- 已于北京时间约 **13:41** 重启本机监督器（PID **1875157**），重新接管三组完成状态，串行备份到 `/home/pengguanqi/Models/VLA-JEPA/table_buttons_multitask_joint6_3000_20261027`；已确认 near 进入 `copying`。**本机归档尚未完成，H200 新权重保留。** 后续以实时 `workflow_status.json`、`groups/<model>/archive_status.json` 和模型目录 `durable_archive.json` 判断状态，不要依据旧 PID 或历史日志重复启动训练。

## 2026-10-27 三档 3,000 条多任务 joint6 训练启动

- 用户要求使用三个合并数据集在 H200 分别训练。正式流程已于北京时间 **20:13 左右**启动，本机监督 PID **1573808**；实时状态以 `outputs/table_buttons_multitask_joint6_3000_20261027/workflow_status.json` 和 `startup_receipt.json` 为准，不要重复启动。入口为该目录 `code/scripts/run_workflow.py`，计划为 `plan.json`。
- `near_multitask_joint6_3000`、`medium_multitask_joint6_3000`、`far_multitask_joint6_3000` 分别使用 H200 **GPU0／1／2**。沿用原生 **state7/action6、不学习夹爪、执行时固定全开度约 0.2 mm**；每组从同一官方预训练重新开始，四个任务接口重新初始化，不 resume 旧模型。
- 每组红／黄／蓝各 1,000 条，三种指令按样本读取。相同布局的三色按 seed42 分组划分为 **2,700 train／300 val**（900／100 个实际布局），保留原 red1000 的 source-ID 划分；评估 seeds10000–10099 未混入。训练 **3 epochs、batch64**，near／medium 为 **14,682 updates**、far **14,685**；warmup734、eval_interval300，验证三色各 500 anchors，以三色等权六关节 first-action MAE 选 best。
- 三份数据已传至 H200 `/home/pengguanqi/Datasets/piper_table_buttons_multitask_3000_20261027`（实体在 `/data/scratch`），全部文件核对一致；复用只读红色视频节省约 5.78 GiB。57 项 CPU 回归、三个真实数据集读取检查和三步 batch64 GPU 短测通过；短测实际输出 `[1,7,6]`、峰值显存 **43.76 GiB**，每批包含三色且参数发生更新。额外监督流程 5 项测试通过。独立审查发现并修正了 probe 元数据重复写入覆盖颜色计数的问题，不影响训练计算。
- **用户已明确授权并完成清理**上轮三个 red_joint6_1000 模型在 H200 的 best／last 模型文件，释放约 **47.8 GiB**；本机原精度备份与两端元数据保留。具体范围、两端重新核对和清理回执在本轮 `storage/checkpoint_cleanup_receipt.json`。旧模型后续推理应使用本机备份，若需 H200 则重新上传。
- 新权重只保留 best／final，无 optimizer/RNG；near／medium 写 `/home/pengguanqi/vlajepa_multitask_joint6_3000_20261027/runs`，far 写 `/data/scratch/pengguanqi/VLA-JEPA-multitask-joint6-3000-20261027/runs`，均不用 `/tmp`。训练后同卡严格重载 best，实际验证三个颜色指令，再串行归档至本机 `/home/pengguanqi/Models/VLA-JEPA/table_buttons_multitask_joint6_3000_20261027`。工作流以实际备份文件锁协调，避免旧归档流程的重复启动误判。
- 日志每约 30 秒同步至本机 `groups/<model>/train.jsonl` 和 `train.log`；H200 监督目录为 `/home/pengguanqi/Worksapce/Research/VLA-JEPA/runs/table_buttons_multitask_joint6_3000_20261027`。完整协议见 [本轮训练说明](table_buttons_multitask_joint6_3000_training.md)。训练和最终归档尚未完成，不能将离线验证误差解释为闭环成功率。

## 2026-10-27 三档 3,000 条多任务数据集合并完成

- 用户要求将九份各 1,000 条单任务数据按近／中／远合并。本机新目录为 `/home/pengguanqi/Datasets/piper_table_buttons_multitask_3000_20261027`，子目录 `near_multitask`、`medium_multitask`、`far_multitask` 各 **3,000 episodes（红／黄／蓝各 1,000）**；帧数分别 **348,008／348,016／348,036**，合计 **9,000 条、1,044,060 帧**。根目录 `merge_summary.json` 和三份 `meta/audit.json` 均成功。
- 原九份 `/home/pengguanqi/Datasets/piper_table_buttons_1000_20261025` 及旧 100／300 条版本保留。原 joint7 数值、30 Hz 时间与两路 RGB 视频保持一致，视频独立复制且不重编码；只重映射 episode／全局帧／任务索引并重算数值统计。三档语言指令保持 `Press the red/yellow/blue button.` 各自一致。
- 每档 1,000 个真实布局由三色任务共享。新的 `meta/export_manifest.json` 保留源 ID、seed、位置及 `layout_group_id`，后续划分训练／验证时按布局分组。`scripts/merge_table_buttons_datasets.py` 已支持动态数量及 `--expected-episodes-per-color 1000`，兼容原 100 条输入；8 项相关测试通过。
- 日志与最终目录读取核查在 `outputs/table_buttons_multitask_3000_20261027`，使用说明见 [桌面按键数据集文档](table_buttons.md#expanded-multi-task-datasets--2026-10-27)。此次只完成本机数据合并，未启动新训练或上传 H200。

## 2026-10-27 三档红色 joint6 1000 的 300 条闭环评估完成

- **新权重本机归档也已于北京时间 2026-10-27 14:07:52 全部完成**：训练 workflow 为 `completed, success=true`，三组 `durable_archive.json` 均 `complete=true`。原精度 best／last 及元数据已校验发布至 `/home/pengguanqi/Models/VLA-JEPA/table_buttons_red_joint6_1000_20261026/<near|medium|far>_red_joint6_1000`；far 的 best 与 last 相同，已用硬链接共享存储。此前总归档流程误判 medium 工作进程退出并发生锁冲突；medium 实际已完成，`recovery_archive_20261027` 仅复用 near／medium 回执并续传 far，没有重训或删除新远端模型。三组完成回执为 `outputs/table_buttons_red_joint6_1000_20261026/completion_summary.json`，不需再次启动归档。
- **正式评估于北京时间 2026-10-27 00:28:56 完成**，约 46.5 分钟。`outputs/table_buttons_red_joint6_1000_eval_20261026/run/workflow_status.json` 为 `complete, success=true`，三组 `audit/summary.json` 均 `audit_passed=true, complete=true`，合计 **300/300、0 missing、0 invalid**，所有执行录像已审核。
- 每组 100 条：**near 3/100（3%）、medium 6/100（6%）、far 44/100（44%）**；严格采集口径分别 **3/100、5/100、38/100**。总任务成功 **53/300（17.67%）**，严格成功 **46/300（15.33%）**。247 个主指标失败全部为 12 秒超时，没有错键、非目标接触、阻断碰撞或非法动作；medium 1 次、far 6 次任务成功含指间闭合接触，故严格口径不通过。
- 同布局原 100 条训练 joint7 基线 near／medium／far 分别 **0/100、1/100、0/100**；旧 medium joint6 为 **3/100 task、2/100 strict**。本轮采用相同 seed10000–10099、匹配场景、12 秒及无平滑。joint7 比较同时改变训练数据量和动作维度，不能单独归因于扩充数据；近／中成功率仍低。
- 报告：[正式结果](../outputs/table_buttons_red_joint6_1000_eval_20261026/run/results/README.md)，机器可读 `run/results/summary.json`，逐条对照 `run/results/per_episode.csv`。`service/cleanup.json` 为 `success=true, all_stopped=true, dry_run=false`，三个专用 H200 服务及 SSH 转发已释放。历史评估 PID1403937 已完成，不要重复启动。

## 2026-10-26 23:42 三档红色 joint6 1000 训练完成；300 条闭环评估启动（历史）

- 此节覆盖下方白天的恢复／等待状态。九份旧 joint7 模型于 **17:11** 完成本机原精度备份并清理 H200 原权重；三组新训练随后并行启动，均完成 **3 epochs / 4,896 updates**。near／medium／far 的最佳 step 分别为 **4100／4100／4896**，训练及 GPU 推理检查分别于 **22:57／23:02／23:04** 全部通过。新权重归档回本机仍由原 workflow 自动进行，不影响评估。
- 用户已要求同时评估三个新模型。正式入口为 `outputs/table_buttons_red_joint6_1000_eval_20261026`，后台 PID **1403937**（以 `startup_receipt.json` 和 `/proc` start_ticks 核对）。`run/workflow_status.json` 聚合进度；`run/<near|medium|far>/summary.json` 为各组实时结果。不要依据旧 PID 重启。
- 每组 **100 条、共 300 条**，严格复用原 near_red／medium_red／far_red 的全部 seed10000–10099 布局、配置、policy seed、12 秒时限和无平滑设置；state7/action6、固定夹爪全宽约 0.2 mm。任务成功与严格采集成功双指标沿用原口径，基础设施失败单独记录。
- **本机 GPU1 三个 Isaac worker 并行**，每组一个，10 条轮换一次，全部录像；H200 GPU0／1／2 分别服务三个最佳权重，专用端口 **18120／18121／18122**，HTTP 输出与独立 GPU 参考共 9 个观测逐值一致。本机已有 PAP 服务和训练归档不动。
- 正式启动前，三个模型各 1 条物理预检均通过独立动作／接触／视频审核；三条均超时，不计入正式分母。新 joint6 计划审计支持三档 1000 条训练身份，139 项相关 CPU 测试通过，旧 medium_red_joint6 协议保留。
- 正式结果需各 `run/<spacing>/audit/summary.json` 为 `audit_passed=true, complete=true`，合计 300 条。工作流自动生成 `run/results/` 与旧 joint7、旧 medium joint6 比较，再仅清理本轮专用服务与 SSH 转发。完整说明见 [本轮评估说明](../outputs/table_buttons_red_joint6_1000_eval_20261026/README.md)。

## 2026-10-26 九组扩充完成；三档红色 joint6 重训与持久存储

- **2026-10-26 下午恢复更新**：初次备份队列在北京时间 03:39 因同 UID 的 `sshd/systemd/sd-pam` 等辅助进程 `/proc/.../fd` 不可读而退出，后续训练队列同步退出，三组 GPU 训练因此尚未启动。13:36 左右已修复逐项权限兼容并通过 23 项回归；可见模型读取引用仍阻止清理，不可读条目如实记入回执。原失败状态与启动记录保存于两运行目录的 `recovery_history/20261026_133520/`。
- 当前恢复后的备份 PID **914515**、训练 workflow PID **914526**，以实时状态及 start_ticks 核对，不要依据旧 PID2396412/2402961 重启。首份 `near_red` 本机完整备份已复用，35 份元数据及模型重新校验后，远端原权重已清理，释放约 9.56 GiB；正在备份 `near_yellow`。持久化数据迁移已实测成功，公开数据路径现已解析到 `/data/scratch/pengguanqi/Datasets/piper_table_buttons_1000_20261025`，原 344,741,327 bytes 中断前缀已校验复制并续传。训练仍需等九个旧权重全部备份清理及余下数据／CPU 验收通过。
- **扩充数据已全部完成**：北京时间 2026-10-26 02:53，全量 `audit.json` 与 `completion_summary.json` 成功，九组各 1,000、共 9,000 条（原 900 + 新 8,100）；数值、物理轨迹及完整视频审核通过。入口仍是 `outputs/table_buttons_dataset_1000_20261025`。这覆盖下方“后台采集中”的历史状态。
- 用户确认这次训练 **近／中／远三个红色单任务模型，原生六关节输出**：`near_red_joint6_1000`、`medium_red_joint6_1000`、`far_red_joint6_1000`。每组从相同官方 base 全新初始化，state7/action6、夹爪固定约 0.2 mm；900/100 划分，3 epochs、4,896 updates、warmup245、batch64、500 验证 anchors。新旧离线 holdout 不同，不直接比较其 MAE；独立仿真评估 seed10000+ 未进入训练。
- 用户进一步要求旧 100 条权重备份／清理以释放配额，新数据不要放 `/tmp`，并减少中间 checkpoint。**当前采用 `best_and_final_model_only`**：只在验证改进时替换单份 best，最终另存 last；不保存 optimizer/RNG，不支持从中间步精确 resume。原子保存峰值已通过测试，每 run 最多两份独立模型，共约 57.4 GiB；最终最佳若同 step 则 hardlink。旧 100 条保存／resume 行为及旧数据 split、fingerprint、contract 均保持兼容。
- 旧九个 joint7 单任务权重约 86.1 GiB，之前本机没有备份；备份清理队列为 `outputs/table_buttons_joint7_retirement_20261026`，状态 `retirement_status.json`。按 scratch4→home5 顺序，原精度模型与配置／processor／日志先备份至 `/home/pengguanqi/Models/VLA-JEPA/table_buttons_joint7_20261023/<group>`；本机 SHA 与持久回执通过后，再复核远端源并只删该 `model.pt`，保留远端 metadata 与 `retirement.json`。不要删除电梯模型、三份多任务模型或旧 joint6 优化器；它们不属于此清理队列。
- **两个后台流程已实际启动**：旧权重备份清理 PID2396412，初始状态为 `near_red` 的 `copying_model`，已观察到模型传输增长；新持久训练 workflow PID2402961，初始状态为 `waiting_for_persistent_storage_release`。两者各有 `startup_receipt.json` 和进程锁；实时状态优先于本文 PID。数据处理与调度准备完成，GPU 训练尚须等待存储门禁通过。
- 新训练入口是 `outputs/table_buttons_red_joint6_1000_20261026/workflow_status.json`，代码／配置已部署 H200。数据传输先等 `persistent_storage_ready.json`；正式训练还需旧九模型全部备份清理完成、真实配额检查和三组远端真实 reader 测试全部通过。三组分别 GPU0/1/2，训练后同卡验证，再原精度归档回本机；按实时状态判断是否已开始 GPU 训练，**不要重复启动**。
- 新数据持久位置为 `/home/pengguanqi/Datasets/piper_table_buttons_1000_20261025` → `/data/scratch/pengguanqi/Datasets/piper_table_buttons_1000_20261025` 实体目录。near/medium 训练权重写入 `/home/pengguanqi/vlajepa_red_joint6_1000_20261026/runs`，far 写入 `/data/scratch/pengguanqi/VLA-JEPA-red-joint6-1000-20261026/runs`，均不使用 `/tmp`。正式启动前检查 home≥40 GiB、数据盘≥22 GiB 的剩余用户配额（数据已传完）；不能用 `df` 的整盘空闲代替 quota。
- 初始临时存储尝试的 workflow PID2301601 已按用户新要求停止，尚未训练，仅传输了约 329 MiB near_red 数据；旧计划／状态保存在新运行的 `storage_attempt_tmp/`。新版会在持久配额足够后迁移这些 partial 数据再续传。新源码不能按旧 tmp 计划启动；权重归档不再宣称远端保留 optimizer。
- 完整协议及日志位置见 [训练说明](table_buttons_joint6_1000_training.md)、`outputs/table_buttons_red_joint6_1000_20261026/README.md`；独立审查为 `reports/final_training_review.json`，旧 tmp 方案审查另存历史文件。此轮尚无新的模型仿真成功率。

## 2026-10-25 中距离红色 joint6 训练与 100 条闭环评估完成

- `medium_red_joint6` 已完成 **3 epochs / 492 updates**，最佳为 step 492，六关节 first-action MAE **1.638603°**（离线误差，不是 SR）。checkpoint SHA 为 `f86b4a4546a0a6cf7089174185af0c1cfb63f26a8b1346d1995fe7287cb94097`。训练使用原 100 条数据；本机原精度 best/last 与元数据已完成校验和归档，回执为 `/home/pengguanqi/Models/VLA-JEPA/table_buttons_joint6_20261025/medium_red_joint6/durable_archive.json`。
- **正式评估已于北京时间 2026-10-25 19:02:55 完成，100/100 条全部通过独立审计，0 missing、0 invalid。** 任务成功 **3/100（3%）**，严格采集口径 **2/100（2%）**；原 joint7 medium_red 对应为 **1/100（1%）**、**0/100（0%）**。新模型 95 次超时、2 次异常接触；无错键、非目标按键触碰、非法动作或关节限位裁剪。净增加 2 次成功，不能据此认定有稳定收益，整体成功率仍低。
- 正式计划复用原 `outputs/table_buttons_policy_eval_20261024/run/plan.json` 的 `medium_red` 全部 100 行：seed 10000–10099、相同配置和 policy seed、12 秒时限、无平滑、原双成功指标。模型原生 state7/action6/noise6，执行接口追加固定全夹爪开度约 0.2 mm。严格指标额外拒绝实际指间闭合接触；本次 3 次任务成功中有 1 次因此未通过严格指标。相同 seed 不表示 6D/7D 噪声逐元素相同，也不承诺 RTX 画面逐像素相同。
- 全流程耗时约 **37 分钟**，使用本机 GPU0 单 worker（每 10 条轮换）与 H200 GPU0 独立 18113 服务。`outputs/table_buttons_medium_red_joint6_eval_20261025/run/workflow_status.json` 为 `state=complete, success=true`，`run/audit/summary.json` 为 `audit_passed=true, complete=true`；`run/service_cleanup.json` 确认专用推理服务与 SSH 转发已清理。历史 workflow PID 2227446 已完成，不需重启；GPU1 数据扩充不受此清理影响。
- 结果见 [比较报告](../outputs/table_buttons_medium_red_joint6_eval_20261025/run/results/README.md)、`run/results/comparison.json`、`per_episode.csv`。配对结果为新模型独有成功 3 次、旧模型独有成功 1 次、同时成功 0 次。独立 smoke 不计入正式 100 条；新 client 保留每次原生六维动作和实际初始噪声供审核，旧 joint7 client 与采集冻结代码未改。

## 2026-10-25 桌面按键九组数据扩充至每组 1,000 条（后台采集中）

- 本轮用户要求九个单任务数据集从各 100 条扩充到各 **1,000 条**。目标共 **9,000 条：原 900 条 + 新采 8,100 条**。原 `/home/pengguanqi/Datasets/piper_table_buttons_20261023` 和旧 raw 均保留；扩充版单独发布至 `/home/pengguanqi/Datasets/piper_table_buttons_1000_20261025/<near|medium|far>_<red|yellow|blue>`。
- **后台监督器已于北京时间 2026-10-25 17:26 启动，PID 1871885。实时状态优先读** `outputs/table_buttons_dataset_1000_20261025/suite_status.json`；日志为 `supervisor.log` 和 `logs/`。入口 `scripts/run_table_buttons_collection_suite.py` 默认只准备，`--run` 才采集。不要依据本文的历史 PID 重复启动，先检查实时状态与进程。
- 每组新 raw 独立复制原 episode 0–99，再从 100 续采至 999；首轮布局 seed 同为 100–999，与现有评估 10000–10099 分离。配置、轨迹、相机、材质、光照与物理成功标准均沿用原采集。语言指令只依赖目标颜色。7D 数据语义保持六关节角度（度）+ 夹爪全开度（厘米），当前 joint6 实验仅在训练时去除夹爪输出，不改这些数据。
- 原 900 条的独立复制、原 raw/LeRobot 未改动及配置一致性已通过核查；关键 CPU 测试 13 项通过。三档首批新增样本各 3 条、共 9 条已通过独立物理轨迹与 18 路完整视频解码检查，见 `reports/new_samples_audit.json`；这不是全量验收。采集使用 **GPU1 三路**（近/中/远），每进程新增 50 条后自动轮换；每组满 1,000 条后用现有导出器生成完整 LeRobot v3，最多两路 CPU 导出。19 个相关源码与各分片配置已冻结，重启前核对，避免长程中混入配置变化。
- 预计 **9–11 小时**，以实时进度为准。只有 `suite_status.json` 为 `complete`、`completion_summary.json` 为成功且最终 `audit.json` 成功确认 **9,000 条**时，才算整个任务完成。按颜色独立重试可能造成少数布局不配对，实际 seed/位置差异保存于 `pairing_report.json`。本轮不启动新训练；H200 上 medium_red_joint6 仍使用原 100 条训练。
- 详情与证据见 [扩充运行说明](../outputs/table_buttons_dataset_1000_20261025/README.md)。导出器会正常重新编码复用的 raw 视频；原始 100 条版本文件本身保持不变。

## 2026-10-25 中距离红色单任务：去除夹爪学习维度重训

- 用户指定只重训 `medium_red`，新实验为 `medium_red_joint6`：保留真实 **7 维状态输入**，动作目标、flow noise、模型输出及训练损失均为 **6 维手臂关节**。推理适配器追加数据集中固定的夹爪开口（约 **0.2 mm 全宽**），不再预测夹爪。选模只使用 `first_joint_mae_deg`。
- 沿用原 100 episodes、90/10 划分、batch 64、学习率、7-step chunk 和 **3 epochs / 492 updates**；从相同官方预训练权重重新初始化四个动作/状态接口，不 resume 旧模型。原数据与十二个既有模型保持。新旧 split 逐字节相同；22 项训练 CPU 回归、8 项推理 CPU 检查及真实 batch64 两步 GPU 短测通过，短测输出 `[1,7,6]`，显存峰值 43.76 GiB。
- **正式训练已在 H200 GPU0 后台启动。实时状态优先读**本机 `outputs/table_buttons_medium_red_joint6_20261025/archive_status.json`、`train.jsonl` 和 `train.log`，不要据本文重复启动。远端监督入口为 `/home/pengguanqi/Worksapce/Research/VLA-JEPA/runs/table_buttons_medium_red_joint6_20261025/supervisor_status.json`。日志约每 20 秒同步本机。
- 因远端 home/scratch 配额不足，大型 checkpoint 与优化器写入 `/tmp/pengguanqi-table-buttons-joint6-20261025/medium_red_joint6`。监督流程在训练结束后自动用 GPU1 严格加载最佳权重、核对六维生成与固定夹爪接口，再将原精度 best/last 模型、processor 和完整小型元数据归档至本机 `/home/pengguanqi/Models/VLA-JEPA/table_buttons_joint6_20261025/medium_red_joint6`。以 `archive_status.json` 的 `state=completed` 和模型目录内 `durable_archive.json` 为最终归档完成证据；本机 last 为 model-only，优化器保留远端，不能当作本机完整恢复 checkpoint。
- 新代码在 H200 VLA-JEPA：独立 joint6 数据适配器、推理模块与服务，以及兼容 joint7 的训练入口分支。原十二模型推理服务未修改。详情与代码快照见 [实验记录](../outputs/table_buttons_medium_red_joint6_20261025/README.md)。本轮只重训和检查推理完整性，尚未评估新模型仿真成功率；旧评估 audit 的初始噪声检查为 joint7，后续评估 joint6 时需按原生 `[1,7,6]` 噪声核对。

## 2026-10-25 桌面按键 12 个模型训练与闭环评估完成

本节覆盖此前评估“运行中”及下方 2026-10-23 的“运行中／等待接续”状态；历史协议与预检记录保留用于复现。训练完成、权重归档和本次完整闭环审核均已核对。

- **九个单任务模型和三个多任务模型均已完成 3 epochs**：九组单任务分别完成 **492 updates**，三组多任务分别完成 **1,470 updates**。两套实验独立从官方预训练权重开始；多任务模型没有接着九个单任务模型的权重训练。原九份数据保持不变，新三份数据是同间距下红、黄、蓝示范的合并版本。
- H200 的两个统一入口分别为 `/home/pengguanqi/Worksapce/Research/VLA-JEPA/runs/table_buttons_joint7_20261023` 和 `/home/pengguanqi/Worksapce/Research/VLA-JEPA/runs/table_buttons_multitask_joint7_20261023`。两者的 `final_verification.json` 与 `validation/final_models.json` 均为 `success=true`；十二组各自的 `completed.json` 均为 `success=true`、`epoch=3`。这些训练完整性验收包含严格重载最佳权重及实际动作预测，其 `task_success_rate_evaluated=false` 仅描述当时的离线验收；之后完成的正式闭环结果见下。
- 三个多任务最佳模型已经按原精度归档到本机 `/home/pengguanqi/Models/VLA-JEPA/table_buttons_multitask_joint7_20261023/<group>/checkpoints/best/model.pt`，`<group>` 为 `near_multitask`、`medium_multitask`、`far_multitask`，对应 best step 为 **1400 / 1300 / 1100**。各 group 目录同时保存 processor、config、split、归一化和训练日志。根目录 `durable_archive_completion.json` 为 `status=completed`、`completed_count=3`，三组归档均完成模型字节数／SHA256 校验及 H200 回执；本次复核三个模型文件存在、大小均为 10,268,096,930 字节。
- 多任务 H200 `suite_status.json` 留在远端训练监督器的终态 `ready_for_local_archive`，本机归档监督器已经完成后续工作，应结合 `durable_archive_completion.json` 判断最终状态。本机 `outputs/.../suite/suite_status.json` 是此前复制的历史快照，仍可能显示训练或等待中；不要因此重启训练。
- **正式闭环评估已完成并通过独立审核：1,800/1,800 episodes，0 missing、0 invalid。** `task_success` 为 **37/1800（2.06%）**，`strict_collector_success` 为 **13/1800（0.72%）**；单任务主指标 **14/900（1.56%）**，多任务 **23/900（2.56%）**。正式计划为 `outputs/table_buttons_policy_eval_20261024/run/plan.json`：18 个模型—任务组合各 100 条，种子 **10000–10099**，匹配原始 `near/medium/far` 采集配置；单任务与多任务模型共享对应布局和请求随机种子。逐颜色结果及完整双指标见 [评估说明](table_buttons_policy_eval.md)、[本机结果汇总](../outputs/table_buttons_policy_eval_20261024/run/results/README.md) 和 [逐组合 CSV](../outputs/table_buttons_policy_eval_20261024/run/results/per_cell.csv)。
- 评估于**北京时间 2026-10-25 03:16:54**结束（UTC 2026-10-24 19:16:54），墙钟耗时约 **4 小时 50 分钟**；之后完成完整轨迹审核和录像解码。`run/workflow_status.json` 与 `run/suite_status.json` 均为 `state=complete`，`run/audit/summary.json` 为 `audit_passed=true`、`complete=true`；最终结果以此审核文件为准。`run/service_cleanup.json` 确认本轮三个 H200 推理服务均已自动结束。历史监督 PID 为 3915546；运行曾使用本机 GPU1 的 3 个 Isaac worker、每进程 20 episodes 轮换，以及 H200 18110/18111/18112 三个四模型缓存服务，当前无需继续盯守或重启。
- 正式运行前已冻结双指标口径：`task_success` 仅允许准确匹配 `/World/Piper/link7` ↔ `/World/Piper/link8` 的手指闭合接触后继续任务；`strict_collector_success` 同时保留原采集器对该接触判失败的标准。两项使用**相同有效 episode 分母**，另记录手指闭合接触率；其余错按、非目标触碰、异常自碰和环境碰撞仍判失败。完整保留模型预测的七维动作（包括夹爪开度），执行端只做物理关节限位裁剪，**不固定夹爪、不添加平滑**。
- 原始严格口径 `outputs/table_buttons_policy_eval_20261024/plan.json` 和三个 `smoke_<spacing>_multitask_red` 预检保留；正式计划通过 `parent_plan` 记录口径变更。随后三个 `smoke2_<spacing>_multitask_red/audit.json` 全部通过独立审核：medium 出现物理任务成功，near/far 超时，medium 的严格采集器指标仍为失败。它们只验证执行与计分链路，**不计入正式 1,800 条评估，也不作为正式 SR**。十二模型 HTTP batch=1 输出与独立离线重载结果逐值一致，证据为 `service_reports/summary.json`；本机已保存 `run/source_archive/` 源码快照。
- 全量终止原因以**超时为主（1,722 条）**，另有 30 条异常接触、11 条非目标手指接触。`outputs/table_buttons_policy_eval_20261024/diagnostics/first_nine_single_red.json` 只读分析了三档单任务红色模型最初各 3 条：观察到悬停、游移或下压不足，未发现这些样本中的数值桥接错误。这是九例失败形态诊断，不能代表所有失败，也不能据此确定训练不足等根因。

## 2026-10-23 三份多任务数据与接续训练队列

- 按用户要求将同间距下的红、黄、蓝合并为 `near_multitask`、`medium_multitask`、`far_multitask`，每份 300 episodes（三色各 100），帧数分别 34,802 / 34,804 / 34,804；它们复用同一批 900 条示范，不是重新采集。原九份数据逐文件哈希不变。
- 本机和 H200 均保存在 `/home/pengguanqi/Datasets/piper_table_buttons_multitask_20261023`；传输完整校验一致。合并入口 `scripts/merge_table_buttons_datasets.py` 使用官方 LeRobot 0.6.1 聚合，无视频重编码；任务、episode、全局帧索引正确重映射，数值统计重算。独立审核确认全部 104,410 行、18 个视频及 episode 相机偏移与源一致。
- H200 新增独立 multi-task adapter/trainer，不改正在运行的九组训练文件。每条样本读取自身颜色语言指令；按共享原 episode ID/seed/完整布局划分 270 train / 30 val，三色分别 90/10；同布局的不同颜色任务不能跨训练与验证。固定验证布局 ID `[18,21,27,33,42,50,56,59,71,92]`，150 anchors（三色各 50），逐色及总体指标均保存，best 采用三色等权动作误差分数。
- 每个新模型独立从相同官方预训练开始，joint7 四接口重新初始化，不继承原九个微调模型；沿用固定 joint7 归一化、batch 64、3 epochs，实际每组 1,470 updates。原九组每组 492 updates 的配置保持不变。
- 新接续控制器已经后台启动，目前状态为 `waiting_for_previous_suite`。H200 统一入口：`/home/pengguanqi/Worksapce/Research/VLA-JEPA/runs/table_buttons_multitask_joint7_20261023`，先读取 `suite_status.json`，不要重复启动。严格等待原九组训练、归档和最终权重验证成功，再用 GPU0/1/2 并行训练 near/medium/far 三模型；训练结束后逐个严格加载原精度和 processor，按三种语言生成动作作完整性验收。这不是仿真 SR 评估。
- H200 持久配额不足追加三份完整模型：新 best 暂存在 `/tmp/pengguanqi-table-buttons-multitask-20261023/<group>/checkpoints/best`，远端统一入口提供对应链接和元数据。本机后台归档监督器在训练/验证完成后，将原精度模型、processor、config、split、归一化及完整日志复制到 `/home/pengguanqi/Models/VLA-JEPA/table_buttons_multitask_joint7_20261023/<group>`；确认实际字节一致、原子发布后写远端 `durable_archive.json`，保留 H200 临时 best 副本。
- 本机实时状态为 `outputs/table_buttons_multitask_h200_training_20261023/local_archive_status.json`（包含远端队列状态，约每 30 秒更新），日志 `local_archive.log`；最终归档完成时 Models 根目录产生 `durable_archive_completion.json`。输入、传输、训练配置和独立审计证据也位于该 outputs 目录。新合并 6 项、适配/训练 23 项、独立验收 12 项、队列/归档 5 项 CPU 测试通过；未提前占用原训练 GPU。

## 2026-10-23 桌面按键九组 VLA-JEPA 独立训练（运行中）

- 用户要求九份数据全部传到 H200 并分别训练。已传至 H200 `/home/pengguanqi/Datasets/piper_table_buttons_20261023`，900 episodes、104,410 帧，全文件校验一致。训练仓库为 `/home/pengguanqi/Worksapce/Research/VLA-JEPA`（路径中的 `Worksapce` 是实际拼写）。
- 新增独立 `starVLA/dataloader/table_buttons_lerobot.py`、`starVLA/training/train_table_buttons.py`、suite controller 与验收脚本。使用双相机 `up/wrist` 和真正 joint7（六关节度数、夹爪全开度厘米），固定除以 `[180,180,180,180,180,180,8]`；没有复用电梯的 pose9/floor 数据逻辑。旧电梯代码和本机冻结推理 clone 保持原样。
- 每组从同一官方预训练权重独立开始，四个动作/状态接口重新初始化；seed 42 共用 90/10 整 episode 划分；batch 64、3 epochs、每组 492 updates。没有从旧电梯、RL 或其他组微调权重恢复。GPU 0/1/2 分别依次训练红/黄/蓝的 near、medium、far。
- 九组统一入口（H200）：`runs/table_buttons_joint7_20261023/suite_status.json`；配置在 `configs/`，控制器日志 `logs/controller.log`，各组日志 `logs/<group>.log`。不要仅依据本文的“运行中”启动重复进程，应先读实时状态。训练过程 JSONL 在 `/tmp/pengguanqi-table-buttons-20261023/<group>/train.jsonl`，完成后复制至统一入口的 `<group>/`。
- 因 H200 用户磁盘配额有限，训练中的 best/last 暂存 `/tmp/pengguanqi-table-buttons-20261023`；完成后仅归档 best 模型及完整小型训练元数据。前四组 near_red/near_yellow/near_blue/medium_red 在统一入口 `<group>/checkpoints/best`，后五组实际在 `/home/pengguanqi/vlajepa_table_buttons_models_20261023/<group>/best`，统一入口保留软链接。归档大小与内容一致验证后，才清理本轮对应临时 checkpoint。
- 验证选取固定 50 个 held-out anchors，以 `first_joint_mae_deg + 22.5 × first_gripper_mae_cm` 选择 best；这是离线动作预测误差，不能当作闭环任务成功率。尚未运行这九个模型的 Isaac 闭环评估。
- 已验证首批三组第 100 步验证及 best/last 真实保存，随后继续训练。独立后台 `scripts/finish_table_buttons_suite.py` 等待九组归档结束，再用 GPU0 逐一严格重载 best 并生成动作，结果写统一入口 `validation/final_models.json` 与 `final_verification.json`；这仍不是 Isaac 成功率评估。
- 本机记录在 `outputs/table_buttons_h200_training_20261023`；H200 使用文档为 `docs/table_buttons_training.md`。适配器真实九组数据检查 30 项通过，trainer 9 项通过，controller 4 项通过；batch64 短测峰值显存 43.72 GiB、第二次 update 2.72 秒，三个可训练组件的参数确有更新。正式运行状态以 suite_status 为准。

## 2026-10-23 桌面按键三档间距与 900 条数据

- 新增 `near/medium/far` 三档随机间距，约束所有按键两两中心距离为 6–10 / 10–18 / 18–30 cm；保留完整相机可见且可执行区域的采样与碰撞检查，不固定三键相对形状。用户确认本次仍用红、黄、蓝，“绿”为笔误。
- 已完成三档 × 三色 × 100，共 **900 条成功物理示范、104,410 个 30 Hz 采样帧**，九份独立 LeRobot v3.0 数据集保存在 `/home/pengguanqi/Datasets/piper_table_buttons_20261023`。每个颜色的指令在各档完全一致，格式为 `Press the red button.`。每组 100 个不同布局，优先使用共同 seed 0–99 以便跨任务比较。
- 数据为 `observation.images.up/wrist` 两路 640×480 RGB；state/action 为六关节角（度）与夹爪**全开度（厘米）**，匹配 PAP 的 7D 接口。动作对应观测后四个 120 Hz 物理步的保持目标；包含 t=0 到首次物理成功后的采样边界，不含撤回，末帧动作重复前一目标。
- 入口是 `scripts/collect_table_buttons.py`、`scripts/export_table_buttons.py`、`scripts/audit_table_buttons_dataset.py`，方法为脚本 IK 示范，不是 VLA/RL 成功率评估。独立审核对全部 900 条重算接触成功、动作/状态时间映射、视频帧与布局，九组官方 LeRobot 读取检查全部通过。原始数据、日志、源代码快照和汇总在 `outputs/table_buttons_dataset_20261023`。
- 独立 `.conda/envs/lerobot` 已安装（Python 3.12，LeRobot 0.6.1、CPU PyTorch 2.10）；有 `--lerobot-root` 的联合审核需使用此环境，原 Isaac 环境没有 pyarrow。Isaac 保持原 Python 3.11 环境。
- 批量采集时必须控制进程寿命，避免反复建 stage 的 CPU 缓存累积。使用 `--max-new-episodes 50 --resume`，退出码 75 / `restart_required` 后重启同命令；退出 0 才是整个配额完成，示例循环见 `docs/table_buttons.md`。保留本机有效 DISPLAY/XAUTHORITY：直接清空的独立试验返回空相机图像，已撤销该试验。实际采集前中断均保留并在原 seed 重采，正式数据无不完整或重复 episode。
- 采集及导出已结束，GPU1 已释放；GPU0 原 Wall-X 推理服务保留。当前这些新增采集与间距改动尚未 commit/push。

## 2026-10-23 三任务与 Wall-X 发布整理

- PressB 现包含电梯按键、PAP 抓取放盒、桌面红黄蓝按键三个独立任务。电梯的既有配置和 RL 实验结果保留；PAP 对齐真实数据集，相机和物体尺寸仍是图像拟合估计；桌面按键的腕部相机光心、内外参对齐电梯，PAP 保持自己的标定。
- PAP 已接上给定 Wall-X 模型，原生 WebSocket 传输两路 RGB、语言及 7D 关节/夹爪状态，执行 32 步动作块。可选关节 EMA 默认关闭，评估显式使用 α=0.8。PAP 和桌面按键尚未进行在线 RL 训练；后者目前是语言颜色解析＋脚本 IK 验证。
- 四个绿色标签代表矩形的四条边，不是四个顶角。修正后的外框约 27.02×22.31 cm，3 cm 方块中心范围约 24.02×19.31 cm。重新完成两组各 100 次，固定红色 **65/100**、随机五色 **54/100**；全部 200 条轨迹复核通过。旧菱形范围的 88/100、85/100 只保留为历史记录。可发布结果、失败位置图及 CSV 见 [PAP 评估包](results/pap_rectangle_20261023/README.md)，原始证据在 `outputs/wallx_pap_eval_rectangle_20261023`。
- Wall-X 独立克隆位于 `ref/wall-x`，从 PressB 忽略，`base` 已推送 `d7a8748f8bcc988d3ea38c9e68af5578e47bd1ea`；依赖版本写入 `repos.lock.json`。它的原生接口保持兼容，新增可选 PressB RPC/auth 扩展及 RTX 5090 部署脚本。VLA-JEPA 的冻结推理可使用该扩展传输，但原生 Wall-X 本身未实现 VLA-JEPA 的 split encode/decode RL 能力。Isaac 环境保持 WebSocket 12，扩展服务端在独立环境使用 14–15。
- 当前相关完整 CPU 测试和补充 WebSocket 集成测试均通过；固定 PAP 实验预设增加了几何、频率及初始化配置校验，默认生成的计划与已评估文件完全一致。离线结果绘图、位置分析、失败阶段分析和轨迹核查已整理为 `scripts/*pap_eval*.py`。
- README 默认英文并保留中文入口，增加三任务运行说明、结果与 PAP 演示。数据、权重、环境和完整录像仍不入 Git；仅发布精选视频与小型统计。real2sim2real 和真机 RL 验证仍未开展。

## 2026-10-09 仓库发布整理

- 主 README 已汇总三节点框架、8 个配置的完整评估、实际训练耗时、XYZ 平滑结论及 3 段保持 1× 时间的精选视频。小型统计在 `docs/results/`，媒体在 `docs/media/`，原始日志／轨迹／checkpoint 仍留在忽略的 `outputs/` 中。
- 本轮将此前未提交的 5090 适配、在线 RL、训练评估与审计代码统一纳入 PressB；下文历史“工作区未提交”提示是当时的状态，当前以 `git status` 为准。CPU 验证 835 项通过；另用隔离 USD 环境补跑 35 项，合计 870 项通过。
- VLA-JEPA 的 `base` 文档提交为 `2cd35ecbc5a5ce6c4074e3ca7e35a0bb6c656b1b`，已推送；**本机推理 checkout 有意保持 clean detached `9c9dc71`**，因为现有 RL 权重绑定该 commit、路径和源码哈希。发布前后完整推理身份逐值相同，H200 也保持原干净版本。不要把 detached 状态当作需要修复的异常，也不要直接切换最新 `base` 运行旧权重。
- ZPRL 参考仓库的新远端是 `https://github.com/pgq18/ZPRL-PressB.git`，已推送 `main` 文档提交 `4bd52f695bad813406a2327cdb2e448565c8abe9`；原算法来源仍为 `a34a1cb`，实际 Isaac RL 适配位于 PressB，不依赖运行 ZPRL。
- 两个独立克隆均从父仓库忽略，版本见 `repos.lock.json`。`/outputs` 的忽略规则同时覆盖目录和本机的数据盘符号链接，避免提交本机绝对路径链接。

## 2026-10-09 XYZ 动作后处理平滑实验

- 已冻结原 XYZ scale=0.02 的 400k checkpoint、噪声策略和 VLA，在 64 环境中完成 **1,320 次**评估，尝试位置 EMA 和位置＋姿态 SLERP 平滑；没有训练或修改模型权重。新增入口 `scripts/eval_smoothed_residual.py`，实现及复现见 [residual_on_frozen_noise.md](residual_on_frozen_noise.md)。
- **没有候选通过全部预设成功率门槛，默认保持 `none`，不自动启用后处理。** 门槛为每组至少 119/120、不低于同期基线，并用预先声明的新种子确认。原始 XYZ 自动评估的 119/120 仍是单次结果；本次无新增平滑基线跨种子为 116–119/120。
- XYZ alpha=0.8：筛选 119/120（同期基线 117），独立种子 117/120（基线 116）；实际共同成功轨迹的关节 jerk 中位数下降约 14%，末端位置 jerk 下降约 30%，时间变化约 +0.7%～+2.7%。平滑改善可复现，但未稳定达到绝对 119/120 门槛，仅保留为实验选项。
- 位置＋姿态 alpha=0.8：筛选 120/120，独立确认 117/120 < 基线 118，拒绝采用；更轻 XYZ alpha=0.9 为 118/120 < 基线 119，也拒绝。alpha=0.6 两种模式均为 117/120，未进入确认。保留所有失败记录，不将筛选最佳成绩当作独立验证结果。
- 证据根目录 `outputs/online_rl_xyz_smoothing_20261009`；初筛为根目录，姿态确认 `confirm/`，XYZ 确认 `confirm_xyz/`，轻度 XYZ 筛选 `conservative/`。各阶段 `analysis_screen/` 或 `analysis_confirm/` 含成功率、失败明细、实际运动指标、滤波重算和冻结审计；源码快照位于对应 `source_archive/`。
- 汇总为根目录 `smoothing_study_report.json`、`all_phase_metrics.csv`、`all_phase_per_button.csv`。**实验性对比视频**为 `videos_exploratory_xyz_a08/videos/xyz_vs_smoothed.mp4`：左无新增平滑、右 XYZ alpha=0.8，42.57 秒、30 fps、原速 1×；来自 seed=20261010 两组实际 120 次评估的首轮中心任务，两侧视频均 12/12，不代表完整评估通过。轨迹、渲染、合成 QA 全通过，GPU 服务已清理。
- `confirm_xyz/plan.json` 的旧描述字段 `confirmation_seed` 残留 20261009，实际 `plan.seed`、两个运行 manifest 及采样均为 **20261010**，说明见 `metadata_clarification.json`；后续脚本已修复该元数据字段。不同方法异步重置导致部分后续样本初态有微小数值差异；初态完全相同子集也复现了平滑改善，不宣称全部 120 对逐位相同。

## 2026-10-08 XYZ 残差 400k 续接入口

- **2026-10-09 结果更新**：本轮训练和自动评估于北京时间 06:14 完成，400,000 条训练 transition、398,000 次更新；120 条固定条件 eval 为 **119/120（99.17%）**。24 键 9/10，其余 25–35 各 10/10；唯一失败为偏移 `(−0.01,−0.025)m` 第 2 次重复的 24→25 误按。已逐条核对仿真原始终止与 reset 记录，统计及审计见运行目录 `eval/per_button_success.csv`、`eval/failed_episodes.csv`、`eval/audit_20261009.json`。训练服务已自动清理。以下启动说明保留用于复现。
- **执行视频已完成**：`outputs/online_rl_residual_on_noise_gamma0995_xyz002_videos_20261009/videos/pose9_vs_xyz.mp4`，42 秒，左侧原 scale 的 9D 残差、右侧本次 XYZ 残差，均冻结同一噪声策略。1× 仿真时间同步，包含全局和腕部视角。另录的 12 个中心布局任务为左 12/12、右 11/12（24→25），与原 120 次固定评估不同；轨迹、渲染、合成三阶段 QA 全通过，录像服务已清理。
- 用户要求冻结原来训练完成的初始噪声策略，仅对每步 XYZ 学习动作残差，scale 为 `[0.02,0.02,0.02]`；沿用 γ=0.995，从头训练 400k，不恢复旧残差。
- 本轮于 2026-10-08 22:45（北京时间）启动，目录为 `outputs/online_rl_residual_on_noise_gamma0995_xyz002_400k_20261008`；实时状态读取该目录 `queue_status.json`、`train/status.json`，不要把本段启动记录当作最新状态。
- `outputs/online_rl_current.json` 已指向本轮。GPU0 训练，GPU1 运行 64 环境仿真和 batch64 冻结 VLA 推理。监督程序自动完成 400k 后的 120 条固定条件评估并清理自建服务。
- [W&B 实时日志](https://wandb.ai/penggq2025-southern-university-of-science-technology/pressb-online-rl/runs/d154fb90) 包含 recent SR、每按键近期成功率、损失及采样速度。训练 SR 与固定条件评估分开。
- 核心新增 `LearnerConfig.residual_mode="xyz"`：actor/replay 真正为 21 维（7×3），actor 仍以完整 63 维基础动作作为条件，critic 使用完整合成 pose9；旋转不加残差，执行时四元数和夹爪沿用解码器原值。默认目标熵为 −10.5；旧 checkpoint 缺省 `pose9` 保持兼容。
- 初始噪声仍为 `outputs/online_rl_fast_20261003/initial_noise_train/last.pt`，noise_scale=1.5；VLA 仍为冻结 step10600。初始化和冻结检查见本轮 `initial.pt`、`composition.json`（均在 `train/`）及 `startup_verification.json`；配置、源码快照、运行指令均留存在运行目录。
- 本轮启动前 338 项相关 CPU 测试通过。详细方法与命令见 [residual_on_frozen_noise.md](residual_on_frozen_noise.md)。历史 γ=0.995 原 scale 的 120 次 eval 为 118/120；半 scale 为 87/120，不能将中心布局演示视频 12/12、9/12 当作完整 eval。

## 2026-10-03 在线 RL 续接入口

本节覆盖下文 2026-10-02 原始交接中的“无运行任务/仅 H200 推理”状态；下文保留原评估和部署历史。

- 用户已授权优化并运行长程在线 RL，动作残差与初始噪声两种方法各 1M transition。三个 HTTP 节点保留跨机器能力，目前部署改为本机：GPU0 SAC，GPU1 Isaac 并行仿真与冻结 VLA-JEPA。
- 当前运行入口始终读取 `outputs/online_rl_current.json`，再读所指 `queue_status.json`；不要依据本文静态 PID 或计数启动重复服务。完整设计与命令见 [online_rl_fast.md](online_rl_fast.md)。
- VLA-JEPA 已在 H200 创建提交 `9c9dc71c8e199c7428b25d103ab666395832c001`，H200 DNS 无法推送，由本机中继成功 push。独立本地克隆为 `src/VLA-JEPA`，独立推理环境 `.conda/envs/vlajepa-inference`；原 Isaac 环境保留。冻结 step10600 权重位于 `.cache/vlajepa/`，SHA 验证和跨 GPU 数值对照见 `outputs/online_rl_local_inference_20261002/`。
- 已实现 64 环境独立结束/重置、批量受限 IK、PhysX 视图、chunk 边界渲染、批量视觉/flow 推理、无损 packed 特征、SAC 与 RPC 等待重叠。一个 transition 仍是一个环境的最多 7 动作 chunk，不能称 episode/s。warmup 后 UTD=1。
- GPU 物理缓冲溢出曾导致不可信速度，已显式配置容量并加入同步原生日志错误检查；真实溢出负例退出 1，正常配置 62/62 专家接触、错按和无接触负例通过。证据见 `outputs/online_rl_fast_probe_20261003/`。
- `outputs/online_rl_fast_20261002` 候选因原生 224 正方形渲染改变垂直视野而停止，120 条 base 仅 2 成功；此候选不能当作任务有效加速。修正版 `outputs/online_rl_fast_20261003` 保持 320×240 的 4:3 相机投影，再缩放为模型的 224×224。运行状态和最新基线以该目录实测为准。
- 旧 12 环境训练目录 `outputs/online_rl_long_20261002_parallel` 在 10207 transition/8001 update 停止并保存。新 collector 仅迁移 actor，critic、replay 和计数重新初始化；不把两套仿真约定的 replay 混合。
- PressB 工作区仍含此前未提交改动，本轮只按用户要求提交和推送 VLA-JEPA 仓库。不要清理工作区或删除历史证据。

更新日期：**2026-10-02（Asia/Shanghai）**。本文面向直接运行在 **5090 工作站**上的新 agent。用户要求在该机器继续接手机械臂按电梯按钮的仿真、数据采集、训练与评估工作。

**5090 项目根目录：`/home/pengguanqi/Workspace/Research/PressB`。** 除明确写出机器名和绝对路径外，本文中的相对路径均以此目录为根。这是交接时的事实快照；GPU、进程、SSH socket 和 Git 工作区状态可能随后变化。

## 1. 当前结论与接手入口

1. 5090 的仿真环境已经部署并完成实际运行验收，支持黑色 PiPER、官方腕部 D435、桌面 D435、真实按钮接触及灯光反馈。
2. 两张 5090 均验证可用。完整专家轨迹演示已成功按下全部 12 个按钮。不要把专家规划演示的 12/12 与模型自主闭环成功率混为一谈。
3. 带面板位置偏移的 1,200 条训练数据已经在原采集机完成，用户已传到 H200。5090 目前只有两批小规模采集预检数据，**没有在 5090 重新采集完整 1,200 条，也没有安装独立 LeRobot 导出环境**。
4. H200 上 VLA-JEPA 已适配 LeRobot v3 和 PiPER 位姿接口。带偏移数据从官方预训练重新微调的运行已完成 **3 epoch / 12,942 step**。当前 eval 选用该运行中验证最佳的 **step_010600**；不是最终 step_012942，也不是旧固定面板权重。
5. 已完成 **5090 仿真 → SSH → H200 推理**的 60 条闭环评估，以及与原采集机的同条件对照。5090 成功 **20/60**，原采集机历史基线 **5/60**；两套仿真结果不完全一致，详细结论见第 6 节。
6. 交接核查时，5090 无 GPU 计算任务；H200 四卡显存均为 0 MiB。本任务四个临时推理服务已停止，SSH 主连接仍可复用。没有等待本 agent 继续盯守的训练或采集任务。
7. **5090 工作区有重要未提交改动。** GitHub 的旧提交不包含本机全部 5090 适配和跨机器 eval 改动。先看 `git status`，保留工作区，见第 8 节。

新 agent 优先阅读：

- 本文；`docs/rtx5090_setup.md`：5090 安装、运行与验收。
- `docs/policy_eval.md`：推理接口、动作语义、闭环执行与审计。
- `outputs/rtx5090_eval_comparison/results/verification.json`、`outputs/rtx5090_eval_comparison/results/summary.md`：本次对比结果。
- `docs/dataset_panel_stratified.md`：带偏移数据集及验收边界。
- `logs/handover_20261002/`：本次交接附带的机器状态、训练证据和历史基线摘要。

## 2. 用户已确定的场景与数据要求

### 2.1 场景

- 黑色 PiPER 放在桌沿，面向墙壁上的电梯面板。
- 面板 2 列 × 6 行：左列由下到上 24–29，右列由下到上 30–35。
- 初始姿态为大臂、小臂折叠于桌面上方，与桌面平行、垂直于桌沿。每次演示/采集结束后归位，再开始下一条。
- 夹爪默认保持夹住按压杆的姿态；配置为两个关节 `+0.004/-0.004 m`，总开度 **0.008 m**。这是持杆状态，不应改成几何零开度。
- 按下时仅按钮**边缘**亮橙色，数字保持清楚；释放回弹后熄灭，不锁存灯光。
- 腕部 D435 使用官方开源适配资产中的原支架和相机几何，不重新设计替代支架。
- 另一台 D435 在机械臂左侧桌面上的矮支架，靠近机械臂。初始视角完整覆盖面板；机械臂不必全部入镜，操作时允许实际遮挡。
- 已使用官方/现有 Isaac 素材搭建桌子等物体；固定来源清单和 SHA 校验见 `assets/sources.lock.json`、`THIRD_PARTY.md`、`scripts/fetch_assets.py`。

### 2.2 面板偏移

基于名义面板位置，**X 表示前后方向，Y 表示左右方向**；高度不随机。

| 方向 | 已使用范围 |
|---|---:|
| 前后 X | −0.010 ～ +0.010 m |
| 左右 Y | −0.025 ～ +0.025 m |

采集配置为 `configs/dataset_panel_stratified_1200.json`：每层 10×10 分层格，各格一条，四个角格取精确角点，其余格内小幅随机采样；每层独立打乱，seed **20260930**。轨迹和时序也有小幅变化。每个候选位置均检查 12 个按钮可达性、关节/碰撞约束和桌面相机面板投影。

### 2.3 数据接口，不可随意改动

- 12 个 task 文本严格为 `Press 24 floor.` … `Press 35 floor.`，每层 100 条，共 1,200 条。
- 最新数据为 **30 Hz**；物理仿真 **120 Hz**；相机每路 **640×480 RGB**。采集器也支持整除 120 的频率，如 10/20/30/60 Hz，但每种频率须单独配置并保存在独立数据目录。
- 训练 episode 从折叠初始状态到目标按键首次按亮的采样帧，**包含成功画面，不包含撤回和归位**。完整原始诊断轨迹包含撤回，导出器裁剪成训练前缀。
- LeRobot 中 state/action 为 8 维 `[x,y,z,qw,qx,qy,qz,gripper_width]`。坐标系是机械臂 **base_link**，位置单位米。state 为当前实测位姿；action 为下一采样时刻的**绝对目标**，不是位姿增量。终止帧使用既有前缀导出的末端 action 规则，不能补撤回动作。
- 学习的 TCP 是 `link6` 局部 Z=**0.1358 m** 的夹爪 TCP；按压杆尖端是局部 Z=**0.24 m**。二者不能混用。
- 模型训练/推理内部为 9 维 `XYZ + rotation6D`：旋转矩阵**前两行**按行展开 `[r00,r01,r02,r10,r11,r12]`。XYZ 保留原值，不归一化、不改坐标系；夹爪不学习。
- 模型双相机顺序为 **global,wrist**，服务将原始 RGB 双线性缩放至 224×224。每次预测 7 个绝对目标，diffusion inference timesteps=4。

## 3. 三台机器、代码与环境

| 机器 | 用途 | 关键位置 |
|---|---|---|
| 5090 工作站 | 当前主工作区、仿真、双相机渲染与闭环控制 | `/home/pengguanqi/Workspace/Research/PressB` |
| H200，5090 上可直接 `ssh h200` | 训练、模型权重、完整训练数据、远程推理 | `/home/pengguanqi/Worksapce/Research/VLA-JEPA` |
| 原采集机 | 历史 Sim 4.5 数据和本机 eval 基线 | `/data/pgq/Workspace/Research/PressB` |

**H200 路径中的 `Worksapce` 是实际拼写，不能自动纠正成 `Workspace`。** 新 agent 在 5090 上，不要把原采集机 `/data/pgq/...` 当成本地可访问目录。

5090 仓库远程：`https://github.com/pgq18/PressB.git`，交接时 HEAD 为 `5fac67e`。H200 仓库远程：`https://github.com/pgq18/VLA-JEPA.git`，HEAD 为 `d90dec8`，交接核查时工作区干净。H200 的 PiPER 适配已经 commit/push；5090 这轮适配尚未 commit/push。

### 3.1 5090 软件栈

Ubuntu 24.04.4 LTS，驱动 580.173.02，两张 RTX 5090，每卡约 32 GB。项目 Python 直接使用：

```bash
cd /home/pengguanqi/Workspace/Research/PressB
.conda/envs/pressb/bin/python --version
```

| 组件 | 已验收版本 |
|---|---|
| Python | 3.11.16 |
| Isaac Sim | 5.0.0，包元数据为 5.0.0.0 |
| Isaac Lab | Git tag v2.2.0，SHA `46dff135f44683f031edf346e544fcfd8456b2bb` |
| Isaac Lab 核心 Python 包 | 0.44.9；与上述源码发布标签并不矛盾 |
| PyTorch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 |
| NumPy / SciPy / Pillow | 1.26.0 / 1.15.3 / 11.2.1 |
| Warp / PyAV / Gymnasium | 1.7.1 / 15.1.0 / 1.2.0 |

PressB 以 editable 安装。完整依赖见 `logs/install-5090-requirements.txt`，`pip check` 已通过。Conda 程序为 `/home/pengguanqi/miniconda3/condabin/conda`。

- 需要重建时使用 `scripts/setup_env_5090.sh` 和 `configs/constraints-5090.txt`。
- **不要对现有环境运行旧 `scripts/setup_env.sh`**：它固定 Python 3.10 / Sim 4.5 / Lab 2.0.2。
- `.cache/usd-inspect` 是隔离的 usd-core 24.11，用于素材处理和独立 USD 审计。只给这些工具单独设置 `PYTHONPATH`，不要全局导出或覆盖 Kit 内置 USD。正式运行报告的 Kit USD `[0,24,5]` 是正常情况。
- `.conda/envs/lerobot` 尚未在 5090 安装。后续需要导出时使用 `scripts/setup_dataset_env.sh` 建独立环境，不把 `requirements-dataset.txt` 装进仿真环境。
- 5090 的有效设备编号是 **0/1**。旧入口有默认 GPU=4，`collect_parallel.py` 默认多卡列表也不适合本机；所有启动命令显式指定 GPU。

### 3.2 H200 环境与代码入口

H200 有四张 H200 NVL，每卡约 140 GiB 显存。Python 为 `/home/pengguanqi/miniconda3/envs/vlajepa-piper/bin/python`；此前实际推理环境为 Torch 2.6.0+cu124。

| 功能 | H200 仓库内路径 |
|---|---|
| LeRobot/PiPER 加载与数据窗口 | `starVLA/dataloader/piper_lerobot.py` |
| PiPER 训练器 | `starVLA/training/train_piper.py` |
| 训练入口/默认配置 | `scripts/run_piper_ft.sh`、`scripts/configs/vlajepa_piper_ft.yaml` |
| 离线 checkpoint 评估 | `scripts/eval_piper_checkpoint.py` |
| 推理服务/模型包装 | `scripts/serve_piper_policy.py`、`starVLA/inference/piper_policy.py` |
| 训练、续训及接口文档 | `docs/piper_lerobot.md`、`docs/piper_continuation.md`、`docs/piper_status.md` |

SSH 首次连接可能很慢，用户明确要求 **ConnectTimeout=120** 并尽量复用连接。5090→H200 现有 socket 是 `/tmp/pressb-5090-h200-eval.sock`。不要在 5090 上使用原采集机→5090 的 `/tmp/pressb-5090-setup.sock`。

## 4. 数据集与训练已完成的范围

### 4.1 当前主数据集

H200：`/home/pengguanqi/Datasets/piper_elevator_lerobot_panel_stratified_press_30hz`。某些 home 路径实际指向 scratch，保留已有链接和布局。

- LeRobot v3.0；官方 LeRobot 0.6.1 加载器已验证。
- 1,200 episodes、12 tasks、306,830 帧，双路共 613,660 张图像，30 Hz。
- 原采集机最终整包大小记录为 7,704,437,651 字节，约 7.70 GB。
- 完整原始轨迹为 648,454 帧；首次亮灯裁剪移除后续 341,624 帧。
- 每层 100 格各一条，四个角点齐全；物理、位置覆盖、整包解码、训练前缀及官方读取通过。
- 相机视锥覆盖不等于操作中完全无遮挡。腕部 1,200 条末帧均提供清晰亮灯证据；桌面有 109 条终点遮挡/像素不足，由腕部与物理记录确认。
- 完整原始 episode 776 的桌面橙色阈值诊断有一帧迟滞，原失败报告仍保留；腕部正确、几何同步正确，训练前缀没有延长或改阈值。详见 `docs/dataset_panel_stratified.md`，不要将其描述成所有全局视角严格首亮检查都通过。

历史固定面板数据还在 H200：`/home/pengguanqi/Datasets/piper_elevator_lerobot_press_30hz`，同为 1,200 条，但为 306,230 帧。两者不要混用。

5090 `datasets/` 中当前只有 `5090_smoke_30hz/` 和 `5090_smoke_30hz_v2/`。第一批保留已定位的灯光残影，第二批才是验收通过版本。eval 输入目录中的 `dataset_metadata/` 只是原训练集元信息副本，不是完整训练数据集。

### 4.2 带偏移数据微调

H200 运行目录：

```text
/data/scratch/pengguanqi/VLA-JEPA-runs/piper_panel_stratified_press30hz_pose9_3epochs_20260929
```

从官方预训练重新开始，`initialization.json` 为 `step=0, continuation=false`。官方文件通过 `/home/pengguanqi/Models/VLA-JEPA/Pretrain/checkpoints/VLA-JEPA-pretrain.pt` 访问，实际文件在 scratch；已记录 SHA `fd929c79d9bbd0bda56c0b952c7acb470d93c6241a519013fe5248c3f3ea5fab`。Qwen 与 V-JEPA2 基础资源在 `/home/pengguanqi/Models/Qwen3-VL-2B-Instruct`、`/home/pengguanqi/Models/vjepa2-vitl-fpc64-256`，可离线加载。

- 按完整 episode 分层划分，seed=42，每层 90 训练/10 验证：**1,080 训练、120 验证**，无重叠，总体覆盖 1,200 条。
- 训练帧/锚点 276,063；验证帧 30,767。不能说 1,200 条全部参与梯度更新。
- 4 卡，每卡 batch=16，全局 batch=64，梯度累积 1，sample_stride=1；每 epoch 4,314 步，分布式每 epoch 补齐 1 个重复样本。
- 完成 **3 epoch / 12,942 步**，`completed.json` 为 `success=true, epoch=3, batch_in_epoch=0`。
- 北京时间 2026-09-30 04:59:01–10:14:03，监督进程总耗时约 **5 小时 15 分 2 秒**，正常退出；训练内部计时为 18,766.898 秒，两种口径不同。
- 每 200 步验证；每条验证 episode 固定 5 个锚点，共 600 个。最佳按首动作位置误差（米）+0.1×旋转误差（弧度）选择。

| 检查点 | 定位 | 离线首动作位置误差 | 离线首动作旋转误差 |
|---|---|---:|---:|
| `checkpoints/step_010600` | 3 epoch 运行中验证最佳，当前闭环使用 | 6.135 mm | 0.330° |
| `checkpoints/step_012942` | 最后一步，完整 3 epoch | 7.779 mm | 0.305° |

模型 `model.pt` SHA256：

```text
step_010600: 2456b1fff5ef2d94a173b502d55b244a6b0810b92c6e39fc6b1527e1d6019312
step_012942: ec4d9a889ff4123e456342efd72c7ee38317e99e4f590e73865d5df08f56621d
```

最佳、最终的 model.pt 和 training.pt 共四文件已做远端 SHA 复核。`checkpoints/best` 和 `last` 分别解析到上述目录。离线误差不是按键成功率；现有跨机器 60 条闭环结果属于 **10600**，不能归到 12942。

旧固定面板运行 `piper_press30hz_pose9_20260929` 的 step2000、`piper_press30hz_pose9_3epochs_from2000_20260929` 的 step12800 是另一条历史链路。当前 `eval_policy.py` 默认权重检查仍指向旧 step2000，必须显式传入本轮 step 与 SHA。

## 5. 5090 上已经验收的仿真

- 35/35 下载资产 SHA 通过，官方腕部支架/D435 来源审计通过。
- 两卡实际 PyTorch CUDA、矩阵乘法及 Warp kernel 通过。
- `outputs/edge_feedback/`：GPU 0 完整专家演示，23,206 个物理步，12 按钮全部按下、回弹、归位，无意外碰撞；两相机各 1,934 帧有效 RGB-D。该演示默认拍照 10 Hz，与数据采集和 eval 的 30 Hz 区分。
- `datasets/5090_smoke_30hz_v2/`：GPU 1 两个并行环境、两条 30 Hz 原始轨迹，517/512 帧；几何同步零帧延迟，首亮与熄灭同步，灭灯后首帧橙色像素为 0。
- Isaac Lab AppLauncher / SimulationContext 在 GPU 1 reset、步进 10 次、正常退出。
- 部署验收时全套测试 **414 passed**；跨机器 eval 改动后，5090 新增资产重定位/结果对比的针对性测试 **36 passed**。不要把这两次测试说成当前全套 450 项已统一重跑。

主证据：`logs/setup-5090-summary.json`、`logs/verification-5090-pytest-final.log`、`outputs/edge_feedback/audit_summary.json`、`outputs/5090_smoke_feedback_v2.json`、`outputs/5090_smoke_camera_sync_v2/report.json`。

运行视频：`outputs/edge_feedback/episode.mp4`；两路为 `wrist_camera/rgb.mp4`、`global_camera/rgb.mp4`。GUI 要在工作站桌面会话启动，普通 SSH 无桌面显示。

## 6. 最新闭环 eval：配置、结论与证据

### 6.1 两边用的是什么

两边都使用第 4 节**带偏移训练**的同一 step10600，权重 SHA 完全一致；都使用该数据集冻结的场景、配置、任务与种子。

eval 模式是 **`center_corners`**：中心 `(0,0)` 与四角 `(±0.01,±0.025) m`，每个位置测全部 12 层，每条件一次，即中心 12 条、偏移 48 条。为便于一一比较，位置固定成五点，并非每次重新随机抽取。两边 60/60 均有 `measured_layout_verified=true`，实际移动过面板。

| worker | 楼层 | 5090 仿真 GPU | H200 推理 GPU / loopback 端口 |
|---|---|---:|---|
| 0 | 24,25,26 | 0 | 0 / 19765 |
| 1 | 27,28,29 | 0 | 1 / 19766 |
| 2 | 30,31,32 | 1 | 2 / 19767 |
| 3 | 33,34,35 | 1 | 3 / 19768 |

每 worker 三个环境、15 条；`seed=20260930`、`max_seconds=15`、`smoothing_window=3`、动作 30 Hz、物理 120 Hz。每请求 7 步动作，在四个物理子步上线性插值，再做三点因果关节均值滤波；历史跨 chunk 保留，每 episode 重置，名义延迟 8.33 ms。`window=1` 才是关闭均值滤波。

模型只接收实时双相机、实测基座 TCP 和 task 文本；没有输入目标按钮坐标、面板偏移或录制专家动作。IK 是受关节位置/速度约束的连续单初值全位姿求解，原模型目标、可执行目标和跟踪残差分别保留。

成功要求目标按钮位移至少 **1.5 mm**、按压杆接触力 **>0.02 N**，无误按、无异常碰撞。首次按钮按下或异常/超时即终止，不执行撤回。首次按错是失败。运行异常不是模型超时。

等待 H200 响应时仿真时钟暂停；它测闭环策略能力，**不是 30 Hz 实时墙钟控制性能**。

### 6.2 结果

| 结果 | 原采集机 / Sim 4.5 | 5090 / Sim 5.0 |
|---|---:|---:|
| 成功 | 5/60（8.3%） | 20/60（33.3%） |
| 误按 | 27 | 22 |
| 超时 | 28 | 18 |
| 异常碰撞 | 0 | 0 |
| 无效策略动作 | 0 | 0 |

34/60 条终止原因及实际按下楼层一致；原 5 条成功全部保留，新增 15 条成功。两边完整运行通过独立审计。

四组历史无损图像、状态和 seed 经 5090→H200 重放，pose9 与 pose8 动作逐值完全一致，最大差为 0。实时闭环的初始 TCP 差异极小（最大位置约 2.84×10⁻⁹ m），但全局/腕部图像平均像素 MAE 为 3.43/5.79（0–255），首个预测 chunk 的 XYZ RMSE 平均约 1.22 mm。

**结论边界**：远程推理链路可用，四组相同输入的输出一致；跨版本完整闭环并未逐条复现。本次 5090 成功更多，不能据此称“5090 硬件提升了模型能力”。Sim 版本、渲染硬件、灯光稳定刷新次数等同时不同，目前没有消融实验单独区分渲染与物理影响；60 个条件各测一次也不是泛化成功率的充分统计。

### 6.3 5090 上的现成证据

以 `outputs/rtx5090_eval_comparison/` 为根：

| 路径 | 内容 |
|---|---|
| `input/scene.usda`、`input/config.json`、`input/dataset_metadata/meta/collection_metadata.json` | 原始冻结输入 |
| `input/assets/` | 完整且已验证的资产包 |
| `input/golden_requests/worker_0..3/` | 四组相同输入复测材料 |
| `center_corners_v1/worker_0..3/` | 正式 60 条完整 eval，含所有 chunk 无损观测、双视频、动作及物理记录 |
| `center_corners_v1/aggregate_report.json`、`center_corners_v1/summary.md`、`center_corners_v1/episodes.csv` | 四 worker 汇总 |
| `results/comparison.json`、`results/summary.md`、`results/verification.json` | 与原采集机的配对比较 |
| `golden_replay.json` | 固定输入逐值对照 |
| `services.json`、`formal_processes.json`、`service_cleanup.json` | 当时的服务/启动/清理证据，PID 不代表现在仍有效 |
| `results/compare_floor33_center.mp4` | 33 层中心任务四宫格，左旧机误按 34、右 5090 按中 33 |
| `results/compare_floor33_center_video.json` | 来源哈希及视频制作说明 |
| `pilot/` | 前置 1 秒链路检查，其超时不计入正式 60 条 |

每 worker 的 `audit.json` 为独立审计，`diagnosis.json` 为模型/IK/跟踪分解，`eval_manifest.json` 为配置与源码身份。`audit_pass` 只代表证据有效，不表示任务成功。MP4 使用 30 Hz CFR，精确终止物理时刻看 `frames.npz`，不能只按视频时长推断。

对照视频右列在 6.517 秒成功终止后定格，左列在 10.742 秒误按；已明确标注定格，不能解释为右侧继续执行。

原采集机的完整历史 baseline 不在 5090；它仍位于 `/data/pgq/Workspace/Research/PressB/outputs/policy_eval_step10600_stratified/center_corners_v1`。本次交接复制了摘要到 `logs/handover_20261002/baseline_evidence/`，配对比较 JSON 也已在 5090，若需重算所有原图指标再取回原始历史记录。对比 JSON 中一些输入路径属于原采集机，不能直接作为 5090 本地路径。

原采集机取回的 `eval_evidence.tar` 只含每条首个 chunk PNG，加全量视频/轨迹；**全部中间观测 PNG 在 5090 原目录**。独立审计最好在 5090 原位置运行，重定位报告引用绝对路径；不要删除工作站上的原始观测。

## 7. 必须保留的兼容修复

1. **Dome 全局光**：并行环境只保留一个有效 Dome，额外实例 `SetActive(False)`。仅设置 intensity=0 曾造成采集图像显著变暗，不能回退。局部 RectLight 另行处理，`lighting.json` 和有效 Dome 数量检查保留。
2. **灯光残影**：5090 `scripts/collect_dataset.py` 的 `LIGHT_SETTLE_CAPTURES=16`，原采集机为 4。灯光切换后做不推进物理时间的刷新，已实测解决灭灯余色。保留图像/状态不推进断言与原验收阈值；更改渲染策略会改变采集 fingerprint，不能续写旧目录。
3. **相机字段**：Sim 5.0 的 `rendering_frame` 可能是 Fabric 分子/分母字典；5090 记录器保留字典和类型说明，也兼容旧整数。`rendering_time` 仍为 API 仿真秒数。不要把分子当帧号，也不要把两个字段写反。
4. **冻结场景资产重定位**：原 snapshot 有旧机器绝对路径。传 `--asset-bundle`，由 `src/pressb/scene_portability.py` 生成本次输出的 `runtime_scene.usda`；不得重建一个“看起来相同”的 scene 并伪造旧 SHA。
5. **资产包闭包**：10 文件包括 PiPER 主 USD+三个配置 USD、桌子/支架、纹理、官方腕部兼容 USD 和两张生成标签。七个唯一绝对资产路径被映射到包内；源文件字节以 SHA 校验，运行副本经逆映射验证 authored USD 内容一致，`OmniPBR.mdl` 保留引擎资源名。
6. **审计身份**：原场景 SHA `d5f5556825db36635dd1636561bc0caa75bdd156d93b6e254922239450e9741a`；collection fingerprint `6ab95889499aa7e4735b1d94d90358c18439814217e43108308fa007035f08a4`。`scene_relocation.json` 及其自身 SHA 写入 manifest；审计重新检查资产和逆映射，并比较完整报告。不要改 collection 元信息绕过检查。
7. manifest 中 `renderer_settings` 是代码请求设置，不是运行时逐项读回，不能单凭它证明跨引擎渲染语义相同。

## 8. 未提交改动与代码地图

交接前核对到的 5090 改动如下；本文及 README 交接入口也会成为本次未提交文档改动。以 `git status --short` 当前输出为准。

| 文件/组 | 目的 |
|---|---|
| `README.md`、`pyproject.toml` | 5090 入口；Python 3.10–3.11 支持 |
| `configs/constraints-5090.txt`、`scripts/setup_env_5090.sh` | Sim 5.0/Lab 2.2 依赖组合 |
| `scripts/verify_gpu_5090.py`、`scripts/verify_simulation_5090.sh`、`docs/rtx5090_setup.md` | 硬件/仿真验收与部署文档 |
| `scripts/collect_dataset.py` | 16 次灯光稳定刷新 |
| `src/pressb/camera_recording.py`、`tests/test_camera_recording_metadata.py` | Sim 5.0 相机元数据兼容 |
| `tests/test_dataset_scene.py`、`tests/test_motion_smoothing.py` | USD 双精度偏移及浮点断言适配 |
| `scripts/eval_policy.py`、`scripts/audit_policy_eval.py`、`src/pressb/scene_portability.py`、`tests/test_scene_portability.py` | 冻结资产跨机器重定位、证据与独立审计 |
| `scripts/compare_policy_eval_runs.py`、`tests/test_compare_policy_eval_runs.py`、`docs/policy_eval.md` | 条件配对、跨机器对照与使用说明 |

**不要 `git reset --hard` / `git clean` 清掉工作区；不要从原采集机整目录覆盖。** 原机仍有 Sim 4.5 的相机与 4 次灯光刷新代码。若后续需要提交，先区分兼容性改动与实验产物，针对性检查后提交。数据、权重、环境、素材、输出、日志都在 `.gitignore` 中；本次交接未执行新的 commit/push。

其他主要模块：`src/pressb/scene.py` 搭场景，`dataset_scene.py` 并行场景/灯光/面板布局，`dataset_planning.py` 专家采集规划，`press_prefix.py` 首次亮灯裁剪，`replay_control.py` 位姿 IK，`motion_smoothing.py` 平滑，`policy_layouts.py` 五位置评估调度，`policy_eval.py` 远程接口/动作检查。

## 9. 常用复现命令

以下是操作参考，不是要求新 agent 接手后自动全部重跑。已有结果已经完成。所有新运行使用未占用的新目录，保留原始证据。

### 9.1 接手检查与专家演示

```bash
cd /home/pengguanqi/Workspace/Research/PressB
git status --short
nvidia-smi
.conda/envs/pressb/bin/python scripts/fetch_assets.py --check-only

# 只在需要新演示时执行；此目录必须未使用。
bash scripts/run.sh --headless --gpu 0 --video --output outputs/handover_demo_new
.conda/envs/pressb/bin/python scripts/audit_episode.py outputs/handover_demo_new
.conda/envs/pressb/bin/python scripts/audit_global_camera.py outputs/handover_demo_new
.conda/envs/pressb/bin/python scripts/check_results.py outputs/handover_demo_new
```

完整演示判据为 `PASS: 12/12 physically pressed in order` 及独立审计成功，不能截断几步就当完整验收。GUI 示例见 `docs/rtx5090_setup.md`。

### 9.2 复用 5090→H200 SSH，并启动一个推理服务

在 5090 检查已有主连接：

```bash
ssh -S /tmp/pressb-5090-h200-eval.sock -o ConnectTimeout=120 -O check h200
```

仅当没有活动主连接时，在单独终端建立并保持该终端：

```bash
ssh -M -S /tmp/pressb-5090-h200-eval.sock -o ConnectTimeout=120 \
  -o ControlPersist=no -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N h200
```

另一个 5090 终端登录 H200：

```bash
ssh -S /tmp/pressb-5090-h200-eval.sock -o ConnectTimeout=120 h200
```

以下在 **H200 shell** 执行。先看显存，GPU 编号按当时余量选择。服务前台运行，保留这个终端或按既有监督进程方式管理；不要使用旧 PID 当作新服务 PID。

```bash
cd /home/pengguanqi/Worksapce/Research/VLA-JEPA
nvidia-smi
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/pengguanqi/miniconda3/envs/vlajepa-piper/bin/python \
  -u -m scripts.serve_piper_policy \
  --checkpoint /data/scratch/pengguanqi/VLA-JEPA-runs/piper_panel_stratified_press30hz_pose9_3epochs_20260929/checkpoints/step_010600 \
  --port 19765 --device cuda:0
```

回到 **5090**，为尚未存在的转发增加映射；已有同端口转发不要重复添加：

```bash
ssh -S /tmp/pressb-5090-h200-eval.sock -o ConnectTimeout=120 \
  -O forward -L 127.0.0.1:19765:127.0.0.1:19765 h200
curl --fail http://127.0.0.1:19765/health
```

确认 `ready`、`checkpoint_verified=true`、step=10600 和上述 SHA。模型首次加载此前约 90 秒，HTTP 端口不应公开监听。每个服务此前约占 10.5 GiB 显存。其他三服务用 GPU 1/2/3 和端口 19766/19767/19768。任务结束只停止自己创建且 argv 匹配的服务，不要广泛 `pkill python`。

### 9.3 5090 一组正式闭环（15 条）

```bash
cd /home/pengguanqi/Workspace/Research/PressB
bash scripts/eval_policy.sh \
  --endpoint http://127.0.0.1:19765/predict \
  --dataset outputs/rtx5090_eval_comparison/input/dataset_metadata \
  --config outputs/rtx5090_eval_comparison/input/config.json \
  --snapshot outputs/rtx5090_eval_comparison/input/scene.usda \
  --asset-bundle outputs/rtx5090_eval_comparison/input/assets \
  --expected-checkpoint-step 10600 \
  --expected-checkpoint-sha256 2456b1fff5ef2d94a173b502d55b244a6b0810b92c6e39fc6b1527e1d6019312 \
  --floors 24,25,26 --panel-layouts center_corners --episodes-per-floor 1 \
  --num-envs 3 --gpu 0 --max-seconds 15 --seed 20260930 \
  --smoothing-window 3 --timeout 120 \
  --output outputs/handover_eval_new/worker_0
```

链路短检查可另设新目录、单层、单环境和 `--max-seconds 1`，通常超时是预期，不能算正式成功率。完整复现按第 6 节四组执行；四并发之前看主机内存和显存，上次可成功运行，内存占用较高。

每 worker 完成后，在 **5090 原位置**审计、诊断、出报告：

```bash
PYTHONPATH=.cache/usd-inspect:src .conda/envs/pressb/bin/python \
  scripts/audit_policy_eval.py --run outputs/handover_eval_new/worker_0 \
  --allow-partial --report outputs/handover_eval_new/worker_0/audit.json
.conda/envs/pressb/bin/python scripts/diagnose_policy_eval.py \
  --run outputs/handover_eval_new/worker_0 \
  --output outputs/handover_eval_new/worker_0/diagnosis.json
.conda/envs/pressb/bin/python scripts/summarize_policy_eval.py outputs/handover_eval_new/worker_0
```

`--allow-partial` 允许只选三层，但该 worker 声明的 15 条仍必须完整；不是忽略失败或跳过缺失记录。上述首次报告文件使用新目录，重新审计旧运行时另取报告文件名。

四组均完成后汇总；此汇总器针对每组 15 条的标准 60 条方案：

```bash
.conda/envs/pressb/bin/python scripts/aggregate_policy_eval.py outputs/handover_eval_new \
  --expected-checkpoint-step 10600 \
  --expected-checkpoint-sha256 2456b1fff5ef2d94a173b502d55b244a6b0810b92c6e39fc6b1527e1d6019312
```

若是与当前 **5090** 基线比较，在两次身份/条件相同且完整后：

```bash
.conda/envs/pressb/bin/python scripts/compare_policy_eval_runs.py \
  --baseline-root outputs/rtx5090_eval_comparison/center_corners_v1 \
  --candidate-root outputs/handover_eval_new \
  --output outputs/handover_eval_new/comparison
```

`comparison_valid=true` 表示条件可比较，不代表行为一致。这个脚本会拒绝不同 checkpoint；如果要比较不同权重，应明确标注为模型对比并使用对应分析流程，不要删掉身份检查硬凑跨机一致性。

### 9.4 后续在 5090 新采 1,200 条的入口

当前尚未在本机执行此整批。先按 `docs/rtx5090_setup.md` 做两条预检，通过后才沿用下面配置：

```bash
cd /home/pengguanqi/Workspace/Research/PressB
.conda/envs/pressb/bin/python scripts/collect_parallel.py \
  --gpus 0,1 --num-envs 3 --fps 30 --episodes-per-task 100 --seed 20260930 \
  --config configs/dataset_panel_stratified_1200.json \
  --snapshot outputs/edge_feedback/scene.usda \
  --output datasets/piper_elevator_raw_panel_stratified_5090_new
```

这是新采集批次，使用 5090 当前场景和软件指纹；不要续写原采集机 Sim 4.5 数据。后续按 `docs/dataset_panel_stratified.md`、`docs/dataset_press_only.md` 和 README 完成物理/覆盖/图像同步审计、首次亮灯裁剪、独立 LeRobot 环境导出及官方读取。不能直接把含撤回的 raw 目录当训练集。

如需重新训练，从 H200 既有 `scripts/run_piper_ft.sh`、运行目录 `config.yaml` 与 `docs/piper_lerobot.md` 入手。根据用户后续要求明确官方重启还是 checkpoint 续训，使用新的 run id；不要重跑旧 `launch.py` 或覆盖已完成的目录。新数据也不能沿用旧 dataset/split fingerprint。

## 10. 尚未完成的研究与后续方向

当前用户最新确认的是：两端 eval 都用了带偏移权重和带偏移环境；本次要求完成交接。没有一项已启动但未完成的实验需要恢复。

后续可能工作，应按用户的新要求选择：

1. 以现有 5090 60 条为新平台基线，定位低成功率和误按上一层的问题。检查 `diagnosis.json`，区分模型原始目标、IK 可行投影、平滑与实际关节跟踪；避免直接用目标坐标修正策略而改变评估含义。
2. 若研究跨机差异，分别设计图像固定输入、固定动作物理回放或渲染参数消融，保持模型/条件一致。现有结果尚未完成这种单因素归因。
3. 评估最终 step12942、增加重复次数或更多内部面板位置；这些均没有包含在当前 60 条结果中。改变 checkpoint 要同步服务、step/SHA 与分析口径。
4. 如果需要在 5090 重新采集/微调，先装独立导出环境，再形成一个新的完整验证数据批次，保留版本差异和训练/验证划分。
5. 整理并提交 5090 的兼容性改动，处理与 GitHub 旧 Sim 4.5 入口并存的问题。当前只完成工作区实现与验收，没有替用户做本轮 commit/push。

## 11. 交接时的操作习惯与判断标准

- 用户希望自主完成已授权的具体工作，减少反复确认；SSH 慢时复用连接并设 120 秒连接超时。
- 开始训练、采集或 GPU 推理前读当前占用；不要影响其他用户作业。交接时空闲不代表后续一直空闲。
- 保留已生成数据、冻结场景、模型与报告，使用独立新输出目录；不要改历史元信息、阈值或 SHA 来让审计“通过”。
- 报告分别说明运行是否完成、独立审计是否通过、任务是否成功；这三件事不同。
- 没有证据就不要承诺与旧版本逐帧一致，也不要把计划或短预检说成完整采集/完整测试。
- H200 和 5090 过去出现过系统时钟约 8 小时差异，跨机器耗时优先看单进程单调计时或同机起止记录；用户日期/时区按 Asia/Shanghai。不要把不同主机日志的墙钟直接相减。

本次交接附带的 `logs/handover_20261002/manifest.json` 列出复制证据的来源、字节数和 SHA256；`state_snapshot.json` 记录交接时的 Git、GPU、环境及连接状态。证据是历史记录，新 agent 应先读再按当前实际情况继续。
