# 面板位置分层采集：12 × 100 条

> 文中的数据集、权重、图像和运行报告属于本地产物，不随 Git 仓库分发；新机器请先按 [README](../README.md) 生成场景。历史结果不代表本次安装已经完成验证。

本批使用 `configs/dataset_panel_stratified_1200.json`，在左右 ±25 mm、前后 ±10 mm 的矩形范围内重新采集。24–35 层分别有 100 条独立仿真 episode。每层把 XY 范围划成 10×10 格，每格恰好一条；四个角格使用矩形的精确角点，其余 96 格在格内随机取位置。每层独立打乱格子顺序，采集 seed 为 `20260930`，不受 GPU 数量或并行调度影响。

每个位置在执行前检查全部 12 个按键的可达轨迹、关节范围、桌面和相机支架间隙、完整面板的相机投影。拒绝采样仅在所属格内进行；四个精确角点不允许退让。实际按压继续记录接触力、按钮行程、亮灭反馈和复位检查。机械臂与两台相机的位置固定，面板高度固定。

训练数据保持 30 Hz、双路 640×480 RGB。每条从折叠初始位姿开始，到目标按键边框首次亮起的采样帧为止，包含成功画面。后续撤回和归位只保留在原始诊断记录中。state/action 继续为基座坐标系的 8 维绝对 TCP 位姿 `[x,y,z,qw,qx,qy,qz,gripper_width]`；位置偏移是 episode 元数据，不增加模型输入维度。模型训练时仍可按既有适配转换为 xyz + 6D 旋转并移除固定夹爪维度。

本批已完成，`pipeline_status.json` 为 `complete`、`validation_complete: true`。最终目录为 LeRobot v3.0 格式，已通过 LeRobot 0.6.1 官方加载器验证。目录文件总大小为 7,704,437,651 字节（约 7.70 GB）；详细证据见验收汇总（本地产物：`outputs/panel_stratified_1200/quality_summary.json`）及官方读取报告（本地产物：`datasets/piper_elevator_lerobot_panel_stratified_press_30hz/meta/published_readback.json`）。

| 内容 | 路径 |
| --- | --- |
| 完整原始记录 | `datasets/piper_elevator_raw_panel_stratified_30hz` |
| LeRobot 首次亮灯数据 | `datasets/piper_elevator_lerobot_panel_stratified_press_30hz` |
| 可恢复的转换分片 | `datasets/piper_elevator_parts_panel_stratified_press_30hz` |
| 本批场景、代码和验证证据 | `outputs/panel_stratified_1200/` |
| 批次实时状态 | `outputs/panel_stratified_1200/pipeline_status.json` |
| 完成及质量汇总 | `outputs/panel_stratified_1200/quality_summary.json` |
| 机器可读完成记录 | `datasets/collection_result_panel_stratified_press_30hz.json` |

任务脚本 `outputs/panel_stratified_1200/run_pipeline.py` 依次执行采集、原始物理审计、每层覆盖审计、首次亮灯裁剪、图像反馈检查、LeRobot 导出和官方读取器验证。只有训练数据发布所需检查通过、额外诊断完成复核后，状态才会写为 `complete`；原始诊断失败不会改写成通过。导出器在每个分片及最终合并时验证全部数值、视频解码和首次亮灯终点；官方读取器另检查全部 episode 的首/中/末样本、动作窗口和末尾 padding。

本批原始数据的物理与位置覆盖审计已全部通过。每层实际覆盖 100 格和四个精确角点。四角共 48 条完整动作的双相机运动几何检查均为零帧延迟；腕部全部亮灭对齐，全局 42 条对齐、6 条因遮挡不足以判断。

实际采集共 1,200 条，24–35 层各 100 条，完整原始记录为 648,454 帧。首次亮灯裁剪保留 306,830 帧、对应两路相机共 613,660 张图像，移除后续 341,624 帧。独立覆盖审计确认每层 100 格各一条、四个角点齐全，偏移实际覆盖前后 ±10 mm、左右 ±25 mm。完整面板在桌面相机视锥内的最小画幅余量约 103.587 像素；这不表示操作过程中面板完全不受机械臂遮挡。

最终整包审计已完成全部 613,660 张图像的解码及亮灯终点检查，错误数为 0。官方读取器另外验证了每条 episode 的首、中、末采样点，共 3,600 个状态及动作窗口、7,200 张图像、1,200 次末端补齐检查，错误数为 0。完整解码与官方批量读取的检查范围在报告中分别记录。

全部 1,200 条腕部末帧均提供了清晰的首次亮灯证据。桌面相机有 1,091 条末端反馈可判定，另 109 条由于目标灯边遮挡或可见像素不足，按既有规则由腕部图像确认。下述第 776 条是完整原始动作中唯一的颜色阈值时序诊断，不是唯一存在桌面视角遮挡的记录。

完整原始视频的颜色阈值检查有一项保留的诊断：第 776 条（32 层）桌面视角的大部分灯边被夹爪遮挡。第 251 帧已经出现较淡的黄色细边，但下一帧才达到原有橙色像素阈值。腕部在正确的第 251 帧清楚亮灯；两路完整运动图像的几何检查均与状态对齐。原始 `feedback.json` 的失败结论和退出码保留，未修改颜色阈值或原始数据，也不宣称所有桌面视角都通过严格首亮阈值检查。

该条训练前缀仍保留 0–251 帧，共 252 帧，未延长到下一帧。既有训练数据验收本来就允许桌面目标暂时遮挡，并强制腕部末帧可见及接触成功；单条导出、最终整包和官方读取均已按未修改的规则通过检查。恢复任务脚本 `resume_after_export.py` 根据独立导出的真实退出码和报告恢复读取阶段；已复核的原始全动作诊断作为附加证据保留，发布目录的 `meta/feedback_review.json` 同步记录此限制。详见逐帧诊断（本地产物：`outputs/panel_stratified_1200/feedback_diagnosis_776/README.md`）、诊断拼图（本地产物：`outputs/panel_stratified_1200/feedback_diagnosis_776/feedback_transitions.png`）及验收范围记录（本地产物：`outputs/panel_stratified_1200/feedback_resolution.json`）。

已完成的原始记录、转换分片、最终训练目录及报告均保留。复现或恢复时必须使用相同采集指纹、代码和配置，不应改写已发布的数据文件；此前固定面板数据及小批随机预览位于各自独立目录。

完成数量、文件身份和验证范围以批次状态、质量汇总及其关联的实际审计报告为准。
