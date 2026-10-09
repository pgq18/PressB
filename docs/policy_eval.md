# PiPER VLA-JEPA 仿真闭环评估

> 文中的数据集、权重、图像和运行报告属于本地产物，不随 Git 仓库分发；新建场景见 [README](../README.md)。跨机器复现已有评估时，应转移原冻结场景及完整资产，按下文的跨机器流程运行。历史结果不代表本次安装已经完成验证。

`scripts/eval_policy.py` 在采集数据时的 Isaac Sim 场景中运行已训练策略。每次推理使用当前全局相机、腕部相机图像和当前实测末端位姿；模型返回动作后，机械臂通过物理仿真运动，再重新观察。它与 `scripts/replay_dataset.py` 的录制动作回放是两种不同的测试。

面板位置随机化数据这轮训练的最佳 `step_010600` 已完成 **60 次闭环评估，成功 5/60（8.3%）**：中心位置 1/12，四个边界角点合计 4/48；27 次误按、28 次超时、没有异常碰撞。评估使用训练时冻结的配置与场景，前后 X=±10 mm、左右 Y=±25 mm，12 个楼层各测试中心和四角一次，每次最多 15 秒，30 Hz 动作、120 Hz 物理步进、3 点因果关节均值滤波。四组各负责三个楼层，推理种子按楼层和位置固定，不随分组改变。这是 60 个固定条件的测试，不能视为任意位置上的统计成功率。见完整汇总（本地产物：`outputs/policy_eval_step10600_stratified/center_corners_v1/summary.md`）、逐条结果（本地产物：`outputs/policy_eval_step10600_stratified/center_corners_v1/episodes.csv`）和成功／误按／超时的双相机预览（本地产物：`outputs/policy_eval_step10600_stratified/center_corners_v1/preview_center_examples.mp4`）。原始双相机录像和物理记录全部保留。

使用 `--panel-layouts center_corners` 可从冻结配置的随机化边界生成这五个位置；每个所选楼层在每个位置运行 `--episodes-per-floor` 次。面板只在机械臂回到初始位姿后换位，物理稳定后验证按钮锚点和实际全局相机投影。布局坐标不会进入策略输入或 IK 目标，未调用专家规划器。`--panel-layouts fixed` 保留原固定位置评估行为。

累计 3 epoch 续训中的最佳 `step_012800` 已在原固定面板场景完成评估：**1/12 成功**（33 层），误按 24→25、25→26、32→33，其余 8 层超时，没有异常碰撞。原场景 SHA、配置、初始姿态、任务种子、控制器和上限时长均与第 2000 步评估一致；初始相机图像也已独立对照。12 条物理记录及 24 路视频通过审计，仿真进程退出码为 0。结果、视频及动作诊断分别见汇总（本地产物：`outputs/policy_eval_step12800_fixed/formal_v1/summary/README.md`）、审计（本地产物：`outputs/policy_eval_step12800_fixed/formal_v1/audit.json`）、模型/IK/实测分解（本地产物：`outputs/policy_eval_step12800_fixed/formal_v1/diagnosis.json`）及旧权重对照（本地产物：`outputs/policy_eval_step12800_fixed/formal_v1/comparison_step2000.json`）。这是每层一次的小批评估，不是成功率的充分统计估计。

## 模型输入与动作含义

- 输入相机：全局、腕部两张实时 RGB 图像，640×480；服务端按训练过程转为 RGB、双线性缩放至 224×224，顺序为 `global,wrist`。
- 状态：从 PhysX 实际关节角计算的夹爪 TCP 位姿；坐标系是 `base_link`，单位为米，TCP 是 `link6` 局部 Z 轴方向 0.1358 m。八维状态顺序为 `x,y,z,qw,qx,qy,qz,gripper_width`，夹爪宽度来自实测关节值。
- 模型输入/学习的旋转表示为旋转矩阵前两行展开的 6D。服务端负责状态四元数转 6D、预测 6D 转四元数；XYZ 不作坐标变换或归一化。
- 模型每次输出七个绝对目标位姿。第一个目标对应下一采样时刻，即 1/30 秒后，其余依次对应后续时刻。
- 夹爪固定关闭在 0.008 m 的总开度，握住按压杆；模型不学习夹爪开合。
- 物理按压杆端点是 `link6` 局部 Z=0.24 m，不能把它误当作学习动作的夹爪 TCP。

任务文本严格为 `Press 24 floor.` 到 `Press 35 floor.`。按钮坐标只用于记录评估距离，不传给模型或 IK 控制器。评估代码不会读取训练集的动作、关节轨迹或专家运动计划；训练数据元信息仅用于核对冻结场景和配置。

## 控制与时间

每次推理期间暂停仿真时钟。响应返回后，依次执行完整七步动作，再采集新观察并推理。每步控制周期 1/30 秒，细分为四个 120 Hz 物理步。该测试衡量仿真闭环策略能力，不代表真实时间部署能满足 30 Hz。

控制器复用回放的单初值连续全位姿 IK。学习策略可能给出不可达或单周期移动过快的位姿，因此控制器用关节位置及速度限制下的最小二乘解执行，并记录原始模型目标与实际可执行命令之间的残差。没有替换为专家目标或搜索其他 IK 分支。每个动作保存只施加关节位置限制的诊断求解结果，以及同时施加速度限制的 IK 结果。原始模型的 XYZ、旋转数值保留在请求响应和动作记录中。

现在默认在 120 Hz 关节线性插值之后加 **3 点因果均值滤波**，跨 chunk 保留历史、每 episode 重置，名义延迟约 8.33 ms；`--smoothing-window 1` 可关闭均值滤波，复现原先仅插值的执行方式。原始 IK 端点、平滑前插值命令、实际滤后命令和对应误差分别保存，模型观测使用真实物理状态。上面的 step2000/step12800 历史评估没有均值滤波；实现、配置和验证范围见[平滑说明](policy_smoothing.md)。

所有环境从与采集一致的折叠初始位姿出发，先物理稳定，再开始记录。并行场景只保留一个有效 Dome 环境光。相机使用相同渲染设置，零物理时间渲染，检查拍照和等待推理均不改变实测状态。

## 成功与终止

当按钮真实位移达到 `press_threshold=0.0015 m` 且按压杆接触力大于 0.02 N，边缘橙灯点亮。首次按钮按下即终止该 episode，不要求撤回或回到初始位姿。

- `task_success`：按下目标按钮，且没有按错按钮。
- `success`：上述任务成功，且没有非预期碰撞。
- 失败原因包括超时、按错按钮、非预期碰撞、无效策略输出；服务连接或仿真异常单独作为运行异常报告。
- 默认每层一次、十二层共十二次，每次最多 15 秒仿真时间。单次结果是初步评估；需要估计随机策略成功率时提高 `--episodes-per-floor`。

当终止事件发生在 30 Hz 两帧之间时，立即额外保存该物理时刻的终止帧。MP4 仍以 30 Hz CFR 编码；准确的末帧仿真时刻查看 `frames.npz`，不要用视频时长代替物理时间。

## 启动模型服务与复用 SSH

复用已经建立的连接，避免每次认证等待。以下是当前连接的控制 socket；如果连接已关闭，只在需要时重新建立一次，连接超时设为 120 秒：

```bash
ssh -o ConnectTimeout=120 -o ControlPath=/tmp/pressb-h200-vlajepa-live.sock -O check h200

# 仅在没有活动主连接时运行；该终端保持开启。
ssh -o ConnectTimeout=120 -o ControlMaster=yes \
  -o ControlPath=/tmp/pressb-h200-vlajepa-live.sock \
  -o ControlPersist=12h -N h200
```

通过同一 socket 打开交互 shell：

```bash
ssh -o ConnectTimeout=120 -o ControlPath=/tmp/pressb-h200-vlajepa-live.sock h200
```

在 H200 shell 中运行服务（选择有空闲显存的 GPU；下面用逻辑 GPU 0）：

```bash
cd /path/to/VLA-JEPA
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/path/to/miniconda3/envs/vlajepa-piper/bin/python \
  -m scripts.serve_piper_policy \
  --checkpoint /path/to/VLA-JEPA-runs/piper_press30hz_pose9_20260929/checkpoints/step_002000 \
  --port 18765 --device cuda:0
```

另一个本地终端在已有连接上增加端口转发；已存在该转发时无需重复运行：

```bash
ssh -o ConnectTimeout=120 -o ControlPath=/tmp/pressb-h200-vlajepa-live.sock \
  -O forward -L 127.0.0.1:18765:127.0.0.1:18765 h200
```

服务仅绑定远端 loopback。`GET http://127.0.0.1:18765/health` 返回权重 SHA256、训练步数、动作约定及加载状态。默认评估脚本严格核对 step 2000 权重 SHA256：

`6793045b0b00e863737fe4c0742fde688472d415ab4d8a788ded6254ebe1d1f4`

评估续训检查点时同时指定 `--expected-checkpoint-step` 和 `--expected-checkpoint-sha256`，两者均须与服务实际加载的权重一致。例如累计 3 epoch 续训中验证评分最好的第 12800 步：

```bash
bash scripts/eval_policy.sh \
  --config outputs/edge30_source/config.json \
  --snapshot outputs/edge30_source/scene.usda \
  --expected-checkpoint-step 12800 \
  --expected-checkpoint-sha256 c2502396b093ec45d2b3fa2ac1d602b29ecafaa147a7bfe14768971028834c72 \
  --max-seconds 15 --num-envs 3 --gpu 4 --episodes-per-floor 1 \
  --seed 20260930 --smoothing-window 1 \
  --output outputs/policy_eval_step12800_fixed/formal_v1
```

服务端 `--checkpoint` 对应目录为 `/path/to/VLA-JEPA-runs/piper_press30hz_pose9_3epochs_from2000_20260929/checkpoints/step_012800`。上述配置明确使用训练时冻结的固定居中面板，不启用新的 X/Y 随机偏移；运行会核对配置内容及场景 SHA256 与训练数据一致。输出目录已存在时，重复评估须换用新的目录。

`POST /predict` 的 JSON 请求：

```json
{
  "task": "Press 24 floor.",
  "state": [0.3, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.008],
  "images": {"global": "BASE64_PNG", "wrist": "BASE64_PNG"},
  "seed": 20260930
}
```

示例状态只是接口示意，实际必须发送实测值。响应包含 `actions_pose8`（7×8）、`actions_pose9`（7×9）、推理耗时和坐标约定；评估端会独立验证四元数与 6D 表示一致、XYZ 未变化、夹爪固定。

## 本地仿真与审计

在 `/path/to/PressB` 中，先跑一秒接口检查；输出目录必须不存在：

```bash
bash scripts/eval_policy.sh \
  --floors 24 --max-seconds 1 --num-envs 1 --gpu 2 \
  --output outputs/policy_eval_step2000/smoke_new
```

正式覆盖十二层，每层一次，三个并行环境：

```bash
bash scripts/eval_policy.sh \
  --max-seconds 15 --num-envs 3 --gpu 2 --episodes-per-floor 1 \
  --output outputs/policy_eval_step2000/eval_new
```

独立审计不导入策略、控制器或 Isaac，重新计算 FK、按钮反馈、动作转换和控制时序，并解码视频：

```bash
.conda/envs/lerobot/bin/python scripts/audit_policy_eval.py \
  --run outputs/policy_eval_step2000/eval_new \
  --report outputs/policy_eval_step2000/eval_new/audit.json
```

若本地数据环境目录名称不同，使用安装了 NumPy、Pillow、PyAV 的 Python；单层检查审计增加 `--allow-partial`。`audit_pass=true` 表示执行证据一致，不能解释为按键成功。

要区分模型预测、IK 可行性约束和真实物理跟踪各自的影响，可在完成后运行记录诊断：

```bash
.conda/envs/lerobot/bin/python scripts/diagnose_policy_eval.py \
  --run outputs/policy_eval_step2000/eval_new \
  --output outputs/policy_eval_step2000/eval_new/diagnosis.json
```

诊断使用独立 FK 实现，比较原始模型位姿隐含的杆尖、IK 命令杆尖和物理实测杆尖到按钮的距离，同时统计整个过程与最后 5 秒的投影误差、跟踪误差和速度限制次数。`--last-seconds` 可改变末段统计窗口；读取尚在运行的目录时，只分析已有 `metadata.json` 的完整 episode，并标记 `run_complete=false`。

## 5090 仿真、H200 推理的跨机器评估

本流程使用 5090 上的 Isaac Sim 5.0 运行场景，H200 上的 VLA-JEPA 服务执行推理。原采集机使用 Isaac Sim 4.5，因此比较须保留软件版本和图像差异证据，不能仅凭相同场景文件就认定物理或渲染结果完全一致。

此次跨机器对照已完成：同一 `step_010600`、同一冻结场景和 60 个条件，5090 为 **20 次成功、22 次误按、18 次超时**，本机基线为 **5、27、28**；两端均无异常碰撞，60 条候选记录全部通过独立审计。34/60 条的终止原因和实际按下楼层一致，原 5 个成功条件全部保留。另将四组历史无损输入经 5090 发到 H200，原始 pose9 和转换后 pose8 动作均逐值相同。初始 TCP 位置的最大差异约 2.84×10⁻⁹ m，初始全局／腕部图像平均像素 MAE 分别为 3.43／5.79（0–255）。这些结果证明当前链路可运行，但不能将两套仿真视为可直接互换的评测基线；渲染和物理各自对结果的影响尚未单独分离。5090 保留其部署时用于清除灯光残影的 16 次稳定渲染，本机历史实现为 4 次。该差异与引擎版本一起记录，未修改控制器或模型来匹配结果。

本地产物：配对报告 `outputs/rtx5090_eval_comparison/results/summary.md`、详细指标 `results/comparison.json`、固定输入复测 `outputs/rtx5090_eval_comparison/golden_replay.json`、33 层中心位置双相机并排视频 `outputs/rtx5090_eval_comparison/results/compare_floor33_center.mp4`。5090 保留全部原始观测 PNG、轨迹和视频；本机证据包取回了全部轨迹／视频与每条 episode 的首个无损观测。本次四个临时推理服务已停止，SSH 主连接仍可复用。

先从采集机转移原始 `scene.usda`、`config.json`、数据集的 `meta/collection_metadata.json`，以及完整资产依赖包。本次输入组织为 `outputs/rtx5090_eval_comparison/input/`：

```text
input/
  scene.usda
  config.json
  dataset_metadata/meta/collection_metadata.json
  assets/asset_bundle.json
  assets/assets/...
  assets/vendor/...
  golden_requests/worker_0/{request.json,global.png,wrist.png}
  golden_requests/worker_1/...
  golden_requests/worker_2/...
  golden_requests/worker_3/...
```

`asset_bundle.json` 的 `schema_version=1`，记录 `source_project_root` 及 `files`；每项包含原机器的绝对 `source_path`、项目内 POSIX `relative_path`、`sha256` 和 `bytes`。文件放在包根目录下对应的 `relative_path`。此冻结场景的完整闭包是 **10 个文件**：PiPER 主 USD 及三个配置 USD、桌子 USD、支架 USD、桌子纹理 PNG、官方腕部兼容 USD、两张文字标签 PNG。生成的腕部资产和标签也必须复制原字节；只重新下载官方源资产不足以复现此场景。引擎资源 `OmniPBR.mdl` 保留原名称，由运行中的 Isaac 提供。评估只需要上述数据集元信息，不需要传输训练集视频或动作。

`--asset-bundle` 会验证全部文件大小、SHA256 和相对引用闭包，再生成本次输出目录下的 `runtime_scene.usda`。原 `scene.usda` 保持不变，仍用其 SHA256 校验训练场景；本次原 SHA 为 `d5f5556825db36635dd1636561bc0caa75bdd156d93b6e254922239450e9741a`。运行时副本只替换七条绝对资产路径，并通过逆映射检查全部 authored USD 内容一致。`scene_relocation.json` 和 `eval_manifest.json` 分别保存原始／运行时文件哈希、完整依赖、映射及报告哈希，审计会重新验证。

以下 SSH 命令均在 **5090 工作站**执行。先检查并复用到 H200 的主连接；仅在它不存在时，在一个保持开启的终端建立连接：

```bash
ssh -o ConnectTimeout=120 -S /tmp/pressb-5090-h200-eval.sock -O check h200

# 仅在没有主连接时执行，并保留此终端。
ssh -o ConnectTimeout=120 -M -S /tmp/pressb-5090-h200-eval.sock \
  -o ControlPersist=no -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N h200
```

另一个 5090 终端复用连接登录 H200：

```bash
ssh -o ConnectTimeout=120 -S /tmp/pressb-5090-h200-eval.sock h200
```

在该 **H200 shell** 启动模型服务。以下示例占用 H200 的逻辑 GPU 0；运行前按空闲显存选择 GPU。服务仅监听 loopback。本次临时服务会在评估结束后停止，后续评估需重新启动：

```bash
cd /home/pengguanqi/Worksapce/Research/VLA-JEPA
conda activate vlajepa-piper
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m scripts.serve_piper_policy \
  --checkpoint /data/scratch/pengguanqi/VLA-JEPA-runs/piper_panel_stratified_press30hz_pose9_3epochs_20260929/checkpoints/step_010600 \
  --port 19765 --device cuda:0
```

回到 **5090**，在已有主连接上添加本地 loopback 转发；已存在时无需重复添加：

```bash
ssh -o ConnectTimeout=120 -S /tmp/pressb-5090-h200-eval.sock \
  -O forward -L 127.0.0.1:19765:127.0.0.1:19765 h200
curl --fail http://127.0.0.1:19765/health
```

健康检查应显示 `status=ready`、`checkpoint_verified=true`、`checkpoint_step=10600`，权重 SHA256 为 `2456b1fff5ef2d94a173b502d55b244a6b0810b92c6e39fc6b1527e1d6019312`。以下在 5090 上运行一组 24–26 层、每层中心和四角各一次，共 15 条；种子、3 点均值滤波、3 个并行环境、15 秒上限均与原对应 worker 保持一致：

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
  --seed 20260930 --smoothing-window 3 --num-envs 3 --max-seconds 15 --gpu 0 \
  --output outputs/rtx5090_eval_comparison/formal_new/worker_0
```

完整对比的其他组分别为 `27,28,29`、`30,31,32`、`33,34,35`，输出到各自的 `worker_1..3`；可依次运行，也可按显存配置并发。四服务并行时，H200 分别使用端口 `19765..19768` 和逻辑 GPU `0..3`，为每个端口建立对应 loopback 转发，5090 的 `--gpu` 只能使用其本机 GPU 编号。不要更改每组的种子、平滑、布局、时长或环境数量；所有输出目录须为新目录。本次实际运行目录为 `outputs/rtx5090_eval_comparison/center_corners_v1/worker_0..3`，上述 `formal_new` 是后续重跑示例。

完成后先在 **5090 原位置**进行独立审计。重定位审计需要可读取运行时 USD 的 `usd-core`；5090 已有隔离安装的兼容 USD，使用如下入口，不启动 Isaac：

```bash
PYTHONPATH=.cache/usd-inspect:src .conda/envs/pressb/bin/python \
  scripts/audit_policy_eval.py \
  --run outputs/rtx5090_eval_comparison/formal_new/worker_0 \
  --report outputs/rtx5090_eval_comparison/formal_new/worker_0/audit.json \
  --allow-partial
```

该审计 Python 还需 NumPy、Pillow、PyAV。`--allow-partial` 仅允许此 worker 没有覆盖全部十二层，其自身声明的 15 条调度仍须全部完成。其余 worker 同样审计；保留输入包及原始路径，使报告中的文件身份能够再次验证。

### 固定输入复测与闭环对比

固定输入（golden）测试从原评估复制无损 PNG、实测状态、任务和种子，再通过 5090→H200 请求同一检查点，比较返回的原始动作。这用于隔离传输、预处理和推理差异；固定输入不驱动本次仿真。示例在 5090 项目目录执行，复测四个已有输入：

```bash
PYTHONPATH=src .conda/envs/pressb/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from pressb.policy_eval import PolicyClient, validate_policy_reply

root = Path('outputs/rtx5090_eval_comparison')
client = PolicyClient('http://127.0.0.1:19765/predict', timeout=120)
rows = []
for directory in sorted((root / 'input/golden_requests').glob('worker_*')):
    recorded = json.loads((directory / 'request.json').read_text())
    images = {}
    for view in ('global', 'wrist'):
        path = directory / f'{view}.png'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == recorded['images'][view]['sha256']
        images[view] = np.asarray(Image.open(path).convert('RGB'))
    reply, seconds = client.predict(recorded['task'], recorded['state'], images, recorded['seed'])
    validate_policy_reply(reply)
    differences = {key: float(np.max(np.abs(np.asarray(reply[key]) - np.asarray(recorded['response'][key]))))
                   for key in ('actions_pose8', 'actions_pose9')}
    rows.append(dict(worker=directory.name, seed=recorded['seed'], response=reply,
                     max_absolute_differences=differences, wall_seconds=seconds))
with (root / 'golden_retest_new.json').open('x') as stream:
    json.dump(rows, stream, indent=2)
print(json.dumps([{'worker': row['worker'], **row['max_absolute_differences']} for row in rows]))
PY
```

完整闭环评估则持续使用 **5090 当前渲染图像和实测状态**，从第一帧到终止独立执行。即使 golden 输出一致，跨版本画面或物理差异仍可能让后续观察与动作分化；首帧画面和初始状态接近也不代表任务结果相同。

待两端所有 worker 完成并审计后，把候选运行目录完整复制回保存原基线的机器，再运行 CPU 对比脚本。例如比较本次目录：

```bash
.conda/envs/pressb/bin/python scripts/compare_policy_eval_runs.py \
  --baseline-root outputs/policy_eval_step10600_stratified/center_corners_v1 \
  --candidate-root outputs/rtx5090_eval_comparison/center_corners_v1 \
  --output outputs/rtx5090_eval_comparison/comparison_new
```

脚本按楼层、重复号和面板位置配对，核对 checkpoint、场景配置、种子、控制参数及覆盖条件，输出 `comparison.json` 和 `summary.md`：包括逐条件成功／误按／超时、实际按下楼层、终止时间、距离、首帧图像和首个动作差异，并保留运行环境及源码差异。未完成、失败或条件不匹配时返回退出码 2，不能计作模型超时。`comparison_valid=true` 表示数据可比较；是否效果一致仍须阅读配对结果，不等同于逐帧或逐动作完全相同。

## 保存的证据

运行根目录含 `eval_manifest.json`（源码和场景身份）、`policy_service.json`（权重身份）、`lighting.json`、`status.json`、`report.json`。每个 `episode_XXXXXX` 含：

- `physics.npz`：120 Hz 实测关节、下发命令、实测 TCP、按钮位移、接触力、灯光、世界系杆端点、动作与物理步索引。
- `frames.npz`：与相机同步的实测状态、关节、灯光和准确物理时间。
- `requests.jsonl`：任务、种子、观察状态、图像哈希与路径、完整模型响应及网络推理耗时。
- `observations/chunk_XXXX_{global,wrist}.png`：每次真正送入模型的无损实时图像。
- `actions.jsonl`：实际执行的 chunk 行、原始模型动作、IK 诊断、插值端点和执行起止物理步。终止后未执行的预测行仍可在原始响应中找到。
- `global.mp4`、`wrist.mp4` 及初末帧 JPG：可视化整个运行及终止状态。
- `metadata.json`：按钮事件、碰撞、终止原因、成功判据、轨迹误差与源文件哈希。

完成评估后，即使所有策略尝试均失败，仿真进程也返回完成状态并保留证据。报告中的任务成功数和独立审计结论应分别阅读。
