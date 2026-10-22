# PressB online RL wire contract (v1)

Three independent processes: simulation owns Isaac and physical state; inference owns a frozen VLA-JEPA; learner owns SAC, replay and rollout scheduling. The learner connects to both services; simulation and inference do not call each other. Simulation and learner status use JSON over HTTP. Frozen inference defaults to the actual Wall-X WebSocket server in `ref/wall-x/packages/openpi_client/websocket_policy_server.py`, with MessagePack and native NumPy arrays. An explicit `--transport http` retains the historical inference API. Model inference and network waits do not advance simulation time. This is continuous online collection, not a guarantee of real-time wall-clock control.

All successful RPC responses include `protocol_version: 1`. HTTP GET `/health` reports `service` (`simulation`, `inference`, `learner`) and `ready`; POST handlers take a JSON object. Over WebSocket, inference health is an RPC operation after the metadata handshake, rather than an HTTP GET on `/health`. Clients never automatically retry state-mutating requests. Optional authentication uses `PRESSB_RL_TOKEN`: `Bearer` on HTTP, `Api-Key` on the WebSocket upgrade. Default bind is loopback. SSH forwarding or a private network connects separate hosts.

## WebSocket inference transport

`InferenceClient(endpoint, timeout, token)` selects WebSocket for `ws://` / `wss://` and the legacy JSON client for `http://` / `https://`. The learner still calls `client.call("/encode", payload)`, `"/decode"`, `"/release"`, and `"/health"`; only the inference transport changes. The simulation client remains `RPCClient`.

The Wall-X server sends metadata immediately after connection. The client requires `pressb_rl: {version: 1, operations: ["health", "encode", "decode", "release"]}` before sending RL operations. Metadata also advertises `state_dim: 8`, `action_dim: 9`, `pred_horizon: 7`, and `rtc_config: {enabled: false}`. A legacy Wall-X model server that only implements `policy.infer()` is not sufficient for residual/noise RL: its backend must also provide encoder features, cached conditioning and explicit-noise flow decoding. This adapter supplies those operations for the existing frozen VLA-JEPA without changing its model or weights.

Split requests are MessagePack dictionaries with the envelope `{"__pressb_rpc__": {"version": 1, "method": "encode", "payload": {...}}}`. Methods are `health`, `encode`, `decode`, and `release`. NumPy arrays retain shape and dtype with the Wall-X `openpi_client.msgpack_numpy` wire format; the PressB client implements that format without importing the reference checkout. The client removes base64 from the simulator's PNG observations and sends the same lossless PNG bytes, without decoding or re-encoding the image. Returned features and actions use native arrays instead of JSON lists or base64 feature strings. The server additionally accepts native uint8 RGB arrays. The simulation HTTP observation and action formats below remain unchanged. The adapter adds no image resize, quantization or normalization.

The server also preserves the ordinary Wall-X observation-to-`policy.infer` path. For this VLA-JEPA adapter, its input aliases are `face_view` (global RGB), `left_wrist_view` (wrist RGB), `prompt` (task text), and `state`. These names do not change the checkpoint's task or numerical contract: state is the current eight-component TCP pose/gripper vector, and learned actions are seven poses with nine XYZ/rotation6D components. They are not Wall-X's seven joint/gripper values. The existing elevator policy still requires its trained task strings and fixed gripper. Accepting the common transport does not adapt the new pick-and-place task or load a Wall-X model.

The server's HTTP `/healthz` is a liveness check only; it does not replace checkpoint/capability verification through inference RPC. Use the client to inspect the model service:

```python
from pressb.online_rl.inference_transport import InferenceClient

client = InferenceClient("ws://127.0.0.1:19891")
try:
    print(client.call("/health"))
finally:
    client.close()
```

The learner's `online-rl` extra supports `websockets>=12,<16` and `msgpack>=1,<2`; the separate model service needs the `inference-server` extra with `websockets>=14,<16`. Keep the Isaac/learner environment's `websockets==12.0` pin required by `isaacsim-kernel==5.0.0`; install the newer server dependency only in the inference environment. See [the launch instructions](online_rl_fast.md#启动). The server is loaded from the reference Wall-X checkout, not a second copied server implementation inside `pressb`; `sync_rl_inference.py` transfers the required reference serving files alongside the addon for a separate inference host.

Validation on 2026-10-22: 209 targeted tests passed, including 21 real-socket WebSocket contracts and complete CPU rollout/learner loops. The actual step-10600 checkpoint was also served on two RTX 5090s through HTTP and WebSocket with identical preprocessing and batch size 2. For two saved observations, pooled features, state, base actions and explicit-noise decoded actions were exactly equal; the original Wall-X client also returned an exactly matching single-observation action. The client used the Isaac environment's WebSockets 12.0 against the isolated server's 15.0.1. Reports are in `outputs/wallx_interface_validation_20261022/gpu_report.json` and `pytest.log`. These checks validate the interface and sampled numerical equivalence; they do not measure new training throughput or task success rates.

## Shared observation

`observation = {task, state: float[8], images: {global: base64_png, wrist: base64_png}, control_state: float[N]}`.

The original RGB contract is uint8 480x640x3. The inference endpoint also accepts 224x224x3 lossless RGB PNGs; `/health.accepted_image_sizes` lists sizes in width/height order. Moving the original PIL bilinear 640x480-to-224x224 resize before transport preserves its pixels. The historical policy intentionally resizes the full 4:3 camera image into a square image, without cropping; rendering directly into a square camera image does not preserve that projection.

Keep the simulator render aspect ratio at 4:3, for example native 320x240 followed by PIL `Image.Resampling.BILINEAR` resize to the 224x224 transport image, or the original 640x480 path. In the Isaac camera path, changing only the render resolution to native 224x224 changes the vertical field of view; retaining the USD camera apertures alone is insufficient. The earlier native-square optimization therefore changed the observation distribution and must not be treated as the historical camera contract. Record render resolution and transport resolution separately. Lower-resolution 4:3 rendering preserves the intended field of view, but sampling and antialiasing can still differ from 640x480, so validate it with a base-policy rollout comparison before attributing success-rate changes to RL.

Both original and adapted inference first convert RGB to the policy's 224x224 image, then use the saved `Qwen2VLImageProcessorFast`. For this checkpoint its minimum pixel count is 65,536 and patch/merge alignment is 32, giving a subsequent 256x256 processor image; rescaling is 1/255 with mean/std 0.5. Matching the final tensor shape does not repair a different field of view at capture time.

State is `[x,y,z,qw,qx,qy,qz,width]`, absolute base_link gripper TCP, metres, TCP offset 0.1358 m, fixed commanded width 0.008 m. `control_state` contains actual joint angles (6), velocities (6), preceding IK endpoint (6), causal linear-command history oldest first (6*smoothing_window), and elapsed/maximum episode time (1). Its schema and dimension are in simulation health. No panel offsets/button coordinates enter observation. Learner observation concatenates pooled encoder feature, pose9, control_state and 12-way floor one-hot from the task.

## Simulation

GET `/health`: scene/config/collection identity, number of environments, physics_hz=120, action_fps=30, action_horizon=7, smoothing_window, control_state_dim, max_seconds, single_gamma, reward contract.

POST `/reset`: `{run_id, request_id, seed, episodes: [{floor, offset_x_m, offset_y_m}, ...]}`. Exactly num_envs episodes. Only allowed initially or after ALL environments end. Creates a `cohort_id`, sets `step_id=0`, and returns `{cohort_id, step_id, items:[{env_id, episode_id, observation, terminated:false, truncated:false, info}], all_done:false}`. Episodes must be within configured training panel bounds; resets preserve physical scene validation. `run_id` owns the server session. POST `/close_run`: `{run_id, request_id}` releases ownership only after all environments have ended. An interrupted active run requires simulator restart, not an implicit reset.

POST `/step`: `{run_id, request_id, cohort_id, step_id, actions:[{env_id, actions_pose8:float[7,8]}, ...]}`. Exactly the currently active environments must appear once. Validate the entire request before stepping. Matching cached request_id+payload returns the previous response, without repeating physics; stale cohort/step or reused ID with different payload is rejected. The default cache keeps two image-bearing responses; evicted request IDs are rejected rather than executed again. Return updated `step_id` and all vector `items`, each with `observation, reward, discount, terminated, truncated, executed_physics_steps, info`. Already-done environments have reward=0, executed_physics_steps=0 and their immutable terminal observation. No autoreset. At a partial terminal tick, capture terminal observation immediately.

Reward is sparse: success=1 at the physical event, all other ticks=0. For k executed physical ticks and g=single_gamma (per 30Hz action), return `reward = g**((k-1)/4)` on success else 0 and `discount = g**(k/4)`. Success/wrong press/collision terminate, time limit truncates; infrastructure exceptions fault the service and are not episodes. Record raw success separately from discounted reward. The learner defaults to no timeout bootstrap for the finite 15s objective; configurable timeout bootstrap uses actual final observations, never reset observations.

## Frozen inference

The route names and JSON representations below document the legacy `--transport http` API. The default WebSocket adapter exposes the same operations and numerical results through the MessagePack envelope above, accepts binary PNG/native RGB inputs, and returns native feature/action arrays. Inference-side context storage and batch scheduling are shared between both transports.

GET `/health`: verified checkpoint step/SHA, model/source identities, feature_dim=2048, feature_schema=mean_embodied_action_tokens, pose/action contracts and bounded cache information. All parameters are frozen; no optimizer or training path. Capabilities additionally include:

- `encoder_batch_mode`: `serial` or `batched`; `microbatch_size` sets the maximum samples per encoder/flow forward.
- `image_preprocess_device`: `cpu` or `cuda`; `torch_threads` records CPU preprocessing parallelism.
- `accepted_image_sizes`: `[[640,480],[224,224]]`.
- `feature_encodings`: `["json","float32_base64"]`.
- `encode_release_context_ids`: `true` when releasing old contexts can be combined with encoding.

POST `/encode`: `{observations:[observation,...], seeds:[int,...], include_base_actions:bool, release_context_ids?:[string,...], feature_encoding?:"json"|"float32_base64"}` -> `{items:[{context_id, feature:float[2048], state_pose9:float[9], base_actions_pose9:float[7,9] (if requested), base_actions_pose8 (if requested)}]}`. The default `json` response remains backward compatible. Each context holds full 32x2048 conditioning tokens and state9 on the inference device. Feature is mean pooled tokens in float32 only for RL; flow always uses all tokens. One seed is required per observation, as an integer in `[0,2**63)`; requests contain 1..cache_size observations.

With `feature_encoding="float32_base64"`, each item **omits `feature`** and instead contains `feature_f32_b64:string` and `feature_dim:2048`. Decode with strict base64 validation into exactly `2048 * 4` bytes, interpreted as contiguous little-endian float32; require finite values. This is lossless transport, with no feature quantization. Other item fields are unchanged. Clients should negotiate using `feature_encodings` and retain JSON fallback for older servers.

Optional `release_context_ids` removes one round trip between consecutive chunks. The server validates the complete request, finishes every microbatch and action pose conversion, then releases the listed old IDs and commits new contexts. A failed encoding does not release previous contexts. Duplicate or already-expired release IDs are harmless. Default bounded-cache eviction still applies when old and new contexts exceed capacity.

`--batch-size 1 --image-preprocess-device cpu` retains the historical serial path. With `--batch-size N` for N>1, each microbatch executes one multimodal encoder forward; requested base actions execute the original four-step flow once over that batch. Per-sample generators retain the original `[1,7,9]` random draw shape, seed and dtype, so regrouping samples does not change their initial noise. Batched BF16 kernels can change features and final actions slightly; sample seeds do not promise byte-identical actions across batch sizes.

`--image-preprocess-device cuda` runs the saved fast image processor on the inference GPU with image grouping. Resize/normalization settings, patch layout, tokenizer and frozen weights remain fixed, but CPU/CUDA image interpolation rounding can differ. This is an explicit numerical configuration and should be recorded in run identity and evaluation settings. The default remains CPU. `--torch-threads` controls CPU intra-op threads (default 4). Real saved-observation numerical and throughput checks are recorded in `outputs/online_rl_inference_perf_20261002`; those 25 distinct saved samples cover floor 24, five layouts and the first five action chunks, and do not replace complete-task or all-floor evaluation. The separate H200/RTX5090 HTTP comparison uses the same saved samples and is recorded in `outputs/online_rl_local_inference_20261002/service_comparison.json`.

POST `/decode`: `{items:[{context_id, initial_noise:float[7,9]},...]}` -> `{items:[{context_id, actions_pose9, actions_pose8}]}`. Exactly the existing four-step flow Euler integration, with explicit initial noise and cached conditioning; no second Qwen pass. In batch mode, valid contexts and noises are concatenated and decoded in microbatches, with outputs in request order. All IDs and finite noise arrays are checked before any flow call. POST `/release`: `{context_ids:[...]}`. Cache is bounded and expires; missing contexts are errors, not silent re-encodes. Repeated `/decode` with the same contexts, noise and batch grouping is deterministic; different batch grouping may introduce floating-point rounding differences.

## Learner and methods

Learner is an HTTP client of simulation and a WebSocket client of inference by default; HTTP inference remains supported when explicitly selected by its URL. It exposes `/health`, `/status`, `/stop` over HTTP on its own port. Only this process imports/updates SAC networks. Each transition is one executed chunk, retaining actual duration discount, terminal final observation, executed RL action, and (for residual) current/next base action. Terminal samples never use a reset observation for bootstrap. Whole-vector reset preserves active-episode clocks. Replay stores features, not RGB or model weights. Checkpoints store optimizer/temperature/RNG/config/base identity; fresh-simulator resume boundaries are explicit.

`action_residual`: actor emits tanh-Gaussian u[7,9]; action9=base9+scale9*u, then rotation rows project to SO(3), quaternion conversion, fixed gripper. Default scale is 0.03 m for XYZ and 0.1 for six dimensionless rotation components; configurable and saved. ZPRL residual actor conditions on observation+base action; critic evaluates combined action. Preserve residual SAC target without entropy and bounded [0,1] critics for this nonnegative finite success objective. CAPS is optional and explicit, disabled by default to isolate the requested residual method.

`initial_noise`: actor emits u[n_noise_steps,9] in [-1,1], noise=1.5*u, repeated periodically to 7 steps. Default n_noise_steps=1; only divisors of 7 (1 or 7) are valid. This reproduces ZPRL's direct initial-noise selection, NOT additive Gaussian noise residual. Warmup uses clamp(N(0,1), -scale, scale)/scale and saves the executed normalized noise. Noise SAC uses unbounded critics and entropy in target. A separate base-policy evaluation uses ordinary Gaussian initial noise; zero noise is not the baseline.

Training reports chunk transitions, 30Hz-equivalent control steps, actual physical ticks, updates, episodes and measured successes separately. Small integration runs establish executability only, not improved success rate. Evaluation freezes the learner and uses explicit seeds/layout schedules in separate output directories.
