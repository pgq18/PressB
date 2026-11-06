import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "packages"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dataclasses
import enum
import logging
import socket
from pathlib import Path
from typing import List, Union
import tyro
from dataclasses import field
import yaml
import ast
from openpi_client import websocket_policy_server
from wall_x.serving.policy.wall_x_policy import WallXPolicy
from wall_x.data.utils import load_norm_stats
import torch
import numpy as np

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, norm_stats):
        self.norm_stats = norm_stats
        self.action_min_stat = norm_stats["action"].min.cpu().numpy()
        self.action_delta = norm_stats["action"].delta.cpu().numpy()
        self.state_min_stat = norm_stats["state"].min.cpu().numpy()
        self.state_delta = norm_stats["state"].delta.cpu().numpy()

    @classmethod
    def _normalize(cls, action, min_stat, delta):
        """
        Normalize action data using min-max normalization.
        """
        # Convert inputs to the same type (numpy) for consistent operations
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
            
        # Ensure delta is not zero to avoid division by zero
        delta = np.where(delta == 0, np.ones_like(delta), delta)
        
        # Perform normalization
        x = (action - min_stat) / delta
        x = x * 2 - 1
        x = np.clip(x, -1, 1)
        return x
    
    def in_process(self, input_data):
        # Keep the output as numpy array as requested
        input_data["state"] = self._normalize(input_data["state"], self.state_min_stat, self.state_delta)
        return input_data


class EnvMode(enum.Enum):
    """Supported environments/datasets."""

    LIBERO = "libero"
    ALOHA = "aloha"


@dataclasses.dataclass
class ModelConfig:
    """Configuration for loading a Wall-X model."""

    # Path to the pretrained model checkpoint
    model_path: str
    # Path to train config yaml
    train_config_path: str
    # Path to the action tokenizer
    action_tokenizer_path: str | None = None
    # Action dimension for the environment
    action_dim: int = 7
    # State dimension for the environment
    state_dim: int = 8
    # Prediction horizon (number of future actions to predict)
    pred_horizon: int = 32
    # Device to run model on
    device: str = "cuda:0"
    # Model dtype (bfloat16, float16, float32)
    dtype: str = "bfloat16"
    # Prediction mode (fast or slow)
    predict_mode: str = "diffusion"
    # Camera key for the environment (can be string like "['face_view', 'left_wrist_view']" or list)
    camera_key: Union[List[str], str] = field(default_factory=lambda: ["face_view", "left_wrist_view"])

    # Input image pre-resize resolution for the longer edge (None means no pre-resize)
    input_image_resolution: int | None = None
    # Path to norm_stats.json (overrides train_config_path setting if provided)
    norm_stats_path: str | None = None
    # Print token composition debug info on first inference
    debug_tokens: bool = False

    def __post_init__(self):
        """Parse camera_key if it's a string representation of a list."""
        if isinstance(self.camera_key, str):
            try:
                self.camera_key = ast.literal_eval(self.camera_key)
            except (ValueError, SyntaxError):
                # If parsing fails, treat as single key
                self.camera_key = [self.camera_key]
        elif isinstance(self.camera_key, list) and len(self.camera_key) == 1 and isinstance(self.camera_key[0], str):
            # Handle case where tyro wraps string as single-element list
            try:
                parsed = ast.literal_eval(self.camera_key[0])
                if isinstance(parsed, list):
                    self.camera_key = parsed
            except (ValueError, SyntaxError):
                pass


@dataclasses.dataclass
class Args:
    """Arguments for the serve_wall_x script."""

    # Environment mode (used for default configurations)
    env: EnvMode = EnvMode.LIBERO

    # Model configuration. If not provided, uses default config for the environment
    model_config: ModelConfig | None = None

    # Default text prompt to use if not provided in observation
    default_prompt: str | None = None

    # Port to serve the policy on
    port: int = 8000

    # Host to bind the server to
    host: str = "0.0.0.0"

    # Enable debug logging
    debug: bool = False

    # Enable Real-Time Action Chunking (RTC) mode
    use_rtc: bool = False

    # RTC: step at which background inference starts
    rtc_s: int = 16

    # RTC: deterministic region length (steps to wait after s before swapping)
    rtc_d: int = 8

    # RTC: action horizon for the broker
    rtc_action_horizon: int = 32


# Default model configurations for each environment
DEFAULT_CONFIGS: dict[EnvMode, ModelConfig] = {
    EnvMode.LIBERO: ModelConfig(
        model_path="/path/to/model",
        action_tokenizer_path="/path/to/action_tokenizer",
        train_config_path="/path/to/train_config",
        state_dim=8,
        action_dim=7,
        pred_horizon=32,
        device="cuda",
        dtype="bfloat16",
        predict_mode="fast",
        camera_key=["front_view", "left_wrist_view"],
    ),
    EnvMode.ALOHA: ModelConfig(
        model_path="/path/to/model",
        action_tokenizer_path="/path/to/action_tokenizer",
        train_config_path="/path/to/train_config",
        state_dim=14,
        action_dim=14,
        pred_horizon=32,
        device="cuda",
        dtype="bfloat16",
        predict_mode="fast",
        camera_key=["face_view", "left_wrist_view", "right_wrist_view"],
    ),
}

def get_model_config(args: Args) -> ModelConfig:
    """Get model configuration from args or defaults."""
    if args.model_config is not None:
        return args.model_config

    if config := DEFAULT_CONFIGS.get(args.env):
        logger.info(f"Using default configuration for {args.env.value}")
        return config

    raise ValueError(
        f"No default configuration for {args.env.value}. "
        f"Please provide --model-config with model_path and action_tokenizer_path."
    )

def create_policy(args: Args) -> WallXPolicy:
    """Create a Wall-X policy from the given arguments."""
    config = get_model_config(args)
    logger.info(f"Creating Wall-X policy with config: {config}")

    # Validate paths
    if not Path(config.model_path).exists():
        logger.warning(f"Model path does not exist: {config.model_path}")

    if config.action_tokenizer_path and not Path(config.action_tokenizer_path).exists():
        logger.warning(
            f"Action tokenizer path does not exist: {config.action_tokenizer_path}"
        )

    train_config = load_config(config.train_config_path)
    if config.norm_stats_path is not None:
        # The model constructs its action/state normalizers from this config;
        # use the same override as the observation DataProcessor below.
        train_config["norm_stats_path"] = config.norm_stats_path

    policy = WallXPolicy(
        model_path=config.model_path,
        train_config=train_config,
        action_tokenizer_path=config.action_tokenizer_path,
        action_dim=config.action_dim,
        agent_pos_dim=config.state_dim,
        pred_horizon=config.pred_horizon,
        device=config.device,
        dtype=config.dtype,
        predict_mode=config.predict_mode,
        default_prompt=args.default_prompt,
        camera_key=config.camera_key,
        input_image_resolution=config.input_image_resolution,
        rtc_s=args.rtc_s,
        rtc_d=args.rtc_d,
        debug_tokens=config.debug_tokens,
    )

    return policy

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["data"]["model_type"] = config.get("model_type")

    return config

def main(args: Args) -> None:
    model_config = get_model_config(args)
    print(f"[DEBUG] camera_key: {model_config.camera_key}, type: {type(model_config.camera_key)}")
    config = load_config(model_config.train_config_path)
    # Only repo_id is needed for inference. Importing the training dataset
    # loader here unnecessarily requires the entire LeRobot training stack.
    lerobot_config = config["data"].get("lerobot_config", {})
    norm_stats = load_norm_stats(
        model_config.norm_stats_path or config.get("norm_stats_path", None),
        lerobot_config.get("repo_id", None)
    )
    print(norm_stats)
    dp = DataProcessor(norm_stats)
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Include RTC config in metadata for clients
    policy_metadata["rtc_config"] = {
        "enabled": args.use_rtc,
        "s": args.rtc_s,
        "d": args.rtc_d,
        "action_horizon": args.rtc_action_horizon,
    }

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        data_processor=dp,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()

if __name__ == "__main__":
    main(tyro.cli(Args))
