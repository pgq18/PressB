"""
LeRobot Dataset Loader - Distributed Version
"""

import numpy as np
import torch
from torch.utils.data import DistributedSampler, random_split
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.transforms import (
    ImageTransformConfig,
    ImageTransforms,
    ImageTransformsConfig,
)
from typing import Protocol, SupportsIndex, TypeVar
from qwen_vl_utils.vision_process import smart_resize
from wall_x.data.config import X2RDataProcessingConfig
from wall_x.data.utils import (
    process_grounding_points,
    get_wallx_normal_text,
    replace_action_token,
    preprocesser_call,
)

from transformers import AutoProcessor
from .utils import load_norm_stats, KEY_MAPPINGS

T_co = TypeVar("T_co", covariant=True)


def build_image_transforms(image_transforms_config):
    if not image_transforms_config or not image_transforms_config.get("enable", False):
        return None

    config_kwargs = {
        "enable": image_transforms_config.get("enable", False),
        "max_num_transforms": image_transforms_config.get("max_num_transforms", 3),
        "random_order": image_transforms_config.get("random_order", False),
    }

    transform_configs = image_transforms_config.get("tfs")
    if transform_configs is not None:
        config_kwargs["tfs"] = {
            name: ImageTransformConfig(**transform_config)
            for name, transform_config in transform_configs.items()
        }

    return ImageTransforms(ImageTransformsConfig(**config_kwargs))


def split_lerobot_episodes(episodes_num, train_test_split=0.95, split_seed=42):
    """
    Deterministically split episode ids into train/test sets with a seeded shuffle.
    This keeps held-out test episodes distributed across the dataset instead of
    always taking the final contiguous episodes.
    """
    if not 0 < train_test_split < 1:
        raise ValueError(
            f"train_test_split must be between 0 and 1, got {train_test_split}"
        )
    if episodes_num < 2:
        raise ValueError(f"episodes_num must be at least 2, got {episodes_num}")

    episodes = np.random.default_rng(split_seed).permutation(episodes_num).tolist()
    train_count = int(episodes_num * train_test_split)
    train_count = min(max(train_count, 1), episodes_num - 1)
    return episodes[:train_count], episodes[train_count:]


# Abstract class for dataset
class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class PreprocessedDataset(Dataset[T_co]):
    def __init__(
        self,
        dataset,
        config,
        dataload_config,
        norm_stats,
        lerobot_config,
        seed=42,
        rank=0,
        world_size=1,
        test_only=False,
        train_episodes=None,
        test_episodes=None,
        split_seed=None,
        train_test_split=None,
    ):
        self.hf_dataset = dataset
        self.train_episodes = train_episodes
        self.test_episodes = test_episodes
        self.split_seed = split_seed
        self.train_test_split = train_test_split

        if test_only:
            self._dataset = dataset
        else:
            self._dataset = None
            self.train_dataset, self.val_dataset = random_split(
                dataset,
                [0.95, 0.05],
                torch.Generator().manual_seed(seed) if seed is not None else None,
            )
            self._train()

        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        # init configs
        self.config = config
        self.use_fast_tokenizer = self.config.get("use_fast_tokenizer", False)
        self.dataload_config = dataload_config
        self.norm_stats = norm_stats
        self.lerobot_config = lerobot_config

        self.data_config = X2RDataProcessingConfig().update(
            train_test_split=self.dataload_config["train_test_split"],
            split_seed=self.dataload_config["split_seed"],
            predict_action_keys=self.dataload_config["predict_action_keys"],
            obs_action_keys=self.dataload_config["obs_action_keys"],
            resolution=self.dataload_config.get("resolution", None),
            priority_order=self.dataload_config.get("priority_order", None),
        )

        self._cam_key_mapping = KEY_MAPPINGS[self.hf_dataset.meta.repo_id]["camera"]
        self._state_key_mapping = KEY_MAPPINGS[self.hf_dataset.meta.repo_id]
        self._action_key_mapping = KEY_MAPPINGS[self.hf_dataset.meta.repo_id]

    def _vision_preprocess(self, frames):
        processed_frames = []
        for key in self.hf_dataset.meta.camera_keys:
            from PIL import Image

            current_obs = frames[key].clone().permute(1, 2, 0)

            img_pil = Image.fromarray((current_obs * 255).to(torch.uint8).cpu().numpy())
            orig_width, orig_height = img_pil.size
            # 2. Apply resolution constraints (if config is not -1)
            target_size = self.data_config.resolution.get(
                self._cam_key_mapping[key], -1
            )
            if target_size != -1:
                # Maintain aspect ratio logic
                if orig_width > orig_height:  # Landscape image
                    new_width = target_size
                    new_height = int(target_size * orig_height / orig_width)
                else:  # Portrait image
                    new_height = target_size
                    new_width = int(target_size * orig_width / orig_height)
                img_pil = img_pil.resize((new_width, new_height))

            # 3. Apply smart scaling (qwen logic)
            current_width, current_height = img_pil.size
            resized_height, resized_width = smart_resize(
                current_height,
                current_width,
                factor=self.data_config.image_factor,
                min_pixels=self.data_config.min_pixels,
                max_pixels=self.data_config.max_pixels,
            )
            resized_img = img_pil.resize((resized_width, resized_height))
            processed_frames.append(resized_img)

        return processed_frames, orig_height, orig_width, resized_height, resized_width

    def __getitem__(self, index):
        data = self._dataset[index]
        image_inputs, h, w, resize_h, resize_w = self._vision_preprocess(data)
        agent_pos = data[self._state_key_mapping["state"]]
        action = data[self._action_key_mapping["action"]]
        frame_index = data["frame_index"]
        instruction_info = {"instruction": data["task"]}
        generate_subtask_ratio = self.data_config.generate_subtask_ratio
        complete_text, generate_subtask = get_wallx_normal_text(
            instruction_info,
            self.dataload_config.get("action_horizon", 33) - 1,
            frame_index,
            self.data_config.priority_order,
            self._cam_key_mapping,
            generate_subtask_ratio=generate_subtask_ratio,
        )
        text = process_grounding_points(
            complete_text, h, w, resize_h, resize_w, self.data_config.model_type
        )
        result = {
            "image_inputs": image_inputs,
            "text": text,
            "action": action,
            "agent_pos": agent_pos,
            "frame_index": frame_index,
        }

        return result

    def __len__(self) -> int:
        return len(self._dataset)

    def _eval(self):
        self._dataset = self.val_dataset

    def _train(self):
        self._dataset = self.train_dataset

    def get_train_dataloader(self):
        """
        Get distributed training dataloader

        Args:
            rank: Current process rank
            world_size: Total number of processes
            seed: Random seed for reproducibility
        """
        self._train()

        batch_size = self.config.get("batch_size_per_gpu", 8)
        num_workers = self.config.get("num_workers", 4)

        # Create distributed sampler
        sampler = DistributedSampler(
            self,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=self.seed,
            drop_last=True,  # Ensure all processes have same number of batches
        )

        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,  # Use distributed sampler instead of shuffle=True
            num_workers=num_workers,
            collate_fn=DataCollator(
                self.config, self.dataload_config, self.norm_stats, self.lerobot_config
            ),
            pin_memory=True,  # Enable for GPU training
            persistent_workers=num_workers > 0,  # Only if num_workers > 0
            prefetch_factor=2,  # Reduce memory usage
            drop_last=True,  # Avoid incomplete batches
        )

        return dataloader, sampler

    def get_val_dataloader(self):
        """
        Get distributed evaluation dataloader (no shuffling for consistent evaluation)
        """
        self._eval()

        batch_size = self.config.get(
            "eval_batch_size_per_gpu", self.config.get("batch_size_per_gpu", 8)
        )
        num_workers = self.config.get("num_workers", 4)

        # Create distributed sampler for evaluation (no shuffle)
        sampler = DistributedSampler(
            self,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,  # No shuffling for evaluation
            drop_last=False,  # Keep all samples for evaluation
        )

        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=DataCollator(
                self.config, self.dataload_config, self.norm_stats, self.lerobot_config
            ),
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2,
            drop_last=False,
        )

        return dataloader, sampler


class DataCollator:
    # Class-level cache for processors to avoid reloading
    _processor_cache = {}
    _action_tokenizer_cache = {}

    def __init__(self, config, dataload_config, stats, lerobot_config):
        self.config = config
        self.dataload_config = dataload_config
        self.stats = stats
        self.action_min_stat = stats["action"].min
        self.action_delta = stats["action"].delta
        self.state_min_stat = stats["state"].min
        self.state_delta = stats["state"].delta
        self.lerobot_config = lerobot_config

        self.use_fast_tokenizer = self.config.get("use_fast_tokenizer", False)
        self.load_processor()

    def load_processor(self):
        processor_path = self.config["pretrained_wallx_path"]
        action_tokenizer_path = self.config["action_tokenizer_path"]

        if (
            self.use_fast_tokenizer
            and action_tokenizer_path not in self._action_tokenizer_cache
        ):
            self._action_tokenizer_cache[action_tokenizer_path] = (
                AutoProcessor.from_pretrained(
                    action_tokenizer_path, trust_remote_code=True
                )
            )

        # Use cached processors if available
        if processor_path not in self._processor_cache:
            processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)
            if self.config.get("padding_side", "left") == "left":
                processor.tokenizer.padding_side = "left"

            new_tokens = ["<|propri|>", "<|action|>"]
            processor.tokenizer.add_tokens(new_tokens)
            if self.use_fast_tokenizer and self.config.get("model_type") == "qwen2_5":
                action_tokenizer = self._action_tokenizer_cache[action_tokenizer_path]
                new_tokens = [
                    f"<|action_token_{i}|>" for i in range(action_tokenizer.vocab_size)
                ]
                processor.tokenizer.add_tokens(new_tokens)
                begin_idx_token = "<|action_token_0|>"
                token_id = processor.tokenizer.convert_tokens_to_ids(begin_idx_token)
                processor.tokenizer.init_kwargs["action_token_start_index"] = token_id
                processor.tokenizer.init_kwargs["action_token_vocab_size"] = (
                    action_tokenizer.vocab_size
                )

            self._processor_cache[processor_path] = processor

        self.processor = self._processor_cache[processor_path]

        if not self.use_fast_tokenizer:
            self.train_action_tokenizer = None
        else:
            self.train_action_tokenizer = self._action_tokenizer_cache[
                action_tokenizer_path
            ]

    @classmethod
    def _normalize(cls, action, min_stat, delta):
        """
        Normalize action data using min-max normalization.
        """
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        x = (action - min_stat) / delta
        x = x * 2 - 1
        x = torch.clamp(x, -1, 1)
        return x

    def __call__(self, batch):
        additional_inputs = {}

        for key in batch[0].keys():
            if key == "agent_pos":
                agent_pos = torch.stack([item["agent_pos"] for item in batch])
                if agent_pos.dim() == 2:
                    agent_pos = agent_pos.unsqueeze(1)
                agent_pos_mask = (~torch.isnan(agent_pos)).float()
                agent_pos.nan_to_num_(nan=0.0)
                agent_pos = self._normalize(
                    agent_pos, self.state_min_stat, self.state_delta
                )
                if agent_pos.shape[-1] != 20:
                    agent_pos = torch.cat(
                        [
                            agent_pos,
                            torch.zeros(
                                agent_pos.shape[0],
                                agent_pos.shape[1],
                                20 - agent_pos.shape[-1],
                            ),
                        ],
                        dim=-1,
                    )
                    agent_pos_mask = torch.cat(
                        [
                            agent_pos_mask,
                            torch.zeros(
                                agent_pos_mask.shape[0],
                                agent_pos_mask.shape[1],
                                20 - agent_pos_mask.shape[-1],
                            ),
                        ],
                        dim=-1,
                    )
                additional_inputs["proprioception"] = agent_pos
                additional_inputs["agent_pos_mask"] = agent_pos_mask
            elif key == "action":
                action = torch.stack([item["action"] for item in batch])
                if action.dim() == 2:
                    action = action.unsqueeze(1)
                dof_mask = (~torch.isnan(action)).float()
                action.nan_to_num_(nan=0.0)
                action = self._normalize(
                    action, self.action_min_stat, self.action_delta
                )
                if action.shape[-1] != 20:
                    action = torch.cat(
                        [
                            action,
                            torch.zeros(
                                action.shape[0], action.shape[1], 20 - action.shape[-1]
                            ),
                        ],
                        dim=-1,
                    )
                    dof_mask = torch.cat(
                        [
                            dof_mask,
                            torch.zeros(
                                dof_mask.shape[0],
                                dof_mask.shape[1],
                                20 - dof_mask.shape[-1],
                            ),
                        ],
                        dim=-1,
                    )
                additional_inputs["action_chunk"] = action
                additional_inputs["dof_mask"] = dof_mask
            elif key == "image_inputs":
                additional_inputs["image_inputs"] = [
                    item["image_inputs"] for item in batch
                ]
            elif key == "text":
                additional_inputs["text"] = [item["text"] for item in batch]
            elif key == "frame_index":
                additional_inputs["frame_index"] = torch.stack(
                    [item["frame_index"] for item in batch]
                )
            else:
                raise NotImplementedError(
                    f"{key} input not implemented in preprocesser"
                )

        additional_inputs["text"] = replace_action_token(
            additional_inputs["text"],
            additional_inputs["action_chunk"],
            self.train_action_tokenizer if self.use_fast_tokenizer else None,
            [self.lerobot_config["repo_id"]] * additional_inputs["text"].__len__(),
            additional_inputs["dof_mask"],
        )

        inputs = preprocesser_call(
            processor=self.processor,
            text=additional_inputs.pop("text"),
            images=additional_inputs.pop("image_inputs"),
            videos=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.dataload_config.get("max_length", 768),
        )

        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")

        # Gating token types
        additional_inputs["moe_token_types"] = inputs.input_ids == action_token_id

        inputs.update(additional_inputs)

        inputs["dataset_names"] = [self.lerobot_config["repo_id"]] * inputs[
            "action_chunk"
        ].shape[0]

        return inputs


def load_lerobot_data(
    config,
    lerobot_config,
    rank=0,
    world_size=1,
    seed=42,
):
    """
    Load LeRobot dataset with distributed support

    Args:
        config: Model configuration
        rank: Current process rank (default: 0)
        world_size: Total number of processes (default: 1)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        dataset: Training dataset
        train_num: Number of training samples per process
        sampler: Distributed sampler (None if world_size=1)
    """

    # Set seed for reproducibility
    torch.manual_seed(seed)

    dataload_config = get_data_configs(config["data"])

    repo_id = lerobot_config.get("repo_id", None)
    assert repo_id is not None, "repo id is required"
    root = lerobot_config.get("root", None)
    meta_info = LeRobotDatasetMetadata(repo_id, root=root)
    dataset_fps = meta_info.fps
    episodes_num = meta_info.total_episodes

    norm_stats_path = config.get("norm_stats_path", None)
    assert (
        norm_stats_path is not None
    ), "norm stats is required, please refer to 'wall-x/scripts/compute_norm_stats.py' to compute stats"
    norm_stats = load_norm_stats(norm_stats_path, repo_id)

    delta_timestamps = {
        # action chunk
        KEY_MAPPINGS[repo_id]["action"]: [
            t / dataset_fps
            for t in range(dataload_config.get("action_horizon", 33) - 1)
        ],
    }
    batch_size = config.get("batch_size_per_gpu", 8)
    split_seed = config.get("data", {}).get("split_seed", seed)
    train_test_split = dataload_config.get("train_test_split", 0.95)
    train_episodes, test_episodes = split_lerobot_episodes(
        episodes_num, train_test_split=train_test_split, split_seed=split_seed
    )
    image_transforms = build_image_transforms(lerobot_config.get("image_transforms"))

    train_dataset = FixedLeRobotDataset(
        repo_id,
        root=root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        video_backend="pyav",
        tolerance_s=lerobot_config.get("tolerance_s", 0.0001),
    )

    if rank == 0:
        print(f"Selected train episodes: {train_dataset.episodes}")
        print(f"Number of train episodes selected: {train_dataset.num_episodes}")
        print(f"Number of train frames selected: {train_dataset.num_frames}")
        print(f"Selected test episodes: {test_episodes}")

    dataset = PreprocessedDataset(
        train_dataset,
        config,
        dataload_config,
        norm_stats,
        lerobot_config,
        seed=seed,
        rank=rank,
        world_size=world_size,
        train_episodes=train_episodes,
        test_episodes=test_episodes,
        split_seed=split_seed,
        train_test_split=train_test_split,
    )

    # Calculate samples per process
    if world_size > 1:
        # With DistributedSampler, each process gets approximately len(dataset) // world_size samples
        samples_per_process = len(dataset) // world_size
        train_num = samples_per_process // batch_size
    else:
        train_num = len(dataset) // batch_size

    if rank == 0:
        print("\n" + "=" * 50)
        print("LeRobot Data Loading Configuration:")
        print(f"✦ RANK: {rank}")
        print(f"✦ WORLD SIZE: {world_size}")
        print(f"✦ BATCH SIZE PER GPU: {batch_size}")
        print(f"✦ REPO ID: {repo_id}")
        print(f"✦ TOTAL DATASET SIZE: {len(dataset)}")
        if world_size > 1:
            print(f"✦ SAMPLES PER PROCESS: {samples_per_process}")
            print(f"✦ BATCHES PER PROCESS: {train_num}")
            print(f"✦ TOTAL BATCHES (ALL PROCESSES): {train_num * world_size}")
        else:
            print(f"✦ TOTAL BATCHES: {train_num}")
        print(f"✦ SEED: {seed}")
        print("=" * 50 + "\n")

    return dataset, train_num


def get_distributed_dataloader(
    dataset, config, rank=0, world_size=1, seed=42, is_train=True
):
    """
    Helper function to get distributed dataloader

    Args:
        dataset: PreprocessedDataset instance
        config: Configuration dict
        rank: Current process rank
        world_size: Total number of processes
        seed: Random seed
        is_train: Whether this is for training (affects shuffling)

    Returns:
        dataloader: Distributed DataLoader
        sampler: DistributedSampler
    """
    if is_train:
        return dataset.get_train_dataloader(rank=rank, world_size=world_size, seed=seed)
    else:
        return dataset.get_val_dataloader(rank=rank, world_size=world_size)


def get_data_configs(config):
    default_data_config = {
        "train_test_split": 0.95,
        "split_seed": 42,
        "batch_size": 8,
        "action_horizon": 21,
        "action_history_length": 0,
        "image_horizon": 1,
        "image_history_length": 0,
        "left_padding": False,
        "right_padding": False,
        "return_first_obs": False,
        "return_last_obs": False,
        "randomize_obs_after": None,
        "datasets": [],
        "labeled_pathes": [],
    }
    data_config = default_data_config | config
    data_config["action_horizon"] += 1

    return data_config


class FixedLeRobotDataset(LeRobotDataset):
    """Fixed version of LeRobotDataset that correctly handles delta_timestamps
    when loading specific episodes (not episode 0).

    The original LeRobotDataset.__getitem__ uses relative index `idx` instead of
    absolute index when calculating query_indices in _get_query_indices, which
    causes incorrect action chunk retrieval for episodes other than episode 0.
    """

    def __getitem__(self, idx) -> dict:
        # Ensure dataset is loaded when we actually need to read from it
        self._ensure_hf_dataset_loaded()
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        # FIX: Use absolute index instead of relative index for delta calculation
        abs_idx = item["index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks.iloc[task_idx].name
        return item


class TestDataset(PreprocessedDataset):
    def __init__(
        self, dataset, config, dataload_config, norm_stats, lerobot_config, seed=42
    ):
        super().__init__(
            dataset,
            config,
            dataload_config,
            norm_stats,
            lerobot_config,
            seed=seed,
            rank=0,
            world_size=1,
            test_only=True,
        )

    def get_dataloader(self):
        """
        Get distributed evaluation dataloader (no shuffling for consistent evaluation)
        """

        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=1,
            collate_fn=DataCollator(
                self.config, self.dataload_config, self.norm_stats, self.lerobot_config
            ),
        )

        return dataloader


def load_test_dataset(
    config,
    lerobot_config,
    seed=42,
    episode=0,
):
    """
    Load test dataset

    Args:
        config: Model configuration
        seed: Random seed for reproducibility (default: 42)

    Returns:
        dataset: Test dataset
    """

    # Set seed for reproducibility
    torch.manual_seed(seed)

    repo_id = lerobot_config.get("repo_id", None)
    assert repo_id is not None, "repo id is required"
    root = lerobot_config.get("root", None)
    tolerance_s = lerobot_config.get("tolerance_s", 0.0001)
    meta_info = LeRobotDatasetMetadata(repo_id, root=root)
    dataset_fps = meta_info.fps
    dataload_config = get_data_configs(config["data"])

    norm_stats_path = config.get("norm_stats_path", None)
    assert (
        norm_stats_path is not None
    ), "norm stats is required, please refer to 'wall-x/scripts/compute_norm_stats.py' to compute stats"
    norm_stats = load_norm_stats(norm_stats_path, repo_id)

    delta_timestamps = {
        # action chunk
        KEY_MAPPINGS[repo_id]["action"]: [
            t / dataset_fps
            for t in range(dataload_config.get("action_horizon", 33) - 1)
        ],
    }

    dataset = FixedLeRobotDataset(
        repo_id,
        episodes=[episode],
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        root=root,
        tolerance_s=tolerance_s,
    )

    print(f"Selected episodes: {dataset.episodes}")
    print(f"Number of episodes selected: {dataset.num_episodes}")
    print(f"Number of frames selected: {dataset.num_frames}")

    dataset = TestDataset(
        dataset, config, dataload_config, norm_stats, lerobot_config, seed=seed
    )

    return dataset
