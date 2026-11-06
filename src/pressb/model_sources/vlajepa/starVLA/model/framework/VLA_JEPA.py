# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoConfig, AutoVideoProcessor, AutoModel, AutoTokenizer, VJEPA2VideoProcessor

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY


def stack_padding_mask(examples, key, length, device):
    """Optional masks are batch-wide; do not silently treat missing rows as real."""
    present = [key in example for example in examples]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"{key} must be present for every example or omitted for all")
    mask = torch.as_tensor(np.stack([example[key] for example in examples]), device=device)
    if mask.dtype != torch.bool or mask.shape != (len(examples), length):
        raise ValueError(f"{key} must be bool [batch, {length}]")
    return mask


def masked_world_model_l1(prediction, target, video_is_pad=None, tubelet_size=2):
    """Map frame padding to future tubelets and then to their spatial tokens.

    The target skips encoder tubelet 0. A target tubelet is valid only if every
    contributing frame is real, since the encoder mixes its tubelet's frames.
    """
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("World-model prediction and target must share [batch, tokens, channels]")
    error = (prediction.float() - target.float()).abs()
    if video_is_pad is None:
        return error.mean()
    if (video_is_pad.dtype != torch.bool or video_is_pad.ndim != 2
            or video_is_pad.shape[0] != error.shape[0]
            or tubelet_size < 1 or video_is_pad.shape[1] % tubelet_size):
        raise ValueError("video_is_pad must be bool [batch, frames] divisible by tubelet_size")
    valid_steps = ~video_is_pad.to(error.device).reshape(error.shape[0], -1, tubelet_size).any(-1)
    valid_steps = valid_steps[:, 1:]
    if not valid_steps.shape[1] or error.shape[1] % valid_steps.shape[1]:
        raise ValueError("Future tubelets do not match the world-model target tokens")
    valid = valid_steps.repeat_interleave(error.shape[1] // valid_steps.shape[1], dim=1)
    denominator = (valid.sum() * error.shape[-1]).clamp_min(1)
    return error.masked_fill(~valid.unsqueeze(-1), 0).sum() / denominator


@FRAMEWORK_REGISTRY.register("VLA_JEPA")
class VLA_JEPA(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - DiT diffusion head for future action sequence modeling
      - JEPA world model for future frame prediction

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        embodied_action_token = self.config.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token
        )

        # TODO speical tokens

        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        encoder_config = self.config.framework.vj2_model
        if encoder_config.get("init_from_config", False):
            # The trainer must subsequently load every encoder tensor from the
            # verified VLA-JEPA checkpoint before using this opt-in path.
            self.vj_encoder = AutoModel.from_config(AutoConfig.from_pretrained(encoder_config.base_encoder))
        else:
            self.vj_encoder = AutoModel.from_pretrained(encoder_config.base_encoder)
        # This is a fixed target encoder, including dropout/eval state. Removing
        # it from autograd also avoids unused trainable parameters under DDP.
        self.vj_encoder.requires_grad_(False)
        self.vj_encoder.eval()
        self.vj_processor = AutoVideoProcessor.from_pretrained(self.config.framework.vj2_model.base_encoder)

        tubelet_size = self.vj_encoder.config.tubelet_size
        if (self.config.framework.vj2_model.num_frames % tubelet_size
                or self.config.framework.vj2_model.num_frames // tubelet_size < 2):
            raise ValueError("Video horizon must contain at least two complete encoder tubelets")
        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.config.framework.vj2_model.num_frames//tubelet_size,
            img_size=((self.vj_encoder.config.image_size, self.vj_encoder.config.image_size)),
            tubelet_size=1,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim=self.vj_encoder.config.hidden_size * 2, # multi view
            action_embed_dim=self.qwen_vl_interface.model.config.hidden_size,
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
        )
        self.replace_prompt = "".join(
            [each * self.config.framework.vj2_model.num_action_tokens_per_timestep for each in
             action_tokens[:self.config.framework.vj2_model.num_frames//tubelet_size - 1]]
        )

        self.embodied_replace_prompt = "".join([embodied_action_token * self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction])

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "vj_encoder"):
            self.vj_encoder.eval()
        return self

    def expand_tokenizer(self, 
                         tokenizer: AutoTokenizer,
                         special_action_token: str = "<|action_{}|>",
                         max_action_tokens: int = 32,
                         embodied_action_token: str = "<|embodied_action|>"):
        action_tokens, action_token_ids = [], []
        for i in range(0, max_action_tokens):
            action_token_i = special_action_token.format(i)
            action_tokens.append(action_token_i)
            if action_token_i not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([action_token_i], special_tokens=True)
                if added == 0:
                    logger.warning(f"Warning: 0 tokens added (they may already exist) action_token_i: {action_token_i}.")
            action_token_id = tokenizer.convert_tokens_to_ids(action_token_i)    
            action_token_ids.append(action_token_id)
        
        if embodied_action_token not in tokenizer.get_vocab():
            added = tokenizer.add_tokens([embodied_action_token], special_tokens=True)
            if added == 0:
                logger.warning(f"Warning: 0 tokens added (they may already exist) embodied_action_token: {embodied_action_token}.")
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            # 2) resize embeddings of vla
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ;tokenizer.vocab_size: {len(tokenizer)}")
        return action_tokens, action_token_ids, embodied_action_token_id

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  # [B, [PIL.Image]]
        batch_videos = [example["video"] for example in examples]  #  [B, V, T, H, W, 3]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"]for example in examples] if "action" in examples[0] else None # [B, horizon, config.action_dim]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        batch_videos = np.stack(batch_videos)  #  [B, V, T, H, W, 3]
        batch_videos = batch_videos.transpose(0,1,2,5,3,4)  # [B, V, T, 3, H, W]
        if batch_videos.shape[1] != 2 or batch_videos.shape[2] != self.config.framework.vj2_model.num_frames:
            raise ValueError("Expected two synchronized views with the configured video horizon")

        # Step 1: QWenVL input format
        if actions is not None:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt},
                prompt_template=self.config.datasets.vla_data.get("CoT_prompt", "")) 
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt},
                prompt_template=self.config.datasets.video_data.get("CoT_prompt", ""))
        
        action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        action_indices = action_indices.nonzero(as_tuple=True)

        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)
        
        device_type = qwen_inputs["input_ids"].device.type
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
        
            # Step 2: JEPA Encoder
            B, V, T, C, H, W = batch_videos.shape
            batch_videos = batch_videos.reshape(B*V, T, C, H, W)  # [B*V, T, C, H, W]
            input_videos = []
            for i in range(B*V):
                input_videos.append(self.vj_processor(
                    videos=batch_videos[i], return_tensors="pt", do_sample_frames=False
                )["pixel_values_videos"].to(self.vj_encoder.device))
            input_videos = torch.cat(input_videos, dim=0)  # [B*V, T, C, H, W]
            if input_videos.shape[:3] != (B * V, T, C):
                raise ValueError("Video processor changed the supplied frame timing or channel layout")
            video_is_pad = stack_padding_mask(examples, "video_is_pad", T, input_videos.device)
            self.vj_encoder.eval()
            with torch.no_grad():
                video_embeddings = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
                num_temporal_steps = T // self.vj_encoder.config.tubelet_size
                tokens_per_clip, embed_dim = video_embeddings.shape[1:]
                tokens_per_step = tokens_per_clip // num_temporal_steps
                video_embeddings = video_embeddings.reshape(B, V, num_temporal_steps, tokens_per_step, embed_dim)
                video_embeddings = video_embeddings.permute(0, 2, 3, 1, 4).contiguous().reshape(
                    B, num_temporal_steps * tokens_per_step, V * embed_dim
                )
            #print(video_embeddings.shape) # [B, T//tubelet_size * dim_per_frame, V*embed_dim]
        
            # Step 3: VJ Predictor
            T = T // self.vj_encoder.config.tubelet_size
            input_states = video_embeddings[:, :video_embeddings.shape[1] // T * (T-1),:]  # [B, (T-1)*dim_per_frame, V*embed_dim]
            gt_states = video_embeddings[:, video_embeddings.shape[1] // T:, :]
            predicted_states = self.vj_predictor(
                input_states,
                action_tokens
            )

            teacher_forcing_wm_loss = masked_world_model_l1(
                predicted_states,
                gt_states,
                video_is_pad,
                self.vj_encoder.config.tubelet_size,
            )
        
        if "action" not in examples[0]:
            return {"wm_loss": teacher_forcing_wm_loss}

        # Step 4: Action Expert Forward and Loss
        # CUDA autocast does not support float32. Explicitly disable it and
        # cast Qwen's bf16 features to the action expert's parameter dtype.
        with torch.autocast(last_hidden.device.type, enabled=False):

            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=self.action_model.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)
            action_is_pad = stack_padding_mask(examples, "action_is_pad", actions.shape[1], actions.device)
            if action_is_pad is not None:
                action_is_pad = action_is_pad[:, -(self.future_action_window_size + 1):]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            embodied_action_repeated = embodied_action_tokens.to(self.action_model.dtype).repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=self.action_model.dtype
                )
                if state.ndim == 2:
                    state = state.unsqueeze(1)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            if action_is_pad is None:
                action_loss = self.action_model(embodied_action_repeated, actions_target_repeated, state_repeated)
            else:
                action_loss = self.action_model(
                    embodied_action_repeated, actions_target_repeated, state_repeated,
                    action_is_pad=action_is_pad.repeat(repeated_diffusion_steps, 1))

        return {"action_loss": action_loss, "wm_loss": teacher_forcing_wm_loss * 0.1}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Encode images/instruction and sample the configured flow-matching head.
        Inputs and outputs use the same coordinates and units as training.
        This method does not normalize, clip, or access dataset statistics.

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            state: Optional [B, state_dim] or [B, state_tokens, state_dim].
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Legacy output key, shape
                    [B, action_horizon, action_dim]; native training units.
        """
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions,
            prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt})
        
        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        #embodied_action_indices = ~torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)

        device_type = qwen_inputs["input_ids"].device.type
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)

        state = torch.as_tensor(np.array(state), device=last_hidden.device, dtype=self.action_model.dtype) if state is not None else None
        if state is not None and state.ndim == 1 and B == 1:
            state = state.unsqueeze(0)
        # Step 4: Action Expert Forward and Loss
        with torch.autocast(last_hidden.device.type, enabled=False):
            pred_actions = self.action_model.predict_action(embodied_action_tokens.to(self.action_model.dtype), state)

        # The historical key names the network output, not an implicit stats
        # transform: Piper's xyz metres + row-major rot6 remain in native units.
        # No dataset normalization statistics are needed by this entry point.
        normalized_actions = pred_actions.float().detach().cpu().numpy()
        return {"normalized_actions": normalized_actions, "embodied_action_tokens": embodied_action_tokens.to(dtype=torch.float32).detach().cpu().numpy()}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
     
    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
