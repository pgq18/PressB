from typing import Dict, List
import logging
import numpy as np
from wall_x.data.utils import preprocesser_call
from qwen_vl_utils.vision_process import smart_resize
import torch
from PIL import Image
from transformers import BatchFeature

logger = logging.getLogger(__name__)


def prepare_batch(
    obs: Dict | List[Dict],
    processor,
    camera_key: List[str],
    agent_pos_dim,
    action_dim,
    pred_horizon,
    fixed_action_dim,
    max_length,
    image_factor: int,
    min_pixels: int,
    max_pixels: int,
    predict_mode: str = "fast",
    device: str = "cuda",
    input_image_resolution: int | None = None,
) -> BatchFeature:
    """Prepare one observation or a true batch into model input format.

    Args:
        obs: Dictionary containing:
            - 'camera_key_0' : image 0
            - 'camera_key_1' : image 1
            ...
            - 'prompt': Text prompt
            - 'state': Robot state/proprioception
            - 'dataset_names': Dataset names

    Returns:
        BatchFeature object ready for model input
    """
    observations = [obs] if isinstance(obs, dict) else list(obs)
    if not observations:
        raise ValueError("At least one observation is required")
    if len(observations) > 1 and predict_mode != "diffusion":
        raise ValueError("Native batched inference requires diffusion mode")
    if processor.tokenizer.padding_side != "left":
        raise ValueError("Wall-X inference requires left tokenizer padding")

    image_batches, texts, states, dataset_names = [], [], [], []
    for observation in observations:
        images = [_as_pil_image(observation[key]) for key in camera_key]
        image_batches.append(process_images(
            images, image_factor, min_pixels, max_pixels, input_image_resolution,
        ))
        texts.append(format_text_with_vision_tokens(
            observation["prompt"], camera_key, predict_mode, pred_horizon,
        ))
        if "state" not in observation:
            raise ValueError("Wall-X observations require robot state")
        state = torch.as_tensor(observation["state"], dtype=torch.float32)
        if state.ndim == 1:
            state = state[None, None, :]
        elif state.ndim == 2:
            state = state[None, :, :]
        if state.ndim != 3 or state.shape[0] != 1:
            raise ValueError("Each observation state must describe exactly one sample")
        if state.shape[-1] > fixed_action_dim:
            raise ValueError("Robot state exceeds the model's fixed action width")
        state = torch.nn.functional.pad(state, (0, fixed_action_dim - state.shape[-1]))
        states.append(state)
        names = observation["dataset_names"]
        if isinstance(names, str):
            names = [names]
        if len(names) != 1 or not isinstance(names[0], str):
            raise ValueError("Each observation requires exactly one dataset name")
        dataset_names.append(names[0])

    # The image processor flattens nested camera lists in sample-major order;
    # preprocesser_call expands image placeholders in the same order. Both the
    # visual encoder and the flow transformer therefore run on the entire batch.
    inputs = preprocesser_call(
        processor=processor, text=texts, images=image_batches, videos=None,
        padding=True, truncation=True, return_tensors="pt", max_length=max_length,
    )
    action_token_id = processor.tokenizer.convert_tokens_to_ids("<|action|>")
    inputs["moe_token_types"] = inputs.input_ids == action_token_id
    if predict_mode == "diffusion" and not torch.all(
        inputs["moe_token_types"].sum(dim=1) == pred_horizon
    ):
        raise ValueError("Observation truncation removed action tokens; increase max_length")
    if len({tuple(state.shape[1:]) for state in states}) != 1:
        raise ValueError("Batched observations must have the same proprioception history length")
    state = torch.cat(states, dim=0)
    agent_pos_mask = torch.ones_like(state)
    agent_pos_mask[:, :, agent_pos_dim:] = 0
    inputs["proprioception"] = state
    inputs["agent_pos_mask"] = agent_pos_mask
    inputs["dataset_names"] = dataset_names
    dof_mask = torch.ones((len(observations), pred_horizon, fixed_action_dim))
    dof_mask[:, :, action_dim:] = 0
    inputs["dof_mask"] = dof_mask
    return BatchFeature(data=dict(inputs)).to(device)


def _as_pil_image(img):
    """Keep legacy image conversion identical for singleton and batch input."""
    if not isinstance(img, np.ndarray):
        return img
    if img.ndim > 3:
        img = np.squeeze(img)
    if img.ndim == 3:
        if img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[2] not in (1, 3):
            raise ValueError(f"Unexpected image shape: {img.shape}")
        if img.shape[-1] == 1:
            img = img[..., 0]
    elif img.ndim != 2:
        raise ValueError(f"Invalid image dimensions: {img.ndim}")
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def process_images(
    images: List[Image.Image],
    image_factor: int,
    min_pixels: int,
    max_pixels: int,
    input_image_resolution: int | None = None,
) -> List[Image.Image]:
    """Process images with optional pre-resize then smart resize.

    Args:
        images: List of PIL Images
        image_factor: Factor for smart resize (default 28)
        min_pixels: Minimum pixels for smart resize
        max_pixels: Maximum pixels for smart resize
        input_image_resolution: Optional resolution for the longer edge (None = no pre-resize).
            Images are resized proportionally to fit within this resolution.

    Returns:
        List of resized PIL Images
    """
    resized_images = []
    for img_pil in images:
        # Pre-resize proportionally based on the longer edge
        if input_image_resolution is not None:
            orig_width, orig_height = img_pil.size
            if orig_width > orig_height:  # Landscape
                new_width = input_image_resolution
                new_height = int(input_image_resolution * orig_height / orig_width)
            else:  # Portrait or square
                new_height = input_image_resolution
                new_width = int(input_image_resolution * orig_width / orig_height)
            img_pil = img_pil.resize((new_width, new_height))

        current_width, current_height = img_pil.size

        # Apply smart scaling (Qwen logic)
        resized_height, resized_width = smart_resize(
            current_height,
            current_width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        resized_img = img_pil.resize((resized_width, resized_height))
        resized_images.append(resized_img)

    return resized_images


def format_text_with_vision_tokens(
    instruction: str,
    camera_key: List[str],
    predict_mode: str = "fast",
    pred_horizon: int = 32,
) -> str:
    """Format text prompt with vision tokens for the model.

    Args:
        instruction: Task instruction text
        camera_key: List of camera names

    Returns:
        Formatted text with special tokens
    """
    # Special tokens for formatting
    role_start_symbol = "<|im_start|>"
    role_end_symbol = "<|im_end|>"
    vision_start_symbol = "<|vision_start|>"
    vision_end_symbol = "<|vision_end|>"
    image_pad_symbol = "<|image_pad|>"
    propri_symbol = "<|propri|>"
    action_symbol = "<|action|>"
    # action_fast_symbol = "<|action_fast|>"

    # Camera name mapping
    camera_name_mapping = {
        "front_view": "front view",
        "face_view": "front view",
        "left_wrist_view": "left wrist view",
        "right_wrist_view": "right wrist view",
        "top_view": "top view",
        "wall_view": "wall view",
    }

    # System prologue
    prologue = (
        f"{role_start_symbol}system\nYou are a helpful assistant.{role_end_symbol}\n"
    )

    # User request with observation
    user_request = f"{role_start_symbol}user\nObservation:"
    if camera_key:
        for cam_name in camera_key:
            view_name = camera_name_mapping.get(cam_name, cam_name)
            user_request += f" {view_name}: {vision_start_symbol}{image_pad_symbol}{vision_end_symbol}"
    user_request += "\nInstruction:"

    text_prompt = (
        f"\nPredict the next action in robot action.\nProprioception: {propri_symbol}\n"
    )
    user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"
    assistant_output = f"{role_start_symbol}assistant\n"
    if predict_mode == "diffusion":
        assistant_output += f"{action_symbol * pred_horizon}"
    complete_text = prologue + user_message + assistant_output

    return complete_text
