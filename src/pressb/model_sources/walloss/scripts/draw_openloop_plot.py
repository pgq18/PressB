import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.data.load_lerobot_dataset import load_test_dataset, get_data_configs


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["data"]["model_type"] = config.get("model_type")

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_horizon", type=int, default=32)
    parser.add_argument("--origin_action_dim", type=int, default=7)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--debug-tokens", action="store_true", help="Print token composition debug info")
    args = parser.parse_args()

    origin_action_dim = args.origin_action_dim
    pred_horizon = args.pred_horizon

    # get train config
    model_path = "/path/to/model"
    action_tokenizer_path = "/path/to/action/tokenizer"
    save_dir = "/path/to/save/dir"
    path = "/path/to/train/config"
    config = load_config(path)

    # load model with customized robot config
    model = Qwen2_5_VLMoEForAction.from_pretrained(
        model_path, train_config=config, action_tokenizer_path=action_tokenizer_path
    )
    model.eval()
    model = model.to("cuda")
    model = model.bfloat16()
    model.debug_token_info = args.debug_tokens

    # get test dataloader
    dataload_config = get_data_configs(config["data"])
    lerobot_config = dataload_config.get("lerobot_config", {})
    dataset = load_test_dataset(config, lerobot_config, seed=42, episode=args.episode)
    dataloader = dataset.get_dataloader()

    total_frames = len(dataloader)

    predict_mode = "fast" if config.get("use_fast_tokenizer", False) else "diffusion"
    action_dim = 20 if predict_mode == "diffusion" else origin_action_dim
    gt_traj = torch.full((total_frames, origin_action_dim), float("nan"))
    pred_traj = torch.full((total_frames, origin_action_dim), float("nan"))

    # use tqdm to show the progress
    for idx, batch in tqdm(
        enumerate(dataloader), total=total_frames, desc="predicting"
    ):
        if idx % pred_horizon == 0 and idx + pred_horizon <= total_frames:
            batch = batch.to("cuda")
            with torch.no_grad():
                outputs = model(
                    **batch,
                    action_dim=action_dim,
                    pred_horizon=pred_horizon,
                    mode="predict",
                    predict_mode=predict_mode,
                )
                pred_traj[idx : idx + pred_horizon] = (
                    outputs["predict_action"][:, :, :origin_action_dim]
                    .detach()
                    .cpu()
                    .squeeze(0)
                )

            # Denormalize ground truth actions
            gt_action_chunk = batch["action_chunk"][:, :, :origin_action_dim]
            dof_mask = batch["dof_mask"].to(gt_action_chunk.dtype)
            denormalized_gt = (
                model.action_preprocessor.normalizer_action.unnormalize_data(
                    gt_action_chunk,
                    [lerobot_config.get("repo_id", "physical-intelligence/libero")],
                    dof_mask,
                ).squeeze(0)
            )
            gt_traj[idx : idx + pred_horizon] = denormalized_gt.detach().cpu()

    gt_traj_np = gt_traj.numpy()
    pred_traj_np = pred_traj.numpy()

    timesteps = gt_traj.shape[0]
    valid_gt = ~np.isnan(gt_traj_np).any(axis=1)
    valid_pred = ~np.isnan(pred_traj_np).any(axis=1)

    print(f"Episode: {args.episode}")
    print(f"Total frames: {total_frames}")
    print(f"Plotted GT frames: {valid_gt.sum()}/{total_frames}")
    print(f"Plotted prediction frames: {valid_pred.sum()}/{total_frames}")
    for name, traj, valid in (
        ("GT", gt_traj_np, valid_gt),
        ("Prediction", pred_traj_np, valid_pred),
    ):
        if not valid.any():
            print(f"{name}: no valid frames")
            continue
        valid_traj = traj[valid]
        traj_min = np.nanmin(valid_traj, axis=0)
        traj_max = np.nanmax(valid_traj, axis=0)
        traj_ptp = traj_max - traj_min
        print(f"{name} min: {np.round(traj_min, 4).tolist()}")
        print(f"{name} max: {np.round(traj_max, 4).tolist()}")
        print(f"{name} ptp: {np.round(traj_ptp, 4).tolist()}")

    fig, axs = plt.subplots(
        origin_action_dim, 1, figsize=(15, 5 * origin_action_dim), sharex=True
    )
    fig.suptitle("Action Comparison for lerobot", fontsize=16)

    for i in range(origin_action_dim):
        axs[i].plot(range(timesteps), gt_traj_np[:, i], label="Ground Truth")
        axs[i].plot(range(timesteps), pred_traj_np[:, i], label="Prediction")
        axs[i].set_ylabel(f"Action Dim {i+1}")
        axs[i].legend()
        axs[i].grid(True)

    axs[-1].set_xlabel("Timestep")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"lerobot_comparison_{args.episode}.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()
