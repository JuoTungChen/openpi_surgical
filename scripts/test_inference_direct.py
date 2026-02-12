#!/usr/bin/env python3
"""
Direct test client that loads videos from MP4 files and compares policy predictions
against ground truth actions from parquet files.

This version works with LeRobot v2 format (external MP4 videos).

Usage:
    # Start the policy server first in another terminal:
    uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_gr00t_local --policy.dir=./checkpoints/exp28_pi05_25k/
    
    # Then run this script:
    uv run scripts/test_inference_direct.py \
        --dataset_path /home/iulian/chole_ws/data/open_h_suturing \
        --num_samples 5 \
        --host 0.0.0.0 \
        --port 8000
"""

import dataclasses
import json
import logging
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import tyro

# Add openpi to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from openpi_client import websocket_client_policy as _websocket_client_policy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Dataset path (LeRobot v2 format with external videos)
    dataset_path: str = "/home/iulian/chole_ws/data/openh_suturebot"
    
    # Policy server connection
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    
    # Sampling parameters
    num_samples: int = 5
    seed: int = 42
    
    # Which video views to load
    video_views: list[str] | None = None  # If None, loads all available views
    
    # Video decoder backend: 'torchcodec', 'decord', or 'pyav'
    video_backend: str = "torchcodec"
    
    # Visualization parameters
    save_plots: bool = True
    output_dir: str = "./test_inference_plots"
    show_plots: bool = False


def load_video_frame(video_path: pathlib.Path, frame_idx: int, backend: str = "torchcodec") -> np.ndarray | None:
    """
    Load a single frame from an MP4 video file using various backends.
    
    Args:
        video_path: Path to the MP4 file
        frame_idx: Frame index to extract
        backend: Video decoder backend ('torchcodec', 'decord', or 'pyav')
        
    Returns:
        RGB image as numpy array [H, W, 3] uint8, or None if failed
    """
    try:
        if backend == "torchcodec":
            # Try torchcodec (best AV1 support)
            try:
                from torchcodec.decoders import VideoDecoder
                import torch
                
                # Create decoder and get specific frame
                decoder = VideoDecoder(str(video_path))
                
                # Get frame by index - torchcodec uses get_frame or get_frames_at
                if hasattr(decoder, 'get_frame'):
                    frame_data = decoder.get_frame(frame_idx)
                elif hasattr(decoder, 'get_frames_at'):
                    frame_data = decoder.get_frames_at(indices=[frame_idx])
                    if isinstance(frame_data, dict) and 'data' in frame_data:
                        frame = frame_data['data'][0]
                    else:
                        frame = frame_data[0]
                else:
                    # Fallback: decode all frames up to the target
                    for i in range(frame_idx + 1):
                        frame_data = decoder.get_next_frame()
                        if i == frame_idx:
                            frame = frame_data['data'] if isinstance(frame_data, dict) else frame_data
                            break
                
                # Handle dict vs tensor
                if isinstance(frame_data, dict):
                    frame = frame_data.get('data', frame_data.get('frame'))
                else:
                    frame = frame_data
                
                # Convert to numpy
                if hasattr(frame, 'numpy'):
                    frame = frame.numpy()
                elif isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                
                # torchcodec returns tensor in [C, H, W] format, convert to [H, W, C]
                if frame.ndim == 3 and frame.shape[0] == 3:
                    frame = np.transpose(frame, (1, 2, 0))
                
                # Ensure uint8
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                
                return frame
            except (ImportError, AttributeError) as e:
                logger.warning(f"torchcodec not available or API changed: {e}, falling back to pyav")
                backend = "pyav"
        
        if backend == "decord":
            # Try decord
            try:
                from decord import VideoReader, cpu
                vr = VideoReader(str(video_path), ctx=cpu(0))
                frame = vr[frame_idx].asnumpy()
                return frame
            except ImportError:
                logger.warning("decord not available, falling back to pyav")
                backend = "pyav"
        
        if backend == "pyav":
            # Try PyAV (av) - better codec support than OpenCV
            try:
                import av
                container = av.open(str(video_path))
                video_stream = container.streams.video[0]
                
                # PyAV approach: decode frames sequentially until we reach target
                target_frame = None
                frame_count = 0
                
                for packet in container.demux(video_stream):
                    for frame in packet.decode():
                        if frame_count == frame_idx:
                            target_frame = frame.to_ndarray(format='rgb24')
                            container.close()
                            return target_frame
                        frame_count += 1
                        
                        # Early exit if we passed the target
                        if frame_count > frame_idx:
                            break
                    
                    if frame_count > frame_idx:
                        break
                
                container.close()
                return target_frame
            except ImportError:
                logger.error("PyAV (av) not available. Please install: pip install av")
                return None
        
        return None
        
    except Exception as e:
        logger.warning(f"Failed to load frame {frame_idx} from {video_path} with {backend}: {e}")
        return None


def load_episode_data(
    dataset_path: pathlib.Path,
    episode_idx: int,
    frame_idx: int,
    video_views: list[str] | None = None,
    video_backend: str = "torchcodec",
    action_horizon: int = 1,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, str]:
    """
    Load data for a single frame from an episode using LeRobot dataset API.
    
    Args:
        action_horizon: Number of future actions to load (for action chunking)
    
    Returns:
        images: Dict mapping view names to RGB images [H, W, 3] uint8
        action: Ground truth action [action_horizon, action_dim] or [action_dim] float32
        state: Proprioceptive state [state_dim] float32
        prompt: Language instruction string
    """
    try:
        # Use LeRobot's dataset to properly read v2.1 format
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        
        # Load dataset (it will use local files if path is local)
        dataset = LeRobotDataset(str(dataset_path), tolerance_s=1)
        
        # Calculate absolute index from episode and frame
        # Find the start index of this episode
        episode_data_index = dataset.episode_data_index
        
        # Check format of episode_data_index
        if isinstance(episode_data_index, dict) and 'from' in episode_data_index and 'to' in episode_data_index:
            # Format: {'from': tensor([starts...]), 'to': tensor([ends...])}
            num_episodes = len(episode_data_index['from'])
            if episode_idx >= num_episodes or episode_idx < 0:
                raise ValueError(f"Episode {episode_idx} out of range (0-{num_episodes-1})")
            from_idx = int(episode_data_index['from'][episode_idx].item())
            abs_idx = from_idx + frame_idx
        elif isinstance(episode_data_index, dict):
            # Format: {episode_idx: tensor([from, to])}
            if episode_idx not in episode_data_index:
                raise ValueError(f"Episode {episode_idx} not found in dataset")
            episode_info = episode_data_index[episode_idx]
            # episode_info is a tensor [from, to]
            if hasattr(episode_info, 'tolist'):
                episode_info = episode_info.tolist()
            abs_idx = episode_info[0] + frame_idx
        else:
            # DataFrame format
            episode_col = 'episode_index' if 'episode_index' in episode_data_index.columns else 'episode_id'
            from_col = 'from' if 'from' in episode_data_index.columns else 'start'
            
            episode_starts = episode_data_index[episode_data_index[episode_col] == episode_idx]
            
            if len(episode_starts) == 0:
                raise ValueError(f"Episode {episode_idx} not found in dataset")
            
            abs_idx = int(episode_starts[from_col].iloc[0]) + frame_idx
        
        # Get the data point(s) - load action_horizon future actions
        data_point = dataset[abs_idx]
        
        # Extract actions for the horizon
        if action_horizon > 1:
            # Load multiple future actions
            # Get episode end index to avoid going past episode boundary
            if isinstance(episode_data_index, dict) and 'from' in episode_data_index and 'to' in episode_data_index:
                to_idx = int(episode_data_index['to'][episode_idx].item())
            elif isinstance(episode_data_index, dict):
                episode_info = episode_data_index[episode_idx]
                if hasattr(episode_info, 'tolist'):
                    episode_info = episode_info.tolist()
                to_idx = episode_info[1]
            else:
                to_idx = int(episode_starts[to_col].iloc[0])
            
            # Calculate how many future actions we can actually get
            max_future_actions = to_idx - abs_idx
            actual_horizon = min(action_horizon, max_future_actions)
            
            # Load actions for all timesteps
            actions_list = []
            for t in range(actual_horizon):
                idx = abs_idx + t
                if idx < len(dataset):
                    dp = dataset[idx]
                    if "action" in dp:
                        act = dp["action"]
                        if hasattr(act, 'numpy'):
                            act = act.numpy()
                        actions_list.append(np.array(act, dtype=np.float32))
            
            if actions_list:
                action = np.stack(actions_list, axis=0)  # (horizon, action_dim)
            else:
                action = np.array([], dtype=np.float32)
        else:
            # Single action
            if "action" in data_point:
                action = data_point["action"]
                if hasattr(action, 'numpy'):
                    action = action.numpy()
                action = np.array(action, dtype=np.float32)
            else:
                action = np.array([], dtype=np.float32)
        
        # Extract state (proprioceptive)
        if "observation.state" in data_point:
            state = data_point["observation.state"]
            if hasattr(state, 'numpy'):
                state = state.numpy()
            state = np.array(state, dtype=np.float32)
        elif "state" in data_point:
            state = data_point["state"]
            if hasattr(state, 'numpy'):
                state = state.numpy()
            state = np.array(state, dtype=np.float32)
        else:
            state = np.array([], dtype=np.float32)
        
        # Extract prompt
        if "language_instruction" in data_point:
            prompt = data_point["language_instruction"]
        elif "instruction.text" in data_point:
            prompt = data_point["instruction.text"]
        else:
            prompt = ""
        
        if isinstance(prompt, bytes):
            prompt = prompt.decode('utf-8')
        elif hasattr(prompt, 'item'):
            prompt = str(prompt.item())
        else:
            prompt = str(prompt)
        
        # Load video frames manually (LeRobot may not load them by default)
        with open(dataset_path / "meta" / "info.json") as f:
            info = json.load(f)
        
        video_path_pattern = info["video_path"]
        images = {}
        
        # Determine which views to load
        if video_views is None:
            video_views = [k for k in info["features"].keys() if info["features"][k]["dtype"] == "video"]
        
        chunk_size = info.get("chunks_size", 1000)
        # Ensure episode_idx is int for division
        episode_idx_int = int(episode_idx)
        chunk_idx = episode_idx_int // chunk_size
        
        for video_key in video_views:
            video_path = dataset_path / video_path_pattern.format(
                episode_chunk=chunk_idx,
                video_key=video_key,
                episode_index=episode_idx_int
            )
            
            if not video_path.exists():
                logger.warning(f"Video file not found: {video_path}")
                continue
            
            frame = load_video_frame(video_path, frame_idx, backend=video_backend)
            if frame is not None:
                # Map dataset keys to model-expected keys
                simple_key = video_key.replace("observation.images.", "")
                # Map to the keys expected by the model
                key_mapping = {
                    "endoscope.left": "base_0_rgb",
                    # "endoscope.right": "endoscope_right_0_rgb",  # Not used by model but keep
                    "wrist.left": "left_wrist_0_rgb",
                    "wrist.right": "right_wrist_0_rgb",
                }
                model_key = key_mapping.get(simple_key, simple_key)
                images[model_key] = frame
        
        return images, action, state, prompt
        
    except ImportError as e:
        logger.error(f"LeRobot not available: {e}")
        logger.error("Please install: pip install lerobot")
        raise
    except Exception as e:
        logger.error(f"Failed to load episode data: {e}")
        raise


def plot_actions_comparison(
    predicted_actions: np.ndarray,
    ground_truth_actions: np.ndarray,
    images: dict[str, np.ndarray],
    sample_idx: int,
    episode_idx: int,
    frame_idx: int,
    output_dir: pathlib.Path | None = None,
    show: bool = False,
):
    """Plot predicted vs ground truth actions with images and 3D trajectory."""
    # Handle 1D vs 2D actions
    # Determine if actions are (horizon, action_dim) or just (action_dim,)
    original_pred_shape = predicted_actions.shape
    original_gt_shape = ground_truth_actions.shape
    
    if predicted_actions.ndim == 1:
        predicted_actions = predicted_actions.reshape(1, -1)  # (action_dim,) -> (1, action_dim)
    
    if ground_truth_actions.ndim == 1:
        ground_truth_actions = ground_truth_actions.reshape(1, -1)  # (action_dim,) -> (1, action_dim)
    
    # Now both are 2D: (horizon, action_dim)
    pred_horizon, pred_action_dim = predicted_actions.shape
    gt_horizon, gt_action_dim = ground_truth_actions.shape
    
    # Match action dimensions
    action_dim = min(pred_action_dim, gt_action_dim)
    predicted_actions = predicted_actions[:, :action_dim]
    ground_truth_actions = ground_truth_actions[:, :action_dim]
    
    # For action chunking: if prediction has multiple timesteps but GT is single,
    # replicate GT across all timesteps for comparison
    if pred_horizon > 1 and gt_horizon == 1:
        # Replicate ground truth to match prediction horizon for plotting
        ground_truth_actions = np.repeat(ground_truth_actions, pred_horizon, axis=0)
        action_horizon = pred_horizon
    elif gt_horizon > 1 and pred_horizon == 1:
        # Prediction is single step but GT has multiple
        predicted_actions = np.repeat(predicted_actions, gt_horizon, axis=0)
        action_horizon = gt_horizon
    else:
        # Both same horizon or both single
        action_horizon = min(pred_horizon, gt_horizon)
        predicted_actions = predicted_actions[:action_horizon, :]
        ground_truth_actions = ground_truth_actions[:action_horizon, :]
    
    # Create main figure with multiple subplots
    # Layout: Top row for images, middle for 3D trajectory, bottom for action comparisons
    n_images = len(images)
    
    # Calculate grid layout
    action_cols = 4
    action_rows = (action_dim + action_cols - 1) // action_cols
    
    # Total layout: image row + 3D plot row + action rows
    total_rows = 1 + 1 + action_rows  # images + 3D + actions
    total_cols = max(n_images, action_cols)
    
    fig = plt.figure(figsize=(5 * total_cols, 4 * total_rows))
    fig.suptitle(f"Sample {sample_idx}: Episode {episode_idx}, Frame {frame_idx}", fontsize=16)
    
    # Create GridSpec for flexible layout
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(total_rows, total_cols, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot images in top row
    image_names = list(images.keys())
    for i, img_name in enumerate(image_names):
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(images[img_name])
        ax_img.set_title(img_name)
        ax_img.axis('off')
    
    # Plot 3D trajectory in second row (spanning multiple columns)
    ax_3d = fig.add_subplot(gs[1, :], projection='3d')
    
    # Extract x, y, z from first 3 action dimensions (absolute positions)
    if action_dim >= 3:
        # Ground truth trajectory
        gt_x = ground_truth_actions[:, 0]
        gt_y = ground_truth_actions[:, 1]
        gt_z = ground_truth_actions[:, 2]
        
        # Predicted trajectory
        pred_x = predicted_actions[:, 0]
        pred_y = predicted_actions[:, 1]
        pred_z = predicted_actions[:, 2]
        
        if action_horizon > 1:
            # Multiple timesteps - plot as trajectory
            ax_3d.plot(gt_x, gt_y, gt_z, 'b-o', label='Ground Truth', linewidth=2, markersize=6)
            ax_3d.plot(pred_x, pred_y, pred_z, 'r--x', label='Predicted', linewidth=2, markersize=6)
            
            # Mark start and end points
            ax_3d.scatter([gt_x[0]], [gt_y[0]], [gt_z[0]], c='blue', s=100, marker='s', label='GT Start')
            ax_3d.scatter([pred_x[0]], [pred_y[0]], [pred_z[0]], c='red', s=100, marker='s', label='Pred Start')
            
            ax_3d.set_title(f'3D Trajectory (First 3 Action Dims: X, Y, Z) - {action_horizon} timesteps')
        else:
            # Single timestep - plot as points
            ax_3d.scatter([gt_x[0]], [gt_y[0]], [gt_z[0]], c='blue', s=200, marker='o', label='Ground Truth', alpha=0.7)
            ax_3d.scatter([pred_x[0]], [pred_y[0]], [pred_z[0]], c='red', s=200, marker='x', label='Predicted', linewidths=3)
            
            # Draw vector from GT to prediction
            ax_3d.plot([gt_x[0], pred_x[0]], [gt_y[0], pred_y[0]], [gt_z[0], pred_z[0]], 
                      'k--', alpha=0.5, linewidth=1, label='Error')
            
            ax_3d.set_title('3D Position (First 3 Action Dims: X, Y, Z) - Single timestep')
        
        ax_3d.set_xlabel('X Position')
        ax_3d.set_ylabel('Y Position')
        ax_3d.set_zlabel('Z Position')
        ax_3d.legend()
        ax_3d.grid(True, alpha=0.3)
    else:
        ax_3d.text(0.5, 0.5, 0.5, 'Not enough dimensions for 3D plot\n(need at least 3)', 
                   ha='center', va='center', fontsize=12)
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
    
    # Plot action comparisons in remaining rows
    for dim in range(action_dim):
        row = 2 + dim // action_cols  # Start from row 2 (after images and 3D)
        col = dim % action_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Plot ground truth and predicted
        if action_horizon == 1:
            # Single point comparison - show as bar chart
            x = np.array([0])
            ax.bar(x - 0.2, ground_truth_actions[:, dim], width=0.4, label='Ground Truth', color='blue', alpha=0.7)
            ax.bar(x + 0.2, predicted_actions[:, dim], width=0.4, label='Predicted', color='red', alpha=0.7)
        else:
            # Multi-step comparison - show as line plot
            ax.plot(ground_truth_actions[:, dim], 'b-o', label='Ground Truth', linewidth=2, markersize=4)
            ax.plot(predicted_actions[:, dim], 'r--x', label='Predicted', linewidth=2, markersize=4)
        
        # Compute error
        mae = np.mean(np.abs(predicted_actions[:, dim] - ground_truth_actions[:, dim]))
        
        # Label based on dimension
        if dim == 0:
            dim_label = f'Dim {dim} - X (MAE: {mae:.4f})'
        elif dim == 1:
            dim_label = f'Dim {dim} - Y (MAE: {mae:.4f})'
        elif dim == 2:
            dim_label = f'Dim {dim} - Z (MAE: {mae:.4f})'
        else:
            dim_label = f'Dim {dim} (MAE: {mae:.4f})'
        
        ax.set_xlabel('Time Step' if action_horizon > 1 else '')
        ax.set_ylabel('Action Value')
        ax.set_title(dim_label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"sample_{sample_idx:03d}_ep{episode_idx}_fr{frame_idx}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {plot_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def compute_metrics(
    predicted_actions: np.ndarray,
    ground_truth_actions: np.ndarray,
) -> dict[str, float]:
    """Compute error metrics."""
    mae = np.mean(np.abs(predicted_actions - ground_truth_actions))
    mse = np.mean((predicted_actions - ground_truth_actions) ** 2)
    rmse = np.sqrt(mse)
    
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }


def main(args: Args):
    """Main testing function."""
    logger.info("Starting direct inference test (LeRobot v2 format)")
    logger.info(f"Arguments: {args}")
    
    # Set random seed
    np.random.seed(args.seed)
    
    dataset_path = pathlib.Path(args.dataset_path)
    
    # Load dataset info
    with open(dataset_path / "meta" / "info.json") as f:
        info = json.load(f)
    
    total_episodes = info.get("total_episodes", 0)
    logger.info(f"Dataset has {total_episodes} episodes")
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir) if args.save_plots else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
    # Connect to policy server
    logger.info(f"Connecting to policy server at {args.host}:{args.port}")
    client = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    
    # Sample random episodes
    episode_indices = np.random.choice(total_episodes, size=min(args.num_samples, total_episodes), replace=False)
    logger.info(f"Testing on episodes: {episode_indices}")
    
    # Load dataset once to check available episodes
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        dataset_lr = LeRobotDataset(str(dataset_path), tolerance_s=1)
        episode_data_index = dataset_lr.episode_data_index
        
        # Get list of available episodes
        if isinstance(episode_data_index, dict) and 'from' in episode_data_index and 'to' in episode_data_index:
            # Format: {'from': tensor([starts...]), 'to': tensor([ends...])}
            num_episodes = len(episode_data_index['from'])
            available_episodes = list(range(num_episodes))
        elif isinstance(episode_data_index, dict):
            # Format: {episode_idx: tensor([from, to])}
            available_episodes = [int(k) for k in episode_data_index.keys()]
        else:
            # DataFrame format
            episode_col = 'episode_index' if 'episode_index' in episode_data_index.columns else 'episode_id'
            available_episodes = [int(x) for x in episode_data_index[episode_col].unique().tolist()]
        
        logger.info(f"Available episodes in dataset: {sorted(available_episodes)}")
        
        # Resample from available episodes
        episode_indices = np.random.choice(available_episodes, size=min(args.num_samples, len(available_episodes)), replace=False)
        logger.info(f"Resampled to available episodes: {episode_indices}")
        
    except Exception as e:
        logger.warning(f"Could not check available episodes: {e}")
        logger.info(f"Will try with episodes 0 to {total_episodes-1}")
        # Fallback to sequential episodes
        episode_indices = np.arange(min(args.num_samples, total_episodes))
    
    # Collect metrics
    all_metrics = []
    
    # Test each sample
    for sample_idx, episode_idx in enumerate(episode_indices):
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing sample {sample_idx + 1}/{len(episode_indices)}")
        
        try:
            print(f"Loading dataset from {dataset_path}")
            # input("Press Enter to continue...")
            # Load LeRobot dataset to get episode info
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            dataset_lr = LeRobotDataset(str(dataset_path), tolerance_s=1)
            
            # Get episode length
            episode_data_index = dataset_lr.episode_data_index
            
            # Check format of episode_data_index
            if isinstance(episode_data_index, dict) and 'from' in episode_data_index and 'to' in episode_data_index:
                # Format: {'from': tensor([starts...]), 'to': tensor([ends...])}
                num_episodes = len(episode_data_index['from'])
                if episode_idx >= num_episodes or episode_idx < 0:
                    logger.warning(f"Episode {episode_idx} out of range (0-{num_episodes-1}), skipping")
                    continue
                from_idx = int(episode_data_index['from'][episode_idx].item())
                to_idx = int(episode_data_index['to'][episode_idx].item())
                episode_length = to_idx - from_idx
            elif isinstance(episode_data_index, dict):
                # Format: {episode_idx: tensor([from, to])}
                if episode_idx not in episode_data_index:
                    logger.warning(f"Episode {episode_idx} not found, skipping")
                    continue
                episode_info = episode_data_index[episode_idx]
                # episode_info is a tensor [from, to]
                if hasattr(episode_info, 'tolist'):
                    episode_info = episode_info.tolist()
                episode_length = episode_info[1] - episode_info[0]
            else:
                # DataFrame format
                episode_col = 'episode_index' if 'episode_index' in episode_data_index.columns else 'episode_id'
                from_col = 'from' if 'from' in episode_data_index.columns else 'start'
                to_col = 'to' if 'to' in episode_data_index.columns else 'end'
                
                episode_info = episode_data_index[episode_data_index[episode_col] == episode_idx]
                
                if len(episode_info) == 0:
                    logger.warning(f"Episode {episode_idx} not found, skipping")
                    continue
                
                episode_length = int(episode_info[to_col].iloc[0] - episode_info[from_col].iloc[0])
            
            if episode_length == 0:
                logger.warning(f"Episode {episode_idx} is empty, skipping")
                continue
            
            frame_idx = np.random.randint(0, episode_length)
            
            # Query policy first to get predicted action shape
            # Load single frame data for inference
            images_single, _, state, prompt = load_episode_data(
                dataset_path=dataset_path,
                episode_idx=episode_idx,
                frame_idx=frame_idx,
                video_views=args.video_views,
                video_backend=args.video_backend,
                action_horizon=1,
            )
            
            if not images_single:
                logger.error("No images loaded, skipping")
                continue
            
            # Query policy - the server expects "image" (singular) not "images"
            obs = {"image": images_single, "state": state, "prompt": prompt}
            result = client.infer(obs)
            
            # The server returns 'actions' (plural) not 'action' (singular)
            if "actions" not in result:
                logger.error(f"Policy response missing 'actions' key. Got keys: {list(result.keys())}")
                continue
            
            pred_action = result["actions"]
            if not isinstance(pred_action, np.ndarray):
                pred_action = np.array(pred_action)
            
            logger.info(f"Predicted action shape: {pred_action.shape}")
            
            # Now load ground truth with matching horizon
            if pred_action.ndim == 2:
                action_horizon = pred_action.shape[0]
            else:
                action_horizon = 1
            
            # Reload data with correct action horizon
            images, gt_action, state, prompt = load_episode_data(
                dataset_path=dataset_path,
                episode_idx=episode_idx,
                frame_idx=frame_idx,
                video_views=args.video_views,
                video_backend=args.video_backend,
                action_horizon=action_horizon,
            )
            
            logger.info(f"Loaded episode {episode_idx}, frame {frame_idx}")
            logger.info(f"  Images: {list(images.keys())}, shapes: {[img.shape for img in images.values()]}")
            logger.info(f"  Ground truth action shape: {gt_action.shape}")
            logger.info(f"  State shape: {state.shape}")
            logger.info(f"  Prompt: {prompt}")
            logger.info(f"  Action horizon: {action_horizon}")
            
            # Actions should now match in horizon
            if pred_action.shape != gt_action.shape:
                logger.warning(f"Shape mismatch after loading: predicted {pred_action.shape} vs ground truth {gt_action.shape}")
                # Adjust if needed
                min_horizon = min(pred_action.shape[0] if pred_action.ndim > 1 else 1, 
                                gt_action.shape[0] if gt_action.ndim > 1 else 1)
                min_dim = min(pred_action.shape[-1], gt_action.shape[-1])
                
                if pred_action.ndim == 2 and gt_action.ndim == 2:
                    pred_action = pred_action[:min_horizon, :min_dim]
                    gt_action = gt_action[:min_horizon, :min_dim]
                elif pred_action.ndim == 1:
                    pred_action = pred_action[:min_dim]
                    gt_action = gt_action[:min_dim] if gt_action.ndim == 1 else gt_action[0, :min_dim]
                elif gt_action.ndim == 1:
                    gt_action = gt_action[:min_dim]
                    pred_action = pred_action[:min_dim] if pred_action.ndim == 1 else pred_action[0, :min_dim]
                    
                logger.info(f"Adjusted to shape: pred={pred_action.shape}, gt={gt_action.shape}")
            
            # Compute metrics
            metrics = compute_metrics(pred_action, gt_action)
            all_metrics.append(metrics)
            
            logger.info(f"Metrics for sample {sample_idx}:")
            logger.info(f"  MAE:  {metrics['mae']:.6f}")
            logger.info(f"  MSE:  {metrics['mse']:.6f}")
            logger.info(f"  RMSE: {metrics['rmse']:.6f}")
            
            # Plot comparison
            plot_actions_comparison(
                predicted_actions=pred_action,
                ground_truth_actions=gt_action,
                images=images,
                sample_idx=sample_idx,
                episode_idx=episode_idx,
                frame_idx=frame_idx,
                output_dir=output_dir,
                show=args.show_plots,
            )
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}", exc_info=True)
            continue
    
    # Compute and display aggregate metrics
    if all_metrics:
        logger.info(f"\n{'='*80}")
        logger.info("Aggregate Metrics Across All Samples:")
        logger.info(f"{'='*80}")
        
        mean_mae = np.mean([m["mae"] for m in all_metrics])
        mean_mse = np.mean([m["mse"] for m in all_metrics])
        mean_rmse = np.mean([m["rmse"] for m in all_metrics])
        
        logger.info(f"Mean MAE:  {mean_mae:.6f}")
        logger.info(f"Mean MSE:  {mean_mse:.6f}")
        logger.info(f"Mean RMSE: {mean_rmse:.6f}")
        
        # Save summary
        if output_dir:
            summary_path = output_dir / "metrics_summary.txt"
            with open(summary_path, "w") as f:
                f.write("Direct Inference Test Results (LeRobot v2)\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Dataset: {args.dataset_path}\n")
                f.write(f"Number of samples: {len(all_metrics)}\n\n")
                f.write(f"Mean MAE:  {mean_mae:.6f}\n")
                f.write(f"Mean MSE:  {mean_mse:.6f}\n")
                f.write(f"Mean RMSE: {mean_rmse:.6f}\n")
            
            logger.info(f"\nSaved summary to {summary_path}")
    
    logger.info(f"\n{'='*80}")
    logger.info("Testing complete!")
    if output_dir:
        logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
