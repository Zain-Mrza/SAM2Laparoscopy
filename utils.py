# ---- SymPy compatibility patch for PyTorch / torchvision ----
import sympy as _sympy

# Some torch/torchvision code expects `sympy.printing` to exist as a module
# attribute, which is not exposed the same way in newer SymPy versions.
try:
    _ = _sympy.printing  # try to access it
except AttributeError:
    import sympy.printing as _sympy_printing
    _sympy.printing = _sympy_printing
# ---- End SymPy patch ----



import os
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from sam3.visualization_utils import show_box, show_mask, show_points
from definitions import ROOT_DIR

from sam3.model_builder import build_sam3_video_model
from sam3.model_builder import build_sam3_video_predictor
import numpy as np

def sam2_point_track_and_overlay(
    x: int = 1282,
    y: int = 358,
    obj_id: int = 1,
    frames_path: str | None = None,
    output_video_path: str = "overlay_supervision.mp4",
    fps: int = 30,
    alpha: float = 0.5,
    device: str = "cuda"
) -> str:
    
    """
    1) Loads SAM 2.1 hiera large
    2) Can only use a single point on frame 0 to define an object (only one object id)
    3) Propagates and keeps masks of the object through all frames
    4) Overlays the masks on the frames
    5) Writes a video
    """

    ############################## SETUP
    if frames_path is None:
        frames_path = os.path.join(ROOT_DIR, "frames")

    checkpoint = os.path.join(
        ROOT_DIR, "external/sam2/checkpoints/sam2.1_hiera_large.pt"
    )
    model_cfg = "configs/sam2.1/sam2.1_hiera_l"

    predictor = build_sam2_video_predictor(
        model_cfg,
        checkpoint,
        device=device,
    )

    # get the points and label it
    # might need to change later if I want to add multiple points
    points = np.array([[x, y]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)

    video_segments: dict[int, dict[int, np.ndarray]] = {}

    ############################## INFERENCE
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(frames_path)

        # add to predictor
        _frame_idx, _object_ids, _masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,                                     # Add at frame 0
            obj_id=obj_id,                                   # May need to expand functionality later
            points=points,
            labels=labels,
        )

        # propagate through the video
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):            
            frame_dict: dict[int, np.ndarray] = {}

            for i, oid in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()  # logits -> bool (for mask)
                frame_dict[int(oid)] = mask
                
            video_segments[int(out_frame_idx)] = frame_dict      # now we have all the masks for the video

    ####################### CV2 OVERLAY
    frames_dir = Path(frames_path)
    frame_paths = sorted(frames_dir.iterdir(), key=lambda p: int(p.stem))

    # Get shape from the first frame
    first_frame = cv2.imread(str(frame_paths[0]))
    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    color = np.array([0, 0, 255], dtype=np.uint8)  # BGR green

    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        # Get mask for this frame if it exists
        mask = video_segments.get(idx, {}).get(obj_id, None)
        if mask is None:
            writer.write(frame)
            continue

        mask = np.squeeze(np.array(mask)).astype(bool)  # (H, W) bool
        if not mask.any():
            writer.write(frame)
            continue

        overlay = frame.copy()
        overlay[mask] = color  # colorize masked pixels

        out_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0.0)
        writer.write(out_frame)

    writer.release()
    return output_video_path

from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image

# assume you already have:
# from sam3.model_builder import build_sam3_video_model
# from your_utils import get_numerically_sorted_frames

import re
import glob

def get_numerically_sorted_frames(frames_dir):
    """
    Returns a list of image file paths sorted by the integer found in the filename.
    Fixes the issue where frame_10.jpg comes before frame_2.jpg.
    """
    frame_files = glob.glob(os.path.join(frames_dir, "*.[jp][pn]g"))
    
    # Extract the number from the filename for sorting
    # e.g., 'path/to/frame_12.jpg' -> 12
    def extract_number(filepath):
        # Find all digit sequences, take the last one (usually the frame number)
        numbers = re.findall(r'\d+', os.path.basename(filepath))
        return int(numbers[-1]) if numbers else 0

    return sorted(frame_files, key=extract_number)

def run_sam3_segmentation_pipeline(
    frames_dir,
    input_point,
    output_video_path,
    alpha=0.3,
    fps=24,
    device=None,
):
    """
    Args:
        frames_dir (str or Path): Directory containing numbered frame images.
        input_point (tuple): (x, y) pixel location on frame 0.
        output_video_path (str or Path): Where to save the overlaid video.
        alpha (float): Opacity of the mask overlay.
        fps (int): Framerate of the original video.
    """
    # -------------------------
    # Device / model setup
    # -------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Initializing SAM 3 on {device}...")
    sam3_model = build_sam3_video_model()
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone

    # Init state
    inference_state = predictor.init_state(video_path=str(frames_dir))

    # Sort frames numerically
    frame_files = get_numerically_sorted_frames(frames_dir)
    # Ensure they are Path objects
    frame_paths = [Path(f) for f in frame_files]

    # Get image size from the first frame
    with Image.open(frame_paths[0]) as img:
        img_width, img_height = img.size

    # -------------------------
    # Add point prompt on frame 0
    # -------------------------
    ann_obj_id = 1
    points = np.array([input_point], dtype=np.float32)
    labels = np.array([1], np.int32)  # 1 = foreground

    rel_points = [[x / img_width, y / img_height] for x, y in points]

    predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=ann_obj_id,
        points=torch.tensor(rel_points, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.int32),
        clear_old_points=False,
    )

    # -------------------------
    # Propagate masks
    # -------------------------
    print("Propagating masks...")
    video_segments = {}  # frame_idx -> (H, W) bool mask

    for frame_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=len(frame_paths),
        reverse=False,
        propagate_preflight=True,
    ):
        if ann_obj_id in obj_ids:
            idx = obj_ids.index(ann_obj_id)
            # store a boolean mask per frame
            mask = (video_res_masks[idx] > 0.0).cpu().numpy().squeeze()
            video_segments[frame_idx] = mask

    # -------------------------
    # Create overlaid video
    # -------------------------
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")

    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

    # BGR color for overlay (this is red; change to [0,255,0] for green)
    color = np.array([0, 255, 0], dtype=np.uint8)

    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"Warning: could not read frame {frame_path}, skipping.")
            continue

        # Get mask for this frame if it exists
        mask = video_segments.get(idx, None)
        if mask is None:
            writer.write(frame)
            continue

        mask = np.asarray(mask).astype(bool)
        if not mask.any():
            writer.write(frame)
            continue

        overlay = frame.copy()
        overlay[mask] = color

        out_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0.0)
        writer.write(out_frame)

    writer.release()
    print(f"Overlay video written to: {output_video_path}")

    return output_video_path, video_segments

