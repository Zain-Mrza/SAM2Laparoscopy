import os
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from sam2.build_sam import build_sam2_video_predictor

from definitions import ROOT_DIR


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