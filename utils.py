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

from definitions import ROOT_DIR

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


from sam3.model_builder import build_sam3_video_predictor
import cv2
import numpy as np
import os


def propagate_in_video(predictor, session_id):
    """
    Helper to stream propagation outputs over all frames in the session.
    Returns:
        outputs_per_frame: dict[int -> dict] mapping frame_index to SAM3 outputs.
    """
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        frame_idx = response["frame_index"]
        outputs_per_frame[frame_idx] = response["outputs"]
    return outputs_per_frame


def run_sam3_with_box_and_point(
    frames_dir="frames",
    out_dir="overlays",
    frame0_name="0.jpg",
    bbox=(372, 300, 600, 500),  # (x, y, width, height) in *pixels*
    point=(500, 700),           # (x, y) in *pixels*
    point_label=1,              # 1 = foreground, 0 = background
    obj_id=1,
    alpha=0.5,
):
    """
    Run SAM3 video segmentation with a bounding box + point prompt and
    save overlayed masks for all frames.

    Args:
        frames_dir (str): Folder containing per-frame JPEGs (0.jpg, 1.jpg, ...)
                          OR a video file path (for SAM3 resource_path).
                          For the overlay step we assume JPEGs in this folder.
        out_dir (str): Output folder to save overlay frames.
        frame0_name (str): Filename of the first frame (used for prompts & sizing).
        bbox (tuple): (x, y, width, height) of bounding box in pixels on frame 0.
        point (tuple): (x, y) of point prompt in pixels on frame 0.
        point_label (int): Point label, 1 for positive, 0 for negative.
        obj_id (int): Object ID used for prompts.
        alpha (float): Transparency for mask overlay (0–1).

    Returns:
        dict: outputs_per_frame mapping frame_index -> SAM3 output dict.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Build predictor
    video_predictor = build_sam3_video_predictor()

    session_id = None
    try:
        # Start a session (SAM3 expects resource_path pointing to frames or video)
        response = video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=frames_dir,
            )
        )
        session_id = response["session_id"]

        # Read frame 0 to get dimensions and normalize bbox
        frame0_path = os.path.join(frames_dir, frame0_name)
        img0 = cv2.imread(frame0_path)
        if img0 is None:
            raise FileNotFoundError(f"Could not read frame 0 at {frame0_path}")

        h, w = img0.shape[:2]
        x, y, bw, bh = bbox

        # Normalize bbox to [0,1] as SAM3 expects:
        bounding_boxes = [[
            x / w,
            y / h,
            bw / w,
            bh / h,
        ]]

        # 1) Add bounding box prompt on frame 0
        _ = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                bounding_boxes=bounding_boxes,
                bounding_box_labels=[1],  # 1 = foreground
                obj_id=obj_id,
            )
        )

        outputs_per_frame = propagate_in_video(video_predictor, session_id)
        
        # 2) Add point prompt on frame 0
        px, py = point
        _ = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                points=[[px, py]],
                point_labels=[point_label],
                obj_id=obj_id,
            )
        )

        # 3) Propagate masks through the entire video
        outputs_per_frame = propagate_in_video(video_predictor, session_id)

        # 4) Create overlay images for all frames
        for frame_idx, frame_output in outputs_per_frame.items():
            # Load the corresponding frame
            frame_path = os.path.join(frames_dir, f"{frame_idx}.jpg")
            img = cv2.imread(frame_path)

            if img is None:
                print(f"Warning: could not read {frame_path}, skipping")
                continue

            h, w = img.shape[:2]

            # Get masks: shape (num_objs, H, W) or (H, W)
            masks = frame_output["out_binary_masks"]  # bool array

            # Ensure (N, H, W)
            if masks.ndim == 2:
                masks = masks[None, ...]

            # Union of all object masks
            union_mask = np.any(masks, axis=0)  # (H, W) bool

            # Resize if needed
            if union_mask.shape != (h, w):
                union_mask = cv2.resize(
                    union_mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            # Green overlay layer (BGR)
            color_layer = np.zeros_like(img, dtype=np.uint8)
            color_layer[union_mask] = np.array([0, 255, 0], dtype=np.uint8)

            # Alpha blend
            overlay = cv2.addWeighted(img, 1 - alpha, color_layer, alpha, 0)

            # Save overlay frame
            out_path = os.path.join(out_dir, f"{frame_idx}.jpg")
            cv2.imwrite(out_path, overlay)
            print(f"Saved {out_path}")

        return outputs_per_frame

    finally:
        # Clean up session and shut down
        if session_id is not None:
            try:
                _ = video_predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=session_id,
                    )
                )
            except Exception as e:
                print(f"Warning: error closing session: {e}")

        try:
            video_predictor.shutdown()
        except Exception as e:
            print(f"Warning: error shutting down predictor: {e}")
