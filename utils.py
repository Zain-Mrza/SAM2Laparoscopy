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
    device: str = "cuda",
) -> str:
    """
    1) Loads SAM 2.1 (hiera large)
    2) Uses a single point on frame 0 to define an object
    3) Propagates the mask through all frames
    4) Overlays the masks on the frames
    5) Writes a video to `output_video_path`
    """

    # ---------------- SAM2 SETUP ----------------
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

    # single point and label
    points = np.array([[x, y]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)

    video_segments: dict[int, dict[int, np.ndarray]] = {}

    # ---------------- SAM2 INFERENCE ----------------
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(frames_path)

        # add the point on frame 0
        _frame_idx, _object_ids, _masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )

        # propagate through the video
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            state
        ):
            frame_dict: dict[int, np.ndarray] = {}
            for i, oid in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()  # logits -> bool
                frame_dict[int(oid)] = mask
            video_segments[int(out_frame_idx)] = frame_dict

    # ---------------- OVERLAY WITH SUPERVISION ----------------
    frames_dir = Path(frames_path)
    frame_paths = sorted(frames_dir.iterdir(), key=lambda p: int(p.stem))

    # video info from first frame
    first_frame = cv2.imread(str(frame_paths[0]))
    h, w = first_frame.shape[:2]

    video_info = sv.VideoInfo(
        width=w,
        height=h,
        fps=fps,
        total_frames=len(frame_paths),
    )

    mask_annotator = sv.MaskAnnotator(
        color=sv.Color.RED,
        opacity=0.5,
    )

    video_sink = sv.VideoSink(
        target_path=output_video_path,
        video_info=video_info,
    )

    with video_sink:
        for idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))

            # no mask for this frame
            if idx not in video_segments or obj_id not in video_segments[idx]:
                video_sink.write_frame(frame)
                continue

            # (H, W) bool mask
            mask = np.array(video_segments[idx][obj_id])
            mask = np.squeeze(mask).astype(bool)

            if not mask.any():
                video_sink.write_frame(frame)
                continue

            # (1, H, W) for supervision
            masks = mask[None, ...]

            # bounding box from mask
            ys, xs = np.where(mask)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            xyxy = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)

            detections = sv.Detections(
                xyxy=xyxy,
                mask=masks,
                class_id=np.array([0]),
            )

            annotated = mask_annotator.annotate(
                scene=frame,
                detections=detections,
            )

            video_sink.write_frame(annotated)

    return output_video_path
