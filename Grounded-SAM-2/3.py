from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget
from dds_cloudapi_sdk import TextPrompt

import os
import cv2
import torch
import numpy as np
import supervision as sv
import argparse
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from utils.track_utils import sample_points_from_masks
import gc

from sam2.sam2_image_predictor import SAM2Base

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def fill_pole(mask: np.ndarray, kernel_size: int = 1) -> np.ndarray:
    """
    Try to fill the pole in the mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(np.bool)
    return mask

def draw_mask(masks, object_labels, color_dict, output_path: Path, size: tuple):
    """
    Draw segmentation masks with specified colors and save to file.
    """
    w, h = size  # PIL returns (width, height)
    image = np.zeros((h, w, 3), dtype=np.uint8)
    foreground = np.zeros((h, w), dtype=np.bool)
    for m in masks:
        foreground = np.logical_or(foreground, m)
    for label, mask in [('background', ~foreground)] + list(zip(object_labels, masks)):
        if label.lower() in color_dict:
            image[mask] = color_dict[label.lower()]
    plt.imsave(output_path, image)

def main():
    args = parse_args()
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    
    # Initialize color dictionary
    color_dict_ks = ['background'] + args.text
    color_dict_vs = np.random.randint(0, 256, (len(args.text) + 1, 3))
    color_dict = dict(zip([k.lower() for k in color_dict_ks], color_dict_vs))
    
    # Convert text prompt to string format for DINOX
    text_prompt = ". ".join(args.text) + "."
    
    # Create output directories
    exp_name = Path(args.video_path).stem
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    src_frame_dir = output_dir / "src_frames"
    seg_frame_dir = output_dir / "seg_frames"
    src_frame_dir.mkdir(exist_ok=True)
    seg_frame_dir.mkdir(exist_ok=True)
    
    # Initialize SAM2 models
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    # init video predictor every splitting
    image_predictor: SAM2ImagePredictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint, apply_postprocessing=True), mask_threshold=0.1)
    
    # Process video frames in batches
    video_info = sv.VideoInfo.from_video_path(args.video_path)
    batch_size = video_info.total_frames // args.split
    start_frame_idxs = list(range(0, video_info.total_frames, batch_size))
    for split_idx, start_frame_idx in enumerate(start_frame_idxs):
        if start_frame_idx < args.checkpoint // batch_size * batch_size:
            continue
        frame_generator = sv.get_video_frames_generator(
            args.video_path,
            start=start_frame_idx,
            end=min(start_frame_idx + batch_size, video_info.total_frames),
        )
        # Save video frames
        src_frame_dir_split = src_frame_dir / f"{split_idx}"
        src_frame_dir_split.mkdir(exist_ok=True, parents=True)
        with sv.ImageSink(
            target_dir_path=src_frame_dir_split,
            overwrite=True,
            image_name_pattern="{:05d}.jpg",
        ) as sink:
            for frame in tqdm(frame_generator, desc="Saving Video Frames"):
                sink.save_image(frame)

        # Scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(src_frame_dir_split)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # Create a video predictor and init
        video_predictor: SAM2VideoPredictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, apply_postprocessing=False)
        print(src_frame_dir_split)
        inference_state = video_predictor.init_state(video_path=str(src_frame_dir_split))
        ann_frame_idx = 0
        img_path = src_frame_dir_split / frame_names[ann_frame_idx]
        client = Client(Config("197f07289a0e672a73da2d70c9e0296a"))
        image_url = client.upload_file(img_path)
        task = DinoxTask(
            image_url=image_url,
            prompts=[TextPrompt(text=text_prompt)],
            bbox_threshold=args.box_threshold,
            iou_threshold=args.iou_threshold,
            targets=[DetectionTarget.BBox],
        )
        client.run_task(task)
        result = task.result
        objects = result.objects

        input_boxes = []
        confidences = []
        class_names = []

        for obj in objects:
            input_boxes.append(obj.bbox)
            confidences.append(obj.score)
            class_names.append(obj.category)
        input_boxes = np.array(input_boxes)

        image_predictor.set_image(np.array(Image.open(img_path).convert("RGB")))
        print(f"OBJECTS: {class_names}")
        masks, _, _ = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        if args.prompt_type == "point":
            all_sample_points = sample_points_from_masks(masks=masks, num_points=args.num_points)
            for object_id, (label, points) in enumerate(zip(class_names, all_sample_points), start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                _, _, _ = video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )
        elif args.prompt_type == "box":
            for object_id, (label, box) in enumerate(zip(class_names, input_boxes), start=1):
                _, _, _ = video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )
        elif args.prompt_type == "mask":
            for object_id, (label, mask) in enumerate(zip(class_names, masks), start=1):
                labels = np.ones((1), dtype=np.int32)
                _, _, _ = video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    mask=mask
                )

        # Map object IDs to class names
        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(class_names, start=1)}

        # Propagate segmentation across video
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            if out_frame_idx + start_frame_idx < args.checkpoint // batch_size * batch_size:
                continue
            out_mask_results = [m.squeeze(0) for m in list((out_mask_logits > 0.0).detach().cpu().numpy())]
            output_path = seg_frame_dir / f"{(out_frame_idx + start_frame_idx):05d}.png"
            draw_mask(out_mask_results, [ID_TO_OBJECTS[obj_id] for obj_id in out_obj_ids], color_dict, output_path, (480, 360))
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Results saved to {seg_frame_dir}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=3407)
    parser.add_argument('-t', '--text', type=str, nargs='+', required=True)
    parser.add_argument('-bt', '--box-threshold', type=float, default=0.25)
    parser.add_argument('-it', '--iou-threshold', type=float, default=0.8)
    parser.add_argument('-v', '--video-path', type=str, required=True)
    parser.add_argument('-o', '--output-dir', type=str, default='./output')
    parser.add_argument('-pt', '--prompt-type', type=str, default='point', 
                      choices=['point', 'box', 'mask'])
    parser.add_argument('-np', '--num-points', type=int, default=10)
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('-c', '--checkpoint', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    main()