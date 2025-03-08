from grounding_dino.groundingdino.util.inference import (
    load_model as gdino_load_model,
    preprocess_caption,
)
from grounding_dino.groundingdino.util.utils import get_phrases_from_posmap
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision import transforms as T
from torchvision.ops import box_convert
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from pprint import pprint
import supervision as sv
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import bisect
import shutil
import torch
import json
import time
import cv2
import gc


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gdino_load_image(image_path: str):
    transform = T.Compose([
            T.ColorJitter(contrast=(1.3, 1.3), brightness=(1.1, 1.1)),
            T.Resize(480, interpolation=T.InterpolationMode.BILINEAR, antialias=False),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_source = Image.open(image_path).convert("RGB")
    image_transformed = transform(image_source)
    image = np.array(image_source)
    return image, image_transformed


def gdino_predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
):
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)
    outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold # if has sigmoid score > box_threshold, admit this query
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    if remove_combined: # take the biggest logit word
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
        
        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
    else: # allow multiple words
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]
    return boxes, logits.max(dim=1)[0], phrases


def flourishing_splitting(
        video_path: Path, output_dir: Path,
        shared_time_stamps: list, batch_size: int,
):
    """
    Flourishing Splitting:
    - Each split has only 1 shared initial frame (0.0 ~ 1.0)
    - The union of all splits is the whole video
    - Splitting with np.random
    """
    np.random.seed(SEED)
    src_frame_dir = output_dir / "src_frame"
    shutil.rmtree(src_frame_dir, ignore_errors=True)
    src_frame_dir.mkdir(exist_ok=True, parents=True)
    video_info = sv.VideoInfo.from_video_path(video_path)
    pprint(video_info)
    with sv.ImageSink(str(src_frame_dir), overwrite=True, image_name_pattern="{:05d}.jpg") as sink:
        frame_generator = sv.get_video_frames_generator(video_path, stride=1, start=0, end=None)
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            # resize
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)
            sink.save_image(frame)

    src_frame_paths = sorted(src_frame_dir.glob("*.jpg"), reverse=False)
    shared_frame_idxs = [round(t * (video_info.total_frames - 1)) for t in shared_time_stamps]
    shared_frame_paths = [src_frame_paths[i] for i in shared_frame_idxs]
    unshared_frame_paths = [p for p in src_frame_paths if p not in shared_frame_paths]
    # take 1 shared frame and (batch_size - 1) unshared frames
    splits = []
    unshared_batch_size = batch_size - 1
    for idx, shared_frame_path in enumerate(shared_frame_paths):
        unshared_slice = np.random.choice(unshared_frame_paths, unshared_batch_size, replace=False)
        paths_slice = [shared_frame_path] + unshared_slice.tolist()
        split_dir = src_frame_dir / f"{idx:02d}"
        split_dir.mkdir(exist_ok=True, parents=True)
        for frame_path in tqdm(paths_slice, desc=split_dir.name):
            shutil.copy(frame_path, split_dir / frame_path.name)
        splits.append({
            "dir": split_dir,
            "shared_frame_path": shared_frame_path,
        })
    shared_frame_paths = [p.name for p in shared_frame_paths]
    pprint(f"Number of splits: {len(shared_frame_paths)}. Seed: {SEED}.")
    pprint(f"Shared frame paths: {shared_frame_paths}")
    return splits


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-v', '--video-path', type=str, required=True)
    parser.add_argument('-t', '--text-prompt', type=str, required=True)
    parser.add_argument('-o', '--output-dir', type=str, required=True)
    parser.add_argument('--skip-video', action='store_true')
    parser.add_argument('--gdino-box-threshold', type=float, required=True)
    parser.add_argument('--gdino-text-threshold', type=float, default=0.0)
    parser.add_argument('--sam2-prompt-type', type=str, default="mask", choices=["point", "box", "mask"])
    parser.add_argument('--sam2-mask-threshold', type=float, default=0.0)
    parser.add_argument('-s', '--seed', type=int, required=True)
    parser.add_argument('-shared', '--shared-time-stamps', nargs='+', type=float)
    parser.add_argument('-b', '--batch-size', type=int, required=True)
    args = parser.parse_args()

    args.gdino_config = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    args.gdino_checkpoint = "gdino_checkpoints/groundingdino_swint_ogc.pth"
    # args.gdino_config = "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    # args.gdino_checkpoint = "gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
    args.sam2_config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    args.sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
    print(args)
    return args


def predict_save_box(
        gdino_config, gdino_checkpoint, splits: dict,
        text_prompt: str, gdino_box_threshold: float, gdino_text_threshold: float, remove_combined: bool = False,
        visualize: bool = False,
):
    # Use Grounding DINO to get the box coordinates on the shared frame
    gdino_model = gdino_load_model(
        model_config_path=gdino_config,
        model_checkpoint_path=gdino_checkpoint,
        device=DEVICE,
    )
    for idx, s in enumerate(splits):
        shared_frame_path = s["shared_frame_path"]
        image_np, image_pt = gdino_load_image(shared_frame_path)
        boxes, confidences, labels = gdino_predict(
            model=gdino_model,
            image=image_pt,
            caption=text_prompt,
            box_threshold=gdino_box_threshold,
            text_threshold=gdino_text_threshold,
            remove_combined=remove_combined,
        )
        h, w, _ = image_np.shape
        boxes = box_convert(boxes * torch.as_tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy").long().tolist()
        confidences = confidences.tolist()
        boxes, confidences, labels = zip(*sorted(zip(boxes, confidences, labels), key=lambda x: x[1], reverse=True))
        pred = {
            "boxes": boxes,
            "confidences": confidences,
            "labels": labels,
        }
        splits[idx]["shared_frame_pred"] = pred
        splits[idx]["shared_frame_np"] = image_np
        with open(shared_frame_path.with_suffix(".json"), "w") as f:
            json.dump(pred, f, indent=4)
        if not visualize: continue
        for box, confidence, label in zip(boxes, confidences, labels):
            image_np_copy = image_np[:, :, ::-1].copy()
            x1, y1, x2, y2 = box
            text = f"{label}: {confidence:.2f}"
            cv2.rectangle(image_np_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np_copy, text, (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("image", image_np_copy)
            print(box, confidence, label)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    del gdino_model
    gc.collect()
    torch.cuda.empty_cache()
    return splits


def build_color_dict(text_prompt: str):
    color_dict_ks = ['background'] + [string.replace(' ', '') for string in text_prompt.split('.')]
    color_dict_vs = np.random.randint(0, 256, (len(color_dict_ks), 3))
    color_dict = dict(zip([k.lower() for k in color_dict_ks], color_dict_vs))
    return color_dict


if __name__ == '__main__':
    torch.inference_mode().__enter__()
    torch.no_grad().__enter__()
    args = parse_args()
    SEED = args.seed

    video_path = Path(args.video_path)
    text_prompt = args.text_prompt
    color_dict = build_color_dict(text_prompt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    skip_video = args.skip_video

    gdino_config = args.gdino_config
    gdino_checkpoint = args.gdino_checkpoint
    gdino_box_threshold = args.gdino_box_threshold
    gdino_text_threshold = args.gdino_text_threshold

    sam2_prompt_type = args.sam2_prompt_type
    sam2_config = args.sam2_config
    sam2_checkpoint = args.sam2_checkpoint
    sam2_mask_threshold = args.sam2_mask_threshold

    # Flourishing Splitting
    shared_time_stamps = args.shared_time_stamps
    batch_size = args.batch_size
    splits = flourishing_splitting(
        video_path=video_path, output_dir=output_dir,
        shared_time_stamps=shared_time_stamps, batch_size=batch_size,
    )

    splits = predict_save_box(
        gdino_config=gdino_config, gdino_checkpoint=gdino_checkpoint,
        text_prompt=text_prompt,
        gdino_box_threshold=gdino_box_threshold,
        gdino_text_threshold=gdino_text_threshold,
        splits=splits,
        visualize=False,
        remove_combined=True,
    )

    if torch.cuda.is_available():
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__() # for SAM
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    for split in splits:
        image_predictor = SAM2ImagePredictor(build_sam2(sam2_config, sam2_checkpoint))
        image_predictor.set_image(split["shared_frame_np"])
        src_frame_dir, pred = split["dir"], split["shared_frame_pred"]
        boxes, labels = np.asarray(pred["boxes"]), pred["labels"]
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=np.ones((len(boxes),), dtype=np.int32),
            box=boxes,
            multimask_output=False,
            return_logits=False,
        )
        masks = masks.squeeze(1)
        del image_predictor, boxes, scores, logits
        torch.cuda.empty_cache()
        gc.collect()

        # find the index of the shared frame in the src_frame_dir
        shared_frame_name = split["shared_frame_path"].name
        split_frame_names = [p.name for p in sorted(src_frame_dir.glob("*.jpg"))]
        ann_frame_idx = split_frame_names.index(shared_frame_name)

        video_predictor: SAM2VideoPredictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint)
        inference_state = video_predictor.init_state(str(src_frame_dir))
        if sam2_prompt_type == "mask":
            for object_id, (label, mask) in enumerate(zip(labels, masks), 1):
                _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    mask=mask
                )
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,
                                                                                              start_frame_idx=ann_frame_idx,
                                                                                              reverse=False):
            print(ann_frame_idx, out_frame_idx, split_frame_names[out_frame_idx])
            img = cv2.imread(str(src_frame_dir / split_frame_names[out_frame_idx]))
            masks = (out_mask_logits > sam2_mask_threshold).cpu().numpy().astype(np.bool)
            print(masks.shape, len(labels), type(labels))
            del out_frame_idx, out_obj_ids, out_mask_logits, masks
            torch.cuda.empty_cache()
            gc.collect()
        del video_predictor
        torch.cuda.empty_cache()
        gc.collect()
