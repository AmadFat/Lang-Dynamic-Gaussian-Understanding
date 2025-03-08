import gc
import cv2
import torch
import shutil
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from sam2.sam2_video_predictor import SAM2VideoPredictor
from grounding_dino.groundingdino.util.inference import load_model, predict
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from torchvision import transforms as T
from pprint import pprint
from collections import defaultdict

if torch.cuda.is_available():
    # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
torch.inference_mode().__enter__()

# GDINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
# GDINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
GDINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
# SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
# SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_tiny.pt"
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
# RESOLUTION = (360, 360) # DNeRF
RESOLUTION = (360, 360) # DyNeRF

# # lego
# EXP = "lego_50"
# BOX_THRESHOLD = 0.3
# MASK_THRESHOLD = 0.3
# TEXT_PROMPT = "redlight. bulldozer. floor. wheel. blur."
# RENDER_SEQ = ["background", "blur", "bulldozer", "floor", "wheel", "redlight"]
# SEED = 18

# # lego_ext
# EXP = "lego_ext_20"
# BOX_THRESHOLD = 0.3
# MASK_THRESHOLD = 0.3
# TEXT_PROMPT = "blur. bulldozer. floor."
# RENDER_SEQ = ["background", "blur", "bulldozer", "floor"]
# SEED = 18

# # jumpingjacks
# EXP = "jumpingjacks_20000"
# BOX_THRESHOLD = 0.3
# MASK_THRESHOLD = 0.2
# TEXT_PROMPT = "hand. head. jacket. shorts. leg. shoe. hair. blur."
# RENDER_SEQ = ["background", "blur", "jacket", "shorts", "leg", "hand", "shoe", "head", "hair"]
# SEED = 3415

# # bouncingballs
# EXP = "bouncingballs_50"
# BOX_THRESHOLD = 0.25
# MASK_THRESHOLD = 0.25
# TEXT_PROMPT = "redball. blueball. greenball. blur. whiteplate."
# RENDER_SEQ = ["background", "blur", "redball", "blueball", "greenball", "whiteplate"]
# SEED = 3422

# # cut_roasted_beef
# EXP = "cut_roasted_beef_20000"
# BOX_THRESHOLD = 0.2
# MASK_THRESHOLD = 0.4
# TEXT_PROMPT = "person. dog. bread. blind. desk. blur. toaster. knife. toy."
# RENDER_SEQ = ["background", "blur", "desk", "knife", "blind", "person", "dog", "toaster", "bread", "toy"]
# SEED = 3438

# flame_salmon_1
EXP = "flame_salmon_1_5"
BOX_THRESHOLD = 0.175
MASK_THRESHOLD = 0.35
TEXT_PROMPT = "window. person. stove. kettle. steak. desk. blur. plate."
RENDER_SEQ = ["background", "blur", "window", "person", "desk", "stove", "kettle", "plate", "steak"]
SEED = 3447

VIDEO_PATH = f"render/src/{EXP}.mp4"
OUTPUT_VIDEO_PATH = f"render/tgt/{EXP}.mp4"
OUTPUT_VIDEO_FRAME_DIR = f"render/tgt/{EXP}_frame/"
OUTPUT_VIDEO_TRACKING_DIR = f"render/tgt/{EXP}_tracking/"
PROMPT_TYPE_FOR_VIDEO = "box"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pprint({
    "GDINO_CONFIG": GDINO_CONFIG,
    "GDINO_CHECKPOINT": GDINO_CHECKPOINT,
    "SAM2_CONFIG": SAM2_CONFIG,
    "SAM2_CHECKPOINT": SAM2_CHECKPOINT,
    "RESOLUTION": RESOLUTION,
    "BOX_THRESHOLD": BOX_THRESHOLD,
    "MASK_THRESHOLD": MASK_THRESHOLD,
    "VIDEO_PATH": VIDEO_PATH,
    "TEXT_PROMPT": TEXT_PROMPT,
    "RENDER_SEQ": RENDER_SEQ,
    "OUTPUT_VIDEO_PATH": OUTPUT_VIDEO_PATH,
    "OUTPUT_VIDEO_FRAME_DIR": OUTPUT_VIDEO_FRAME_DIR,
    "OUTPUT_VIDEO_TRACKING_DIR": OUTPUT_VIDEO_TRACKING_DIR,
    "PROMPT_TYPE_FOR_VIDEO": PROMPT_TYPE_FOR_VIDEO,
    "DEVICE": DEVICE,
    "SEED": SEED,
})

def load_image(image_path: str):
    transform = T.Compose([
        T.Resize(list(RESOLUTION), interpolation=T.InterpolationMode.NEAREST_EXACT),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    image_pt = transform(image_pil)
    return image_pil, image_np, image_pt

def build_color_dict(text_prompt: str):
    color_dict = {}
    for k in ['background'] + [s.replace(' ', '') for s in text_prompt.split('.') if s not in [" ", ""]]:
        color_dict[k] = np.random.randint(0, 256, (3,))
    pprint(color_dict)
    return color_dict

if __name__ == "__main__":
    np.random.seed(SEED)
    color_dict = build_color_dict(TEXT_PROMPT)
    gdino_model = load_model(GDINO_CONFIG, GDINO_CHECKPOINT, DEVICE)
    video_predictor: SAM2VideoPredictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)
    print(sv.VideoInfo.from_video_path(VIDEO_PATH))
    frame_generator = sv.get_video_frames_generator(VIDEO_PATH)
    video_frame_dir = Path(OUTPUT_VIDEO_FRAME_DIR)
    shutil.rmtree(OUTPUT_VIDEO_FRAME_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_VIDEO_TRACKING_DIR, ignore_errors=True)
    video_frame_dir.mkdir(exist_ok=True, parents=True)
    Path(OUTPUT_VIDEO_TRACKING_DIR).mkdir(exist_ok=True, parents=True)
    with sv.ImageSink(video_frame_dir, True, "{:05d}.jpg") as sink:
        for idx, frame in enumerate(tqdm(frame_generator, desc="Saving Video Frames")):
            if idx % 3 in [1, 2]: continue # only for dynerf
            frame = cv2.resize(frame, RESOLUTION, interpolation=cv2.INTER_NEAREST_EXACT)
            sink.save_image(frame)
    frame_paths = sorted(video_frame_dir.glob("*.jpg"))
    inference_state = video_predictor.init_state(str(video_frame_dir))
    init_frame_idx = 0
    image_pil, image_np, image_pt = load_image(str(frame_paths[init_frame_idx]))
    boxes, confidences, labels = predict(
        gdino_model, image_pt, TEXT_PROMPT,
        BOX_THRESHOLD, 0.0, remove_combined=True,
    )
    del gdino_model
    torch.cuda.empty_cache()
    gc.collect()
    h, w, _ = image_np.shape
    boxes = boxes * torch.as_tensor([w, h, w, h])
    boxes = box_convert(boxes, "cxcywh", "xyxy").numpy()
    confidences = confidences.tolist()
    if PROMPT_TYPE_FOR_VIDEO == "box":
        for object_id, box in enumerate(boxes, 1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state, init_frame_idx,
                object_id, box=box,
            )
    else:
        raise ValueError
    for frame_idx, obj_idxs, logits in tqdm(video_predictor.propagate_in_video(inference_state)):
        frame_path = frame_paths[frame_idx]
        segment = defaultdict()
        for obj_idx, logit in zip(obj_idxs, logits):
            label = labels[obj_idx - 1]
            mask = (logit.sigmoid() > MASK_THRESHOLD).cpu().numpy()[0]
            segment[label] = np.logical_or(segment.get(label, False), mask)
        foreground = np.logical_or.reduce(list(segment.values()))
        segment['background'] = ~foreground
        canvas = np.zeros_like(image_np)
        print(segment.keys())
        for label in RENDER_SEQ:
            if label in segment:
                mask = segment[label]
                canvas[mask] = color_dict[label]
        # cv2.imshow("Segmentation", canvas)
        cv2.imwrite(Path(OUTPUT_VIDEO_TRACKING_DIR) / frame_path.with_suffix(".png").name, canvas)
        torch.cuda.empty_cache()
        gc.collect()




# """
# Step 4: Propagate the video predictor to get the segmentation results for each frame
# """
# video_segments = {}  # video_segments contains the per-frame segmentation results
# for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
#     video_segments[out_frame_idx] = {
#         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#         for i, out_obj_id in enumerate(out_obj_ids)
#     }

# """
# Step 5: Visualize the segment results across the video and save them
# """

# if not os.path.exists(SAVE_TRACKING_RESULTS_DIR):
#     os.makedirs(SAVE_TRACKING_RESULTS_DIR)

# ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

# for frame_idx, segments in video_segments.items():
#     img = cv2.imread(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx]))
    
#     object_ids = list(segments.keys())
#     masks = list(segments.values())
#     masks = np.concatenate(masks, axis=0)
    
#     detections = sv.Detections(
#         xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
#         mask=masks, # (n, h, w)
#         class_id=np.array(object_ids, dtype=np.int32),
#     )
#     box_annotator = sv.BoxAnnotator()
#     annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
#     label_annotator = sv.LabelAnnotator()
#     annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
#     mask_annotator = sv.MaskAnnotator()
#     annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
#     cv2.imwrite(os.path.join(SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)


# """
# Step 6: Convert the annotated frames to video
# """

# create_video_from_images(SAVE_TRACKING_RESULTS_DIR, OUTPUT_VIDEO_PATH)