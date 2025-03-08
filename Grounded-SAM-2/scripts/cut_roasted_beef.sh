python 3.py --seed 21 --box-threshold 0.175 --iou-threshold 0.4 --prompt-type mask --num-points 10 --split 60 \
    --text man dog curtain wall knife bottle cupboard beef table window oven drawing toy book bowl cup ceil fruit \
    --video-path render/src/cut_roasted_beef_iter20000_360p.mp4 \
    --output-dir render/tgt/ \
    --checkpoint 287