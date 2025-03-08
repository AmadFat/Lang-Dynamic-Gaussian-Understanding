python flourishing_splitting.py \
    -v render/src/cut_roasted_beef_iter20000.mp4 \
    -t "man. dog. knife. bottle. cupboard. beef. table. window. painting. bowl. cup. fruit. book. toy." \
    -o render/tgt/cut_roasted_beef_iter20000 \
    --gdino-box-threshold 0.2 \
    --seed 150 --batch-size 8 \
    -shared 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0