import pathlib
from PIL import Image
import numpy as np


def get_EFM_factor(seg_last_path, seg_next_path, seg_type_size: int):
    seg_last = Image.open(str(seg_last_path)).convert("RGB")
    seg_next = Image.open(str(seg_next_path)).convert("RGB")
    ntype = min(len(seg_last.getcolors()), len(seg_next.getcolors()))
    assert ntype <= seg_type_size
    seg_last = np.array(seg_last)
    seg_next = np.array(seg_next)
    flow = np.any(seg_last - seg_next, axis=-1)
    seg_type_size, ntype = seg_type_size - 1, ntype - 1 # remove background
    seg_type_size = seg_type_size - 1 # remove blur
    ntype = min(ntype, seg_type_size)
    f = ntype / seg_type_size * (1 - np.mean(flow))
    # print(f, ntype, np.mean(flow))
    return f, ntype


if __name__ == "__main__":
    render_seq = ["background", "blur", "desk", "knife", "blind", "person", "dog", "toaster", "bread", "toy"]
    seg_type_size = len(render_seq)
    exp = "flame_salmon_1"
    iter = 20000
    seg_dir = pathlib.Path(f"render/tgt/{exp}_{iter}_tracking/")
    assert seg_dir.exists() and seg_dir.is_dir()
    seg_paths = sorted(seg_dir.glob("*.png"))
    factors = []
    for seg_last_path, seg_next_path in zip(seg_paths[:-1], seg_paths[1:]):
        f, _ = get_EFM_factor(seg_last_path, seg_next_path, seg_type_size)
        factors.append(f)
    # print(factors)
    print(exp, iter, np.mean(factors))