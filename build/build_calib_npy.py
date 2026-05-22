"""Build (N, 3, 640, 640) float32 calibration tensor for onnx2tf -oiqt."""
import sys, numpy as np
from PIL import Image
from pathlib import Path

LIST = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('project-1-at-2026-05-13-06-40-8e81e090/calib_subset_500.txt')
OUT  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('build/calib_500x3x640x640_float32.npy')
SIZE = 640

paths = [Path(p.strip()) for p in LIST.read_text().splitlines() if p.strip()]
N = len(paths)
print(f"loading {N} images -> {OUT}", flush=True)

# memmap to avoid 2.4 GB allocation
OUT.parent.mkdir(parents=True, exist_ok=True)
hdr_shape = (N, 3, SIZE, SIZE)
arr = np.lib.format.open_memmap(str(OUT), mode='w+', dtype=np.float32, shape=hdr_shape)

def letterbox(img: Image.Image, sz: int) -> np.ndarray:
    w, h = img.size
    s = min(sz / w, sz / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new('RGB', (sz, sz), (114, 114, 114))
    canvas.paste(img, ((sz - nw) // 2, (sz - nh) // 2))
    a = np.asarray(canvas, dtype=np.float32) / 255.0   # H,W,3
    return a.transpose(2, 0, 1)                         # 3,H,W

for i, p in enumerate(paths):
    with Image.open(p) as im:
        im = im.convert('RGB')
        arr[i] = letterbox(im, SIZE)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N}", flush=True)

arr.flush()
del arr
print("done")
