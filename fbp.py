from google.colab import drive
drive.mount('/content/drive')

import os
import re
import math
import time
import hashlib
import random
import shutil
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from skimage.transform import iradon
from scipy.ndimage import gaussian_filter1d

# ============================================================
# SETTINGS
# ============================================================

SEED = 123

# --- Drive paths ---
DRIVE_SIMULIX_DIR = Path("/content/drive/MyDrive/simulixteliko")
DRIVE_RESULTS_DIR = Path("/content/drive/MyDrive/simulixteliko_results")
DRIVE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Local paths ---
LOCAL_SIMULIX_DIR = Path("/content/simulix_local")
LOCAL_RESULTS_DIR = Path("/content/simulix_results_local")
LOCAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Copy data locally ---
COPY_DATA_TO_LOCAL = True
RECOPY_LOCAL_DATA = False

SEARCH_DIRS = [LOCAL_SIMULIX_DIR if COPY_DATA_TO_LOCAL else DRIVE_SIMULIX_DIR]

# ============================================================
# GEOMETRY
# ============================================================

SINOGRAM_PROJ_COUNT = 72
SINOGRAM_DET_COUNT = 128
SINOGRAM_THETA_MAX = 360.0

PROJ_COUNT_CANDIDATES = [72, 76, 80, 90, 128]
ANGLE_PAD_MULT = 8
DET_MULT = 16

# ============================================================
# IMAGE / TRAINING
# ============================================================

IMG_SIZE = 90

BATCH = 8
EPOCHS = 120
LR = 5e-5
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 0
PATIENCE = 15
MAX_DEPTH = 8
VAL_EVERY = 2

BASE_CH = 48
USE_AMP = True

FIXED_TRAIN = 2600
FIXED_VAL = 200
FIXED_TEST = 200

# ============================================================
# PLOT SETTINGS
# ============================================================

SINO_CMAP = "turbo"
IMG_CMAP = "turbo"
ERR_CMAP = "turbo"

# ============================================================
# FBP SETTINGS
# ============================================================

SINOGRAM_FORMAT = "line_integrals"

FBP_THETA_MAX = SINOGRAM_THETA_MAX
FBP_FLIP_DET = False
FBP_FLIP_ANG = True
FBP_CIRCLE = True
FBP_FILTER = "ramp"

# ============================================================
# MODEL INPUT / PATIENT-LIKE DEGRADATION
# ============================================================

INPUT_FBP_FILTER = "ramp"

USE_PATIENT_LIKE_DEGRADATION = True

MODEL_PROJ_COUNT = 72
MODEL_CANVAS_DET = 128
MODEL_ACTIVE_DET = 128

ACTIVE_DET_JITTER = 0
CENTER_JITTER = 0

DEG_BLUR_DET_SIGMA = (0.2, 0.8)
DEG_BLUR_ANG_SIGMA = (0.0, 0.15)
DEG_BACKGROUND_FRAC = (0.005, 0.025)
DEG_COUNT_SCALE = (20.0, 40.0)
DEG_GAUSS_STD = (0.0, 0.004)

TRAIN_INPUT_SEED_OFFSET = 10000

# ============================================================
# OUTPUT SMOOTHING
# ============================================================

APPLY_FINAL_SMOOTHING = True
FINAL_SMOOTH_KERNEL = 5
FINAL_SMOOTH_SIGMA = 1.0

# ============================================================
# LOCAL SSIM-LIKE LOSS SETTINGS (μόνο για training loss)
# ============================================================

LOSS_SSIM_WINDOW_SIZE = 11
LOSS_SSIM_WINDOW_SIGMA = 1.5
LOSS_SSIM_K1 = 0.01
LOSS_SSIM_K2 = 0.03

# ============================================================
# REPORTED METRIC SETTINGS
# ============================================================

# Αυτό είναι το metric που θα τυπώνεται σε val/test/patient comparison
# και που θα χρησιμοποιείται για τη σωστή τελική σύγκριση:
# full image, no sliding, C1=C2=0, max-normalized per image
REPORT_SSIM_K1 = 0.0
REPORT_SSIM_K2 = 0.0

# ============================================================
# LOSS WEIGHTS
# ============================================================

LAMBDA_WEIGHTED_L1 = 0.50
LAMBDA_SSIM = 0.15
LAMBDA_EDGE = 0.15
LAMBDA_CONTRAST = 0.20

# ============================================================
# CACHE
# ============================================================

CACHE_FBP_TO_DISK = True
REBUILD_FBP_CACHE = True

FBP_CACHE_DIR = LOCAL_RESULTS_DIR / "fbp_cache_simulix"
if REBUILD_FBP_CACHE:
    shutil.rmtree(FBP_CACHE_DIR, ignore_errors=True)
FBP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DEVICE / SEED
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

if device == "cuda":
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
else:
    scaler = torch.amp.GradScaler("cpu", enabled=False)

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?")

def autocast_context():
    if USE_AMP and device == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()

# ============================================================
# LOCAL COPY
# ============================================================

def copy_drive_to_local(src: Path, dst: Path, force=False):
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(f"Δεν βρέθηκε ο φάκελος: {src}")

    if dst.exists() and not force:
        print(f"Using existing local copy: {dst}")
        return

    if dst.exists() and force:
        shutil.rmtree(dst, ignore_errors=True)

    print(f"Copying data locally:\n  from {src}\n  to   {dst}")
    t0 = time.time()
    shutil.copytree(src, dst)
    print(f"Local copy completed in {time.time() - t0:.1f} sec")

if not DRIVE_SIMULIX_DIR.exists():
    raise FileNotFoundError(
        f"Δεν βρέθηκε ο φάκελος:\n{DRIVE_SIMULIX_DIR}\n"
        f"Έλεγξε ότι υπάρχει σωστά στο Drive."
    )

if COPY_DATA_TO_LOCAL:
    copy_drive_to_local(DRIVE_SIMULIX_DIR, LOCAL_SIMULIX_DIR, force=RECOPY_LOCAL_DATA)

SIMULIX_DIR = SEARCH_DIRS[0]
print("Using folder:", SIMULIX_DIR)

# ============================================================
# HELPERS
# ============================================================

def robust_01(x: np.ndarray, p1=1.0, p2=99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [p1, p2])
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)

def normalize_to_max_one(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    m = float(np.max(x))
    if m <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x / m).astype(np.float32)

def mae_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))

def rmse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))

def next_mult(x: int, m: int) -> int:
    return int(((x + m - 1) // m) * m)

def read_numbers(path: Path) -> np.ndarray:
    txt = path.read_text(errors="ignore")
    txt = txt.replace("D", "E").replace("d", "E").replace("...", " ")
    nums = _NUM_RE.findall(txt)
    if not nums:
        raise ValueError(f"No numbers found in {path}")
    return np.array([float(v) for v in nums], dtype=np.float32)

def infer_proj_count_from_flat_length(n, preferred=None, candidates=None, preferred_detectors=SINOGRAM_DET_COUNT):
    checked = []
    seen = set()

    ordered = []
    if preferred is not None:
        ordered.append(preferred)
    if candidates is not None:
        ordered.extend(candidates)

    for c in ordered:
        if c is None or c in seen or c <= 0:
            continue
        seen.add(c)

        if n % c == 0:
            det = n // c
            score = abs(det - preferred_detectors)
            if preferred is not None and c == preferred:
                score -= 1000
            checked.append((score, c, det))

    if not checked:
        return None

    checked.sort(key=lambda x: (x[0], x[1]))
    return checked[0][1]

def orient_sinogram_matrix(M, proj_count=None, candidates=None):
    M = np.asarray(M, dtype=np.float32)
    r, c = M.shape

    if proj_count is not None:
        if r == proj_count:
            return M.astype(np.float32)
        if c == proj_count:
            return M.T.astype(np.float32)

    cand_hits = []
    seen = set()
    if candidates is not None:
        for cc in candidates:
            if cc is None or cc in seen:
                continue
            seen.add(cc)
            if r == cc:
                cand_hits.append(("rows", cc))
            if c == cc:
                cand_hits.append(("cols", cc))

    if len(cand_hits) == 1:
        side, _ = cand_hits[0]
        return M.astype(np.float32) if side == "rows" else M.T.astype(np.float32)

    if r <= c:
        return M.astype(np.float32)
    return M.T.astype(np.float32)

def resize_sinogram_np(sino: np.ndarray, out_A=None, out_D=None) -> np.ndarray:
    sino = np.asarray(sino, dtype=np.float32)
    A, D = sino.shape
    if out_A is None:
        out_A = A
    if out_D is None:
        out_D = D
    if (A, D) == (out_A, out_D):
        return sino.astype(np.float32)

    t = torch.from_numpy(sino)[None, None].float()
    t = F.interpolate(t, size=(out_A, out_D), mode="bilinear", align_corners=False)
    return t[0, 0].cpu().numpy().astype(np.float32)

def resize_image_np(img: np.ndarray, out_h=IMG_SIZE, out_w=IMG_SIZE) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if img.shape == (out_h, out_w):
        return img.astype(np.float32)
    t = torch.from_numpy(img)[None, None].float()
    t = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return t[0, 0].cpu().numpy().astype(np.float32)

def stable_int_from_path(path: Path, offset=0) -> int:
    h = hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return offset + int(h, 16)

def save_matrix_txt(path: Path, arr: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(arr, dtype=np.float32), fmt="%.10f")

# ============================================================
# GAUSSIAN KERNEL CACHE
# ============================================================

_GAUSS_KERNEL_CACHE = {}

def get_gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype):
    key = (kernel_size, float(sigma), str(device), str(dtype))
    if key in _GAUSS_KERNEL_CACHE:
        return _GAUSS_KERNEL_CACHE[key]

    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    _GAUSS_KERNEL_CACHE[key] = kernel
    return kernel

def gaussian_blur_torch(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    if kernel_size <= 1:
        return x

    x = x.float()
    c = x.shape[1]
    kernel = get_gaussian_kernel2d(kernel_size, sigma, x.device, x.dtype)
    weight = kernel.expand(c, 1, kernel_size, kernel_size)
    pad = kernel_size // 2

    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y = F.conv2d(x_pad, weight, groups=c)
    return y

def finalize_prediction_torch(pred: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    if APPLY_FINAL_SMOOTHING:
        pred = gaussian_blur_torch(pred, FINAL_SMOOTH_KERNEL, FINAL_SMOOTH_SIGMA)
    return torch.clamp(pred, 0.0, 1.0)

# ============================================================
# LOCAL SSIM-LIKE LOSS / COMPONENTS (μόνο για training loss)
# ============================================================

def local_filter_torch(x: torch.Tensor, kernel_size=LOSS_SSIM_WINDOW_SIZE, sigma=LOSS_SSIM_WINDOW_SIGMA) -> torch.Tensor:
    x = x.float()
    c = x.shape[1]
    kernel = get_gaussian_kernel2d(kernel_size, sigma, x.device, x.dtype)
    weight = kernel.expand(c, 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x_pad, weight, groups=c)

def local_std_torch(x: torch.Tensor, kernel_size=LOSS_SSIM_WINDOW_SIZE, sigma=LOSS_SSIM_WINDOW_SIGMA) -> torch.Tensor:
    mu = local_filter_torch(x, kernel_size, sigma)
    var = local_filter_torch(x * x, kernel_size, sigma) - mu * mu
    var = torch.clamp(var, min=0.0)
    return torch.sqrt(var + 1e-12)

def ssim_components_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_size=LOSS_SSIM_WINDOW_SIZE,
    sigma=LOSS_SSIM_WINDOW_SIGMA,
    K1=LOSS_SSIM_K1,
    K2=LOSS_SSIM_K2,
):
    x = x.float()
    y = y.float()

    C1 = (K1 * 1.0) ** 2
    C2 = (K2 * 1.0) ** 2
    C3 = C2 / 2.0

    mu_x = local_filter_torch(x, kernel_size, sigma)
    mu_y = local_filter_torch(y, kernel_size, sigma)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = local_filter_torch(x * x, kernel_size, sigma) - mu_x_sq
    sigma_y_sq = local_filter_torch(y * y, kernel_size, sigma) - mu_y_sq
    sigma_xy = local_filter_torch(x * y, kernel_size, sigma) - mu_xy

    sigma_x_sq = torch.clamp(sigma_x_sq, min=0.0)
    sigma_y_sq = torch.clamp(sigma_y_sq, min=0.0)

    sigma_x = torch.sqrt(sigma_x_sq + 1e-12)
    sigma_y = torch.sqrt(sigma_y_sq + 1e-12)

    l_map = (2.0 * mu_xy + C1) / (mu_x_sq + mu_y_sq + C1 + 1e-12)
    c_map = (2.0 * sigma_x * sigma_y + C2) / (sigma_x_sq + sigma_y_sq + C2 + 1e-12)
    s_map = (sigma_xy + C3) / (sigma_x * sigma_y + C3 + 1e-12)

    ssim_map = l_map * c_map * s_map

    return {
        "ssim": ssim_map.mean(),
        "luminosity": l_map.mean(),
        "contrast": c_map.mean(),
        "structure": s_map.mean(),
    }

def ssim_torch(x, y):
    return ssim_components_torch(x, y)["ssim"]

# ============================================================
# REPORTED GLOBAL SSIM (FULL 90x90, NO CROP, NO SLIDING)
# ============================================================

def global_ssim_full_np(a: np.ndarray, b: np.ndarray):
    """
    Global SSIM πάνω σε όλη την εικόνα 90x90.
    - no crop
    - no sliding window
    - normalize each image to its own max = 1
    - C1 = C2 = 0
    """

    x = normalize_to_max_one(a).astype(np.float64)
    y = normalize_to_max_one(b).astype(np.float64)

    mu_x = float(np.mean(x))
    mu_y = float(np.mean(y))

    dx = x - mu_x
    dy = y - mu_y

    var_x = float(np.mean(dx * dx))
    var_y = float(np.mean(dy * dy))
    cov_xy = float(np.mean(dx * dy))

    eps = 1e-12

    lum = (2.0 * mu_x * mu_y) / (mu_x * mu_x + mu_y * mu_y + eps)
    contrast_structure = (2.0 * cov_xy) / (var_x + var_y + eps)
    ssim_val = lum * contrast_structure

    return {
        "ssim": float(ssim_val),
        "luminosity": float(lum),
        "contrast_structure": float(contrast_structure),
        "mu_x": float(mu_x),
        "mu_y": float(mu_y),
        "var_x": float(var_x),
        "var_y": float(var_y),
        "cov_xy": float(cov_xy),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
    }

def report_metrics_np(a: np.ndarray, b: np.ndarray):
    g = global_ssim_full_np(a, b)
    x = g["x"]
    y = g["y"]

    return {
        "ssim": g["ssim"],
        "mae": mae_np(x, y),
        "mse": mse_np(x, y),
        "rmse": rmse_np(x, y),
        "luminosity": g["luminosity"],
        "contrast_structure": g["contrast_structure"],
        "a": x,
        "b": y,
        "shape": x.shape,
    }

# ============================================================
# FILE DISCOVERY / PAIRING
# ============================================================

def walk_files(root: Path, exts=(".sin", ".mtx"), max_depth=6):
    root = Path(root)
    if not root.exists():
        return
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        cur = Path(dirpath)
        depth = len(cur.parts) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue
        for fn in filenames:
            low = fn.lower()
            if any(low.endswith(ext) for ext in exts):
                yield cur / fn

def find_pairs(search_dirs):
    sins, mtxs = [], []

    for r in search_dirs:
        for p in walk_files(r, exts=(".sin",), max_depth=MAX_DEPTH):
            if p.name.lower().startswith("patient_"):
                continue
            sins.append(p)

        for p in walk_files(r, exts=(".mtx",), max_depth=MAX_DEPTH):
            low = p.name.lower()
            if low.startswith("mlem_recon_"):
                continue
            if low.startswith("patient_"):
                continue
            mtxs.append(p)

    sin_by_key = {(p.parent, p.stem): p for p in sins}
    pairs = []

    for m in mtxs:
        k = (m.parent, m.stem)
        if k in sin_by_key:
            pairs.append((m.stem, sin_by_key[k], m))

    if not pairs:
        sin_by_stem = {}
        for s in sins:
            sin_by_stem.setdefault(s.stem, []).append(s)
        for m in mtxs:
            cands = sin_by_stem.get(m.stem, [])
            if len(cands) >= 1:
                pairs.append((m.stem, cands[0], m))

    pairs = sorted(pairs, key=lambda x: (str(x[1].parent), x[0]))

    print("Found sin files :", len(sins))
    print("Found mtx files :", len(mtxs))
    print("Matched pairs   :", len(pairs))

    if len(pairs) == 0:
        raise RuntimeError("No matched .sin/.mtx pairs found by stem.")

    print("Example pair    :", pairs[0][0], "|", pairs[0][1].name, "|", pairs[0][2].name)
    return pairs

# ============================================================
# LOADERS
# ============================================================

def load_sin_native(path: Path, proj_count=SINOGRAM_PROJ_COUNT, candidates=PROJ_COUNT_CANDIDATES) -> np.ndarray:
    txt = path.read_text(errors="ignore").replace("D", "E").replace("d", "E").replace("...", " ")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]

    rows = []
    counts = []

    for ln in lines:
        toks = _NUM_RE.findall(ln)
        if not toks:
            continue
        counts.append(len(toks))
        rows.append([float(t) for t in toks])

    if rows and len(set(counts)) == 1:
        M = np.array(rows, dtype=np.float32)
        M = orient_sinogram_matrix(M, proj_count=proj_count, candidates=candidates)
        M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

        if M.shape != (SINOGRAM_PROJ_COUNT, SINOGRAM_DET_COUNT):
            M = resize_sinogram_np(M, out_A=SINOGRAM_PROJ_COUNT, out_D=SINOGRAM_DET_COUNT)

        return M.astype(np.float32)

    arr = read_numbers(path)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    n = int(arr.size)

    inferred_proj = infer_proj_count_from_flat_length(
        n=n,
        preferred=proj_count,
        candidates=candidates,
        preferred_detectors=SINOGRAM_DET_COUNT
    )

    if inferred_proj is None:
        raise ValueError(f"{path.name}: δεν βρέθηκε σωστό projection count για n={n}")

    D = n // inferred_proj
    M = arr.reshape(inferred_proj, D).astype(np.float32)

    if M.shape != (SINOGRAM_PROJ_COUNT, SINOGRAM_DET_COUNT):
        M = resize_sinogram_np(M, out_A=SINOGRAM_PROJ_COUNT, out_D=SINOGRAM_DET_COUNT)

    return M

def load_mtx_2d(path: Path, out_size=IMG_SIZE, normalize=True) -> np.ndarray:
    path = Path(path)
    try:
        m = np.loadtxt(path, dtype=np.float32)
    except Exception:
        m = read_numbers(path)

    if isinstance(m, np.ndarray) and m.ndim == 2:
        img = m.astype(np.float32)
    else:
        m = np.asarray(m, dtype=np.float32).reshape(-1)
        n = int(m.size)

        if n == out_size * out_size:
            img = m.reshape(out_size, out_size).astype(np.float32)
        else:
            s = int(round(math.sqrt(n)))
            if s * s != n:
                raise ValueError(f"{path.name}: cannot reshape target (n={n})")
            img = m.reshape(s, s).astype(np.float32)

    if img.shape != (out_size, out_size):
        img = resize_image_np(img, out_h=out_size, out_w=out_size)

    if normalize:
        img = normalize_to_max_one(img)

    return img.astype(np.float32)

# ============================================================
# SINOGRAM PROCESSING
# ============================================================

def sinogram_to_line_integrals(sino_native: np.ndarray, mode=SINOGRAM_FORMAT) -> np.ndarray:
    s = np.asarray(sino_native, dtype=np.float32)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    if mode == "line_integrals":
        return s.astype(np.float32)

    if mode == "transmission":
        s = np.clip(s, 1e-6, None)
        s = -np.log(s)
        return s.astype(np.float32)

    raise ValueError(f"Unknown SINOGRAM_FORMAT: {mode}")

def prepare_projection_for_fbp(sino_native: np.ndarray) -> np.ndarray:
    P = sinogram_to_line_integrals(sino_native, SINOGRAM_FORMAT)

    if FBP_FLIP_DET:
        P = P[:, ::-1]

    if FBP_FLIP_ANG:
        P = P[::-1, :]

    return np.asarray(P, dtype=np.float32)

def pad_angles_circular_np(sino: np.ndarray, target_A: int) -> np.ndarray:
    A, D = sino.shape
    if A == target_A:
        return sino
    if A > target_A:
        start = (A - target_A) // 2
        return sino[start:start + target_A, :]

    pad = target_A - A
    pre = pad // 2
    post = pad - pre

    top = sino[-pre:, :] if pre > 0 else sino[:0, :]
    bottom = sino[:post, :] if post > 0 else sino[:0, :]
    return np.concatenate([top, sino, bottom], axis=0)

def pad_detector_np(sino: np.ndarray, target_D: int) -> np.ndarray:
    A, D = sino.shape
    if D == target_D:
        return sino
    if D > target_D:
        start = (D - target_D) // 2
        return sino[:, start:start + target_D]

    pad = target_D - D
    left = pad // 2
    right = pad - left

    mode = "reflect" if (D > 1 and left < D and right < D) else "edge"
    return np.pad(sino, ((0, 0), (left, right)), mode=mode)

def prepare_sino_for_display(sino_native: np.ndarray, A_pad: int, D_pad: int) -> np.ndarray:
    s = prepare_projection_for_fbp(sino_native)
    s = robust_01(s, 1, 99)
    s = pad_angles_circular_np(s, A_pad)
    s = pad_detector_np(s, D_pad)
    return s.astype(np.float32)

def prepare_sino_for_display_from_proj(proj: np.ndarray, A_pad: int, D_pad: int) -> np.ndarray:
    s = robust_01(proj, 1, 99)
    s = pad_angles_circular_np(s, A_pad)
    s = pad_detector_np(s, D_pad)
    return s.astype(np.float32)

def degrade_simulix_projection_to_patient_like(sino_native: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    proj = prepare_projection_for_fbp(sino_native)
    proj = resize_sinogram_np(proj, out_A=MODEL_PROJ_COUNT, out_D=MODEL_CANVAS_DET)
    proj = np.clip(proj, 0.0, None)
    proj = robust_01(proj, 1, 99)

    sigma_det = float(rng.uniform(*DEG_BLUR_DET_SIGMA))
    sigma_ang = float(rng.uniform(*DEG_BLUR_ANG_SIGMA))

    if sigma_det > 0:
        proj = gaussian_filter1d(proj, sigma=sigma_det, axis=1, mode="nearest").astype(np.float32)
    if sigma_ang > 0:
        proj = gaussian_filter1d(proj, sigma=sigma_ang, axis=0, mode="nearest").astype(np.float32)

    bg_frac = float(rng.uniform(*DEG_BACKGROUND_FRAC))
    count_scale = float(rng.uniform(*DEG_COUNT_SCALE))
    gauss_std = float(rng.uniform(*DEG_GAUSS_STD))

    vmax = float(proj.max()) if proj.size > 0 else 1.0
    proj = np.clip(proj, 0.0, None)
    proj = proj + bg_frac * max(vmax, 1e-6)

    noisy = rng.poisson(np.clip(proj, 0.0, None) * count_scale).astype(np.float32) / max(count_scale, 1e-6)

    if gauss_std > 0:
        noisy = noisy + rng.normal(0.0, gauss_std, size=noisy.shape).astype(np.float32)

    noisy = np.clip(noisy, 0.0, None)
    return noisy.astype(np.float32)

def canonicalize_patient_projection(sino_native: np.ndarray):
    proj = prepare_projection_for_fbp(sino_native)
    proj = resize_sinogram_np(proj, out_A=MODEL_PROJ_COUNT, out_D=MODEL_CANVAS_DET)
    proj = np.clip(proj, 0.0, None)
    proj = robust_01(proj, 1, 99)

    active_range = (0, MODEL_CANVAS_DET - 1)
    return proj.astype(np.float32), active_range

# ============================================================
# FBP
# ============================================================

def fbp_reconstruction(
    sino_native: np.ndarray = None,
    out_size: int = IMG_SIZE,
    proj: np.ndarray = None,
    theta_max: float = None,
    filter_name: str = None
):
    if proj is None:
        if sino_native is None:
            raise ValueError("Δώσε είτε sino_native είτε proj")
        proj = prepare_projection_for_fbp(sino_native)

    proj = np.asarray(proj, dtype=np.float32)

    if theta_max is None:
        theta_max = FBP_THETA_MAX
    if filter_name is None:
        filter_name = FBP_FILTER

    angles = np.linspace(0.0, theta_max, proj.shape[0], endpoint=False)

    rec = iradon(
        proj.T,
        theta=angles,
        output_size=out_size,
        circle=FBP_CIRCLE,
        filter_name=filter_name
    )

    rec = np.nan_to_num(rec.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    rec = robust_01(rec, 1, 99)
    return rec.astype(np.float32)

def make_fbp_cache_path(sin_path: Path) -> Path:
    key = hashlib.md5(str(sin_path.resolve()).encode("utf-8")).hexdigest()[:16]
    proj_tag = f"proj{SINOGRAM_PROJ_COUNT}" if SINOGRAM_PROJ_COUNT is not None else "projAUTO"
    theta_tag = f"theta{int(FBP_THETA_MAX)}"

    name = (
        f"{sin_path.stem}_{key}_img{IMG_SIZE}_{proj_tag}_{theta_tag}_{SINOGRAM_FORMAT}_"
        f"flipA{int(FBP_FLIP_ANG)}_flipD{int(FBP_FLIP_DET)}_{FBP_FILTER}.npy"
    )
    return FBP_CACHE_DIR / name

def make_model_input_cache_path(sin_path: Path) -> Path:
    key = hashlib.md5(str(sin_path.resolve()).encode("utf-8")).hexdigest()[:16]
    name = (
        f"{sin_path.stem}_{key}_input_img{IMG_SIZE}_proj{MODEL_PROJ_COUNT}_"
        f"det{MODEL_CANVAS_DET}_{INPUT_FBP_FILTER}_patientlike{int(USE_PATIENT_LIKE_DEGRADATION)}.npy"
    )
    return FBP_CACHE_DIR / name

# ============================================================
# DATASET
# ============================================================

class ReconDataset(Dataset):
    def __init__(self, search_dirs):
        all_pairs = find_pairs(search_dirs)
        self.samples = []

        dmax = 0
        amax = 0
        angle_counts = []

        print("\nValidating dataset files...")
        t0 = time.time()

        valid_items = []
        for stem, sin_path, mtx_path in all_pairs:
            try:
                sino_native = load_sin_native(
                    sin_path,
                    proj_count=SINOGRAM_PROJ_COUNT,
                    candidates=PROJ_COUNT_CANDIDATES
                )
                _ = load_mtx_2d(mtx_path, IMG_SIZE)

                amax = max(amax, sino_native.shape[0])
                dmax = max(dmax, sino_native.shape[1])
                angle_counts.append(sino_native.shape[0])

                valid_items.append({
                    "stem": stem,
                    "sin_path": sin_path,
                    "mtx_path": mtx_path
                })
            except Exception as e:
                print(f"Skip {stem}: {e}")

        if len(valid_items) == 0:
            raise RuntimeError("No valid samples after loading.")

        if USE_PATIENT_LIKE_DEGRADATION:
            self.A_pad = next_mult(MODEL_PROJ_COUNT, ANGLE_PAD_MULT)
            self.D_pad = next_mult(MODEL_CANVAS_DET, DET_MULT)
        else:
            self.A_pad = next_mult(amax, ANGLE_PAD_MULT)
            self.D_pad = next_mult(dmax, DET_MULT)

        print("Valid samples    :", len(valid_items))
        print("Detected angles  :", sorted(set(angle_counts)))
        print("Display sino size:", (self.A_pad, self.D_pad))
        print("Patient-like mode:", USE_PATIENT_LIKE_DEGRADATION)
        print(f"Validation time  : {time.time() - t0:.1f} sec")

        print("\nPrecomputing model inputs / targets / display sinograms...")
        t1 = time.time()

        for i, item in enumerate(valid_items, start=1):
            sino_native = load_sin_native(
                item["sin_path"],
                proj_count=SINOGRAM_PROJ_COUNT,
                candidates=PROJ_COUNT_CANDIDATES
            )
            target = load_mtx_2d(item["mtx_path"], IMG_SIZE)

            if USE_PATIENT_LIKE_DEGRADATION:
                rng = np.random.default_rng(
                    stable_int_from_path(item["sin_path"], TRAIN_INPUT_SEED_OFFSET)
                )

                model_proj = degrade_simulix_projection_to_patient_like(sino_native, rng)
                sino_vis = prepare_sino_for_display_from_proj(model_proj, self.A_pad, self.D_pad)

                if CACHE_FBP_TO_DISK:
                    cache_path = make_model_input_cache_path(item["sin_path"])
                    if cache_path.exists():
                        try:
                            baseline = np.load(cache_path).astype(np.float32)
                        except Exception:
                            baseline = fbp_reconstruction(
                                proj=model_proj,
                                out_size=IMG_SIZE,
                                theta_max=SINOGRAM_THETA_MAX,
                                filter_name=INPUT_FBP_FILTER
                            )
                            np.save(cache_path, baseline.astype(np.float32))
                    else:
                        baseline = fbp_reconstruction(
                            proj=model_proj,
                            out_size=IMG_SIZE,
                            theta_max=SINOGRAM_THETA_MAX,
                            filter_name=INPUT_FBP_FILTER
                        )
                        np.save(cache_path, baseline.astype(np.float32))
                else:
                    baseline = fbp_reconstruction(
                        proj=model_proj,
                        out_size=IMG_SIZE,
                        theta_max=SINOGRAM_THETA_MAX,
                        filter_name=INPUT_FBP_FILTER
                    )
            else:
                sino_vis = prepare_sino_for_display(sino_native, self.A_pad, self.D_pad)

                if CACHE_FBP_TO_DISK:
                    cache_path = make_fbp_cache_path(item["sin_path"])
                    if cache_path.exists():
                        try:
                            baseline = np.load(cache_path).astype(np.float32)
                        except Exception:
                            baseline = fbp_reconstruction(sino_native, IMG_SIZE)
                            np.save(cache_path, baseline.astype(np.float32))
                    else:
                        baseline = fbp_reconstruction(sino_native, IMG_SIZE)
                        np.save(cache_path, baseline.astype(np.float32))
                else:
                    baseline = fbp_reconstruction(sino_native, IMG_SIZE)

            inp_t = torch.from_numpy(baseline).unsqueeze(0).to(torch.float16)
            tgt_t = torch.from_numpy(target).unsqueeze(0).to(torch.float16)
            sino_t = torch.from_numpy(sino_vis).unsqueeze(0).to(torch.float16)

            self.samples.append((inp_t, tgt_t, sino_t, item["stem"]))

            if i % 100 == 0 or i == len(valid_items):
                print(f"  precomputed {i}/{len(valid_items)}")

        print(f"Precompute time  : {time.time() - t1:.1f} sec")
        print("Dataset ready.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp_t, tgt_t, sino_t, stem = self.samples[idx]
        return inp_t.clone(), tgt_t.clone(), sino_t.clone(), stem

# ============================================================
# MODEL
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResidualUNetRefiner(nn.Module):
    def __init__(self, base_ch=48):
        super().__init__()

        self.inc = DoubleConv(1, base_ch)
        self.d1  = Down(base_ch, base_ch * 2)
        self.d2  = Down(base_ch * 2, base_ch * 4)
        self.d3  = Down(base_ch * 4, base_ch * 8)

        self.u1 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.u2 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.u3 = Up(base_ch * 2, base_ch, base_ch)

        self.out = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, baseline):
        x1 = self.inc(baseline)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)

        y = self.u1(x4, x3)
        y = self.u2(y, x2)
        y = self.u3(y, x1)

        correction = self.out(y)
        pred = torch.sigmoid(baseline + correction)
        return pred

model = ResidualUNetRefiner(base_ch=BASE_CH).to(device)
print("Model ready.")

# ============================================================
# METRICS / LOSS
# ============================================================

def rmse_torch(pred, true):
    return torch.sqrt(F.mse_loss(pred, true) + 1e-12)

def edge_map(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx * dx + dy * dy + 1e-12)

def contrast_loss_torch(pred, true):
    pred_std = local_std_torch(pred)
    true_std = local_std_torch(true)
    return F.l1_loss(pred_std, true_std)

def loss_fn(pred, true):
    weights = 1.0 + 3.0 * true
    weighted_l1 = (weights * torch.abs(pred - true)).mean()

    ssim_loss = 1.0 - ssim_torch(pred, true)
    edge_loss = F.l1_loss(edge_map(pred), edge_map(true))
    ctr_loss = contrast_loss_torch(pred, true)

    total = (
        LAMBDA_WEIGHTED_L1 * weighted_l1 +
        LAMBDA_SSIM * ssim_loss +
        LAMBDA_EDGE * edge_loss +
        LAMBDA_CONTRAST * ctr_loss
    )
    return total

# ============================================================
# PREVIEW
# ============================================================

@torch.no_grad()
def preview_sample(dataset, idx=0):
    idx = max(0, min(idx, len(dataset) - 1))
    inp_t, tgt_t, sino_t, stem = dataset[idx]

    sino_np = sino_t[0].float().numpy()
    inp_np  = inp_t[0].float().numpy()
    tgt_np  = tgt_t[0].float().numpy()

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(sino_np, cmap=SINO_CMAP, aspect="auto", vmin=0, vmax=1)
    plt.title(f"Model-input sinogram\n{stem}")
    plt.xlabel("Detector bin")
    plt.ylabel("Angle")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 3, 2)
    plt.imshow(inp_np, cmap=IMG_CMAP, vmin=0, vmax=1)
    plt.title("Baseline input reconstruction")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 3, 3)
    plt.imshow(tgt_np, cmap=IMG_CMAP, vmin=0, vmax=1)
    plt.title("Target image")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

# ============================================================
# DATA
# ============================================================

dataset = ReconDataset(SEARCH_DIRS)

sin_files = [p for p in SIMULIX_DIR.rglob("*.sin") if not p.name.lower().startswith("patient_")]
mtx_files = [p for p in SIMULIX_DIR.rglob("*.mtx") if not p.name.lower().startswith("mlem_recon_")]
print("\nFiles inside simulix folder:")
print("sin files:", len(sin_files))
print("mtx files:", len(mtx_files))

print("\n=== PREVIEW BEFORE TRAINING ===")
preview_sample(dataset, idx=0)

n_total = len(dataset)
assert FIXED_TRAIN + FIXED_VAL + FIXED_TEST == n_total, (
    f"Το split δεν ταιριάζει με το dataset: "
    f"{FIXED_TRAIN}+{FIXED_VAL}+{FIXED_TEST} != {n_total}"
)

n_train, n_val, n_test = FIXED_TRAIN, FIXED_VAL, FIXED_TEST

train_ds, val_ds, test_ds = random_split(
    dataset,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED)
)

loader_kwargs = dict(
    batch_size=BATCH,
    num_workers=NUM_WORKERS,
    pin_memory=(device == "cuda"),
)

train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

print(f"Split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

# ============================================================
# OPTIMIZER / SCHEDULER
# ============================================================

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
)

best_model_local_path = LOCAL_RESULTS_DIR / "best_patientlike_unet_refiner_72x128_to_90x90.pth"
best_model_drive_path = DRIVE_RESULTS_DIR / "best_patientlike_unet_refiner_72x128_to_90x90.pth"

# ============================================================
# EVAL
# ============================================================

@torch.no_grad()
def evaluate(loader):
    model.eval()

    total_loss = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    total_lum = 0.0
    total_cs = 0.0
    count = 0

    for inp, img, _, _ in loader:
        inp = inp.to(device, non_blocking=True).float()
        img = img.to(device, non_blocking=True).float()

        with autocast_context():
            pred_raw = model(inp)

        pred = finalize_prediction_torch(pred_raw)
        loss = loss_fn(pred, img)

        pred_np = pred[:, 0].float().cpu().numpy()
        img_np  = img[:, 0].float().cpu().numpy()

        bs = inp.size(0)
        total_loss += float(loss.item()) * bs

        for i in range(bs):
            rep = report_metrics_np(pred_np[i], img_np[i])

            total_ssim += rep["ssim"]
            total_rmse += rep["rmse"]
            total_lum += rep["luminosity"]
            total_cs += rep["contrast_structure"]
            count += 1

    return {
        "loss": total_loss / max(1, count),
        "ssim": total_ssim / max(1, count),
        "luminosity": total_lum / max(1, count),
        "contrast_structure": total_cs / max(1, count),
        "rmse": total_rmse / max(1, count),
    }

# ============================================================
# TRAIN
# ============================================================

hist_train_loss = []
hist_val_loss = []
hist_val_ssim = []
hist_val_lum = []
hist_val_cs = []
hist_val_rmse = []
hist_lr = []
hist_val_epochs = []

best_val_loss = float("inf")
best_state = None
patience_counter = 0

print("\n=== TRAIN START ===")
print(f"Final smoothing: {APPLY_FINAL_SMOOTHING} | kernel={FINAL_SMOOTH_KERNEL}x{FINAL_SMOOTH_KERNEL} | sigma={FINAL_SMOOTH_SIGMA}")
print("Reported metric: FULL 90x90 | no crop | no sliding | max-normalized | C1=C2=0")

for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    seen = 0
    t0 = time.time()

    for inp, img, _, _ in train_loader:
        inp = inp.to(device, non_blocking=True).float()
        img = img.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with autocast_context():
            pred_raw = model(inp)

        pred = finalize_prediction_torch(pred_raw)
        loss = loss_fn(pred, img)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = inp.size(0)
        running += float(loss.item()) * bs
        seen += bs

    train_loss = running / max(1, seen)
    current_lr = optimizer.param_groups[0]["lr"]

    hist_train_loss.append(train_loss)
    hist_lr.append(current_lr)

    do_val = ((epoch + 1) % VAL_EVERY == 0) or ((epoch + 1) == EPOCHS)

    if do_val:
        val_metrics = evaluate(val_loader)
        scheduler.step(val_metrics["loss"])

        hist_val_loss.append(val_metrics["loss"])
        hist_val_ssim.append(val_metrics["ssim"])
        hist_val_lum.append(val_metrics["luminosity"])
        hist_val_cs.append(val_metrics["contrast_structure"])
        hist_val_rmse.append(val_metrics["rmse"])
        hist_val_epochs.append(epoch + 1)

        print(
            f"Epoch {epoch+1:03d} | "
            f"lr {current_lr:.2e} | "
            f"train_loss {train_loss:.5f} | "
            f"val_loss {val_metrics['loss']:.5f} | "
            f"val_ssim {val_metrics['ssim']:.4f} | "
            f"val_lum {val_metrics['luminosity']:.4f} | "
            f"val_cs {val_metrics['contrast_structure']:.4f} | "
            f"val_rmse {val_metrics['rmse']:.5f} | "
            f"time {time.time() - t0:.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, best_model_local_path)
            patience_counter = 0
            print("Saved best model ->", best_model_local_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break
    else:
        print(
            f"Epoch {epoch+1:03d} | "
            f"lr {current_lr:.2e} | "
            f"train_loss {train_loss:.5f} | "
            f"(no validation this epoch) | "
            f"time {time.time() - t0:.1f}s"
        )

if best_state is not None:
    model.load_state_dict(best_state)
    shutil.copy2(best_model_local_path, best_model_drive_path)
    print("Best model copied to Drive ->", best_model_drive_path)

# ============================================================
# TEST
# ============================================================

test_metrics = evaluate(test_loader)
print("\n=== TEST ===")
print(f"Test loss       : {test_metrics['loss']:.5f}")
print(f"Test SSIM       : {test_metrics['ssim']:.4f}")
print(f"Test Luminosity : {test_metrics['luminosity']:.4f}")
print(f"Test C*S        : {test_metrics['contrast_structure']:.4f}")
print(f"Test RMSE       : {test_metrics['rmse']:.5f}")

# ============================================================
# TRAIN / VAL PLOTS
# ============================================================

plt.figure(figsize=(20, 8))

plt.subplot(2, 3, 1)
plt.plot(hist_train_loss, label="train loss")
if len(hist_val_epochs) > 0:
    plt.plot(hist_val_epochs, hist_val_loss, label="val loss")
plt.title("Train / Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 2)
if len(hist_val_epochs) > 0:
    plt.plot(hist_val_epochs, hist_val_ssim, label="val SSIM")
    plt.plot(hist_val_epochs, hist_val_lum, label="val Luminosity")
plt.title("Validation SSIM / Luminosity")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 3)
if len(hist_val_epochs) > 0:
    plt.plot(hist_val_epochs, hist_val_cs, label="val C*S")
plt.title("Validation Contrast*Structure")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 4)
if len(hist_val_epochs) > 0:
    plt.plot(hist_val_epochs, hist_val_rmse, label="val RMSE")
plt.title("Validation RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(hist_lr, label="lr")
plt.title("Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 6)
if len(hist_val_epochs) > 0:
    plt.plot(hist_val_epochs, hist_val_ssim, label="SSIM")
    plt.plot(hist_val_epochs, hist_val_cs, label="C*S")
plt.title("SSIM Overview")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# VISUALIZE TEST SAMPLES
# ============================================================

@torch.no_grad()
def visualize_test_samples(loader, n_show=4):
    model.eval()

    shown = 0
    for inp, img, sino, stems in loader:
        inp = inp.to(device, non_blocking=True).float()
        img = img.to(device, non_blocking=True).float()

        with autocast_context():
            pred_raw = model(inp)

        pred = finalize_prediction_torch(pred_raw)

        for i in range(inp.size(0)):
            if shown >= n_show:
                return

            sino_np = sino[i, 0].float().cpu().numpy()
            inp_np  = inp[i, 0].float().cpu().numpy()
            img_np  = img[i, 0].float().cpu().numpy()
            pred_np = pred[i, 0].float().cpu().numpy()

            rep_inp = report_metrics_np(inp_np, img_np)
            rep_pred = report_metrics_np(pred_np, img_np)

            inp_norm = rep_inp["a"]
            img_norm = rep_pred["b"]
            pred_norm = rep_pred["a"]
            err_np = np.abs(pred_norm - img_norm)

            print("\n------------------------------------------")
            print(f"Sample: {stems[i]}")
            print(f"  full image shape = {rep_pred['shape']}")
            print(f"  SSIM total       = {rep_pred['ssim']:.6f}")
            print(f"  Luminosity       = {rep_pred['luminosity']:.6f}")
            print(f"  C*S              = {rep_pred['contrast_structure']:.6f}")
            print(f"  RMSE             = {rep_pred['rmse']:.6f}")

            plt.figure(figsize=(18, 4))

            plt.subplot(1, 5, 1)
            plt.imshow(sino_np, cmap=SINO_CMAP, aspect="auto", vmin=0, vmax=1)
            plt.title(f"Model-input sinogram\n{stems[i]}")
            plt.xlabel("Detector bin")
            plt.ylabel("Angle")
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.subplot(1, 5, 2)
            plt.imshow(inp_norm, cmap=IMG_CMAP, vmin=0, vmax=1)
            plt.title("Baseline input\n(normalized)")
            plt.axis("off")
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.subplot(1, 5, 3)
            plt.imshow(img_norm, cmap=IMG_CMAP, vmin=0, vmax=1)
            plt.title("True image\n(normalized)")
            plt.axis("off")
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.subplot(1, 5, 4)
            plt.imshow(pred_norm, cmap=IMG_CMAP, vmin=0, vmax=1)
            plt.title(
                f"Predicted (5x5 smooth)\n"
                f"SSIM={rep_pred['ssim']:.4f}\n"
                f"L={rep_pred['luminosity']:.4f}  C*S={rep_pred['contrast_structure']:.4f}"
            )
            plt.axis("off")
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.subplot(1, 5, 5)
            plt.imshow(err_np, cmap=ERR_CMAP)
            plt.title("Absolute error")
            plt.axis("off")
            plt.colorbar(fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.show()

            shown += 1

print("\n=== TEST SAMPLE VISUALIZATION ===")
visualize_test_samples(test_loader, n_show=4)

# ============================================================
# PATIENT + REFERENCE COMPARISON
# ============================================================

def resolve_reference_file(candidates):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    return None

PATIENT_COMPARE_PAIRS = [
    {
        "tag": "A",
        "patient_sin": DRIVE_SIMULIX_DIR / "Patient_a_p72.sin",
        "deep_out": DRIVE_RESULTS_DIR / "patient_a_deep_raw_90x90.mtx",
        "ref_candidates": [
            DRIVE_SIMULIX_DIR / "mlem_recon_a.mtx",
            DRIVE_RESULTS_DIR / "mlem_recon_a.mtx",
            Path("/mnt/data/mlem_recon_a.mtx"),
        ],
    },
    {
        "tag": "B",
        "patient_sin": DRIVE_SIMULIX_DIR / "Patient_b_p72.sin",
        "deep_out": DRIVE_RESULTS_DIR / "patient_b_deep_raw_90x90.mtx",
        "ref_candidates": [
            DRIVE_SIMULIX_DIR / "mlem_recon_b.mtx",
            DRIVE_RESULTS_DIR / "mlem_recon_b.mtx",
            Path("/mnt/data/mlem_recon_b.mtx"),
        ],
    },
]

@torch.no_grad()
def compare_patient_to_reference(patient_sin_path, reference_mtx_path, deep_out_path, title_tag=""):
    patient_sin_path = Path(patient_sin_path)
    reference_mtx_path = Path(reference_mtx_path)
    deep_out_path = Path(deep_out_path)

    sino_native = load_sin_native(
        patient_sin_path,
        proj_count=SINOGRAM_PROJ_COUNT,
        candidates=PROJ_COUNT_CANDIDATES
    )

    model_proj, active_range = canonicalize_patient_projection(sino_native)
    sino_vis = prepare_sino_for_display_from_proj(model_proj, dataset.A_pad, dataset.D_pad)

    baseline = fbp_reconstruction(
        proj=model_proj,
        out_size=IMG_SIZE,
        theta_max=SINOGRAM_THETA_MAX,
        filter_name=INPUT_FBP_FILTER
    )

    x = torch.from_numpy(baseline).unsqueeze(0).unsqueeze(0).float().to(device)

    model.eval()
    with autocast_context():
        pred_raw = model(x)

    pred = finalize_prediction_torch(pred_raw)[0, 0].float().cpu().numpy()

    # Αποθήκευση patient deep image μόνο στο Drive
    save_matrix_txt(deep_out_path, pred)
    print(f"Saved patient deep raw image -> {deep_out_path}")

    # Φόρτωση reference
    ref = load_mtx_2d(reference_mtx_path, out_size=IMG_SIZE, normalize=False)

    # Reported metrics with correct method
    baseline_parts = report_metrics_np(baseline, ref)
    pred_parts = report_metrics_np(pred, ref)

    baseline_img = baseline_parts["a"]
    pred_img = pred_parts["a"]
    ref_img = pred_parts["b"]

    err_base = np.abs(baseline_img - ref_img)
    err_pred = np.abs(pred_img - ref_img)

    improve_mae = pred_parts["mae"] - baseline_parts["mae"]
    improve_rmse = pred_parts["rmse"] - baseline_parts["rmse"]
    improve_ssim = pred_parts["ssim"] - baseline_parts["ssim"]
    improve_lum = pred_parts["luminosity"] - baseline_parts["luminosity"]
    improve_cs = pred_parts["contrast_structure"] - baseline_parts["contrast_structure"]

    print("\n==================================================")
    print(f"Patient file : {patient_sin_path}")
    print(f"Saved deep   : {deep_out_path}")
    print(f"Reference    : {reference_mtx_path}")
    print(f"Active range : {active_range}")
    print(f"Deep output  : Gaussian smoothing {FINAL_SMOOTH_KERNEL}x{FINAL_SMOOTH_KERNEL}, sigma={FINAL_SMOOTH_SIGMA}")
    print("--------------------------------------------------")
    print("Reported metric config:")
    print("  full image      = True")
    print("  image size      = 90x90")
    print("  crop            = False")
    print("  sliding window  = False")
    print("  normalization   = each image -> max = 1")
    print("  C1              = 0")
    print("  C2              = 0")
    print("--------------------------------------------------")
    print("Baseline vs reference")
    print(f"  MAE         = {baseline_parts['mae']:.6f}")
    print(f"  MSE         = {baseline_parts['mse']:.6f}")
    print(f"  RMSE        = {baseline_parts['rmse']:.6f}")
    print(f"  SSIM total  = {baseline_parts['ssim']:.6f}")
    print(f"  Luminosity  = {baseline_parts['luminosity']:.6f}")
    print(f"  C*S         = {baseline_parts['contrast_structure']:.6f}")

    print("Deep vs reference")
    print(f"  MAE         = {pred_parts['mae']:.6f}")
    print(f"  MSE         = {pred_parts['mse']:.6f}")
    print(f"  RMSE        = {pred_parts['rmse']:.6f}")
    print(f"  SSIM total  = {pred_parts['ssim']:.6f}")
    print(f"  Luminosity  = {pred_parts['luminosity']:.6f}")
    print(f"  C*S         = {pred_parts['contrast_structure']:.6f}")

    print("Difference (deep relative to baseline)")
    print(f"  ΔMAE        = {improve_mae:.6f}   (negative is better)")
    print(f"  ΔRMSE       = {improve_rmse:.6f}   (negative is better)")
    print(f"  ΔSSIM       = {improve_ssim:.6f}   (positive is better)")
    print(f"  ΔLuminosity = {improve_lum:.6f}   (positive is better)")
    print(f"  ΔC*S        = {improve_cs:.6f}   (positive is better)")

    plt.figure(figsize=(24, 8))

    plt.subplot(2, 4, 1)
    plt.imshow(sino_vis, cmap=SINO_CMAP, aspect="auto", vmin=0, vmax=1)
    plt.title(f"Patient sinogram {title_tag}")
    plt.xlabel("Detector bin")
    plt.ylabel("Angle")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 4, 2)
    plt.imshow(baseline_img, cmap=IMG_CMAP, vmin=0, vmax=1)
    plt.title(
        f"Baseline\n"
        f"SSIM={baseline_parts['ssim']:.4f}\n"
        f"L={baseline_parts['luminosity']:.4f}  C*S={baseline_parts['contrast_structure']:.4f}"
    )
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 4, 3)
    plt.imshow(pred_img, cmap=IMG_CMAP, vmin=0, vmax=1)
    plt.title(
        f"Deep (5x5 smooth)\n"
        f"SSIM={pred_parts['ssim']:.4f}\n"
        f"L={pred_parts['luminosity']:.4f}  C*S={pred_parts['contrast_structure']:.4f}"
    )
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 4, 4)
    plt.imshow(ref_img, cmap=IMG_CMAP, vmin=0, vmax=1)
    plt.title("Reference MLEM\n(normalized)")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 4, 5)
    plt.imshow(err_base, cmap=ERR_CMAP)
    plt.title("Abs error\nBaseline vs MLEM")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 4, 6)
    plt.imshow(err_pred, cmap=ERR_CMAP)
    plt.title("Abs error\nDeep vs MLEM")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 4, 7)
    plt.imshow(np.abs(pred_img - baseline_img), cmap=ERR_CMAP)
    plt.title("Abs difference\nDeep vs Baseline")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(2, 4, 8)
    vals = [
        baseline_parts["luminosity"], pred_parts["luminosity"],
        baseline_parts["contrast_structure"], pred_parts["contrast_structure"],
        baseline_parts["ssim"], pred_parts["ssim"],
    ]
    names = ["Base L", "Deep L", "Base C*S", "Deep C*S", "Base SSIM", "Deep SSIM"]
    plt.bar(np.arange(len(names)), vals)
    plt.xticks(np.arange(len(names)), names, rotation=45, ha="right")
    plt.title(f"Metric summary {title_tag}")
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "tag": title_tag,
        "patient_file": str(patient_sin_path),
        "deep_file": str(deep_out_path),
        "reference_file": str(reference_mtx_path),

        "baseline_mae": baseline_parts["mae"],
        "baseline_mse": baseline_parts["mse"],
        "baseline_rmse": baseline_parts["rmse"],
        "baseline_ssim": baseline_parts["ssim"],
        "baseline_luminosity": baseline_parts["luminosity"],
        "baseline_contrast_structure": baseline_parts["contrast_structure"],

        "deep_mae": pred_parts["mae"],
        "deep_mse": pred_parts["mse"],
        "deep_rmse": pred_parts["rmse"],
        "deep_ssim": pred_parts["ssim"],
        "deep_luminosity": pred_parts["luminosity"],
        "deep_contrast_structure": pred_parts["contrast_structure"],

        "delta_mae": pred_parts["mae"] - baseline_parts["mae"],
        "delta_rmse": pred_parts["rmse"] - baseline_parts["rmse"],
        "delta_ssim": pred_parts["ssim"] - baseline_parts["ssim"],
        "delta_luminosity": pred_parts["luminosity"] - baseline_parts["luminosity"],
        "delta_contrast_structure": pred_parts["contrast_structure"] - baseline_parts["contrast_structure"],
    }

comparison_results = []

print("\n=== PATIENT COMPARISONS (ONLY 2 FILES) ===")
for item in PATIENT_COMPARE_PAIRS:
    patient_sin = item["patient_sin"]
    deep_out = item["deep_out"]
    ref_mtx = resolve_reference_file(item["ref_candidates"])

    if not patient_sin.exists():
        print("Missing patient file:", patient_sin)
        continue

    if ref_mtx is None:
        print("Missing reference MLEM for:", patient_sin.name)
        print("Expected one of:")
        for c in item["ref_candidates"]:
            print("  ", c)
        continue

    res = compare_patient_to_reference(
        patient_sin_path=patient_sin,
        reference_mtx_path=ref_mtx,
        deep_out_path=deep_out,
        title_tag=item["tag"]
    )
    comparison_results.append(res)

# ============================================================
# SUMMARY TABLE + BAR CHARTS
# ============================================================

def print_comparison_summary(results):
    if len(results) == 0:
        print("No comparison results available.")
        return

    print("\n================ FINAL SUMMARY ================")
    for r in results:
        print(f"[Patient {r['tag']}]")
        print(
            f"  Baseline: "
            f"MAE={r['baseline_mae']:.6f} "
            f"RMSE={r['baseline_rmse']:.6f} "
            f"SSIM={r['baseline_ssim']:.6f} "
            f"L={r['baseline_luminosity']:.6f} "
            f"C*S={r['baseline_contrast_structure']:.6f}"
        )
        print(
            f"  Deep    : "
            f"MAE={r['deep_mae']:.6f} "
            f"RMSE={r['deep_rmse']:.6f} "
            f"SSIM={r['deep_ssim']:.6f} "
            f"L={r['deep_luminosity']:.6f} "
            f"C*S={r['deep_contrast_structure']:.6f}"
        )

        if r["deep_rmse"] < r["baseline_rmse"]:
            print("  RMSE winner     : Deep")
        else:
            print("  RMSE winner     : Baseline")

        if r["deep_ssim"] > r["baseline_ssim"]:
            print("  SSIM winner     : Deep")
        else:
            print("  SSIM winner     : Baseline")

        if r["deep_contrast_structure"] > r["baseline_contrast_structure"]:
            print("  C*S winner      : Deep")
        else:
            print("  C*S winner      : Baseline")

def plot_comparison_summary(results):
    if len(results) == 0:
        return

    tags = [r["tag"] for r in results]

    base_mae = [r["baseline_mae"] for r in results]
    deep_mae = [r["deep_mae"] for r in results]

    base_rmse = [r["baseline_rmse"] for r in results]
    deep_rmse = [r["deep_rmse"] for r in results]

    base_ssim = [r["baseline_ssim"] for r in results]
    deep_ssim = [r["deep_ssim"] for r in results]

    base_cs = [r["baseline_contrast_structure"] for r in results]
    deep_cs = [r["deep_contrast_structure"] for r in results]

    x = np.arange(len(tags))
    w = 0.35

    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.bar(x - w/2, base_mae, w, label="Baseline")
    plt.bar(x + w/2, deep_mae, w, label="Deep")
    plt.xticks(x, tags)
    plt.title("MAE vs Reference")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    plt.subplot(1, 4, 2)
    plt.bar(x - w/2, base_rmse, w, label="Baseline")
    plt.bar(x + w/2, deep_rmse, w, label="Deep")
    plt.xticks(x, tags)
    plt.title("RMSE vs Reference")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    plt.subplot(1, 4, 3)
    plt.bar(x - w/2, base_ssim, w, label="Baseline")
    plt.bar(x + w/2, deep_ssim, w, label="Deep")
    plt.xticks(x, tags)
    plt.title("SSIM vs Reference")
    plt.ylabel("SSIM")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    plt.subplot(1, 4, 4)
    plt.bar(x - w/2, base_cs, w, label="Baseline")
    plt.bar(x + w/2, deep_cs, w, label="Deep")
    plt.xticks(x, tags)
    plt.title("C*S vs Reference")
    plt.ylabel("C*S")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

print_comparison_summary(comparison_results)
plot_comparison_summary(comparison_results)

# ============================================================
# SAVE SUMMARY TO DRIVE
# ============================================================

def save_results_summary(results, out_path):
    if len(results) == 0:
        return

    lines = []
    lines.append(
        "tag,patient_file,deep_file,reference_file,"
        "baseline_mae,baseline_mse,baseline_rmse,baseline_ssim,baseline_luminosity,baseline_contrast_structure,"
        "deep_mae,deep_mse,deep_rmse,deep_ssim,deep_luminosity,deep_contrast_structure,"
        "delta_mae,delta_rmse,delta_ssim,delta_luminosity,delta_contrast_structure"
    )

    for r in results:
        lines.append(
            f"{r['tag']},{r['patient_file']},{r['deep_file']},{r['reference_file']},"
            f"{r['baseline_mae']:.8f},{r['baseline_mse']:.8f},{r['baseline_rmse']:.8f},{r['baseline_ssim']:.8f},"
            f"{r['baseline_luminosity']:.8f},{r['baseline_contrast_structure']:.8f},"
            f"{r['deep_mae']:.8f},{r['deep_mse']:.8f},{r['deep_rmse']:.8f},{r['deep_ssim']:.8f},"
            f"{r['deep_luminosity']:.8f},{r['deep_contrast_structure']:.8f},"
            f"{r['delta_mae']:.8f},{r['delta_rmse']:.8f},{r['delta_ssim']:.8f},"
            f"{r['delta_luminosity']:.8f},{r['delta_contrast_structure']:.8f}"
        )

    out_path = Path(out_path)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("Saved comparison summary ->", out_path)

summary_csv_path = DRIVE_RESULTS_DIR / "patient_comparison_summary.csv"
save_results_summary(comparison_results, summary_csv_path)

print("\nDONE.")
print("Best model saved locally at:", best_model_local_path)
print("Best model copied to Drive at:", best_model_drive_path)
print("Patient deep raw images saved at:")
print(" ", DRIVE_RESULTS_DIR / "patient_a_deep_raw_90x90.mtx")
print(" ", DRIVE_RESULTS_DIR / "patient_b_deep_raw_90x90.mtx")
print("Comparison summary saved at:", summary_csv_path)

