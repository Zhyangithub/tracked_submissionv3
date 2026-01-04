import numpy as np
import torch
from pathlib import Path

from model_def import UNetV2, SequenceInference

# TrackRAD: uploaded model will be extracted here
MODEL_DIR = Path("/opt/ml/model")
MODEL_PATH = MODEL_DIR / "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Load model ONCE (global, required by TrackRAD)
# =====================================================
_model = UNetV2(
    in_channels=4,
    out_channels=1,
    base_ch=64,
)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    _model.load_state_dict(ckpt["model_state_dict"])
else:
    _model.load_state_dict(ckpt)

_model.to(DEVICE)
_model.eval()

_sequence_infer = SequenceInference(_model)

# =====================================================
# REQUIRED ENTRY POINT (DO NOT RENAME)
# =====================================================
def run_algorithm(
    frames: np.ndarray,                 # (W, H, T)
    target: np.ndarray,                 # (W, H, 1)
    frame_rate: float,
    magnetic_field_strength: float,
    scanned_region: str,
) -> np.ndarray:
    """
    Must return (W, H, T) uint8
    """

    # ---------- sanity checks ----------
    assert frames.ndim == 3, f"frames must be (W,H,T), got {frames.shape}"
    assert target.ndim == 3 and target.shape[-1] == 1, \
        f"target must be (W,H,1), got {target.shape}"

    W, H, T = frames.shape

    # ---------- numpy -> torch ----------
    # (W,H,T) -> (T,1,H,W)
    frames_t = torch.from_numpy(frames).permute(2, 1, 0).unsqueeze(1).float()
    target_t = torch.from_numpy(target[..., 0]).unsqueeze(0).unsqueeze(0).float()

    frames_t = frames_t.to(DEVICE)
    target_t = target_t.to(DEVICE)

    # ---------- inference ----------
    with torch.no_grad():
        preds = _sequence_infer.infer(frames_t, target_t)
        # preds: (T, H, W)

    # ---------- torch -> numpy ----------
    preds = preds.detach().cpu().numpy()        # (T,H,W)
    preds = preds.transpose(2, 1, 0)             # (W,H,T)

    return preds.astype(np.uint8)
