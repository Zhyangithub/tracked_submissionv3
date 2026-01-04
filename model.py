# model.py
import numpy as np
import torch
from pathlib import Path

from model_def import UNetV2, SequenceInference  # 如果你原来在 model.py 定义网络，把名字对齐即可

RESOURCE_PATH = Path("/opt/ml/model")  # 平台上传的 model 会解压到这里
MODEL_PATH = RESOURCE_PATH / "best_model.pth"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load model ONCE (global)
# =========================
_model = UNetV2(
    in_channels=4,
    out_channels=1,
    base_ch=64,
)

ckpt = torch.load(MODEL_PATH, map_location=_device)

if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    _model.load_state_dict(ckpt["model_state_dict"])
else:
    _model.load_state_dict(ckpt)

_model.to(_device)
_model.eval()

_sequence_infer = SequenceInference(_model)


# =========================
# REQUIRED ENTRY POINT
# =========================
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

    # ---------- shape check ----------
    assert frames.ndim == 3
    assert target.ndim == 3 and target.shape[-1] == 1

    W, H, T = frames.shape

    # ---------- to torch ----------
    # (W, H, T) -> (T, 1, H, W)
    frames_t = torch.from_numpy(frames).permute(2, 0, 1).unsqueeze(1).float()
    target_t = torch.from_numpy(target[..., 0]).unsqueeze(0).unsqueeze(0).float()

    frames_t = frames_t.to(_device)
    target_t = target_t.to(_device)

    # ---------- inference ----------
    with torch.no_grad():
        preds = _sequence_infer.infer(frames_t, target_t)
        # preds: (T, H, W)

    # ---------- back to numpy ----------
    preds = preds.detach().cpu().numpy()          # (T, H, W)
    preds = preds.transpose(1, 2, 0)               # (W, H, T)

    return preds.astype(np.uint8)
