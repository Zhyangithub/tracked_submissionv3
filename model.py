"""
TrackRAD2025 Submission - Corrected Version
Fixes:
1. Normalization mismatch (Percentile vs Max)
2. Geometry transposition (W,H,T vs T,H,W)
3. Architecture loading
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom, label as scipy_label, sum as scipy_sum

# =============================================================================
# 配置
# =============================================================================
RESOURCE_PATH = Path("resources")
WEIGHTS_PATH = RESOURCE_PATH / "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
BASE_CHANNELS = 64  # 必须与训练时的配置一致

# =============================================================================
# 1. 模型架构 (必须与训练代码完全一致)
# =============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False)
        )
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(concat))

class CBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)

class DoubleConvWithAttention(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_attention: bool = True):
        super().__init__()
        self.conv = nn.Sequential(ConvBNReLU(in_ch, out_ch), ConvBNReLU(out_ch, out_ch))
        self.residual = ResidualBlock(out_ch)
        self.attention = CBAM(out_ch) if use_attention else nn.Identity()
    def forward(self, x):
        x = self.conv(x)
        x = self.residual(x)
        return self.attention(x)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvWithAttention(in_ch, out_ch)
    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConvWithAttention(in_ch, out_ch)
        self.skip_attention = nn.Sequential(nn.Conv2d(in_ch // 2, in_ch // 2, 1), nn.Sigmoid())
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x2 = x2 * self.skip_attention(x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, padding=6, dilation=6, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, padding=12, dilation=12, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, padding=18, dilation=18, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(out_ch * 5, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x):
        size = x.shape[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=False)
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.fuse(out)

class UNetV2(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_ch: int = 64):
        super().__init__()
        self.inc = DoubleConvWithAttention(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)
        self.aspp = ASPP(base_ch * 16, base_ch * 16)
        self.up1 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch)
        
        # 定义辅助头以匹配权重文件 (即使推理不用)
        self.deep_out4 = nn.Conv2d(base_ch * 8, out_channels, 1)
        self.deep_out3 = nn.Conv2d(base_ch * 4, out_channels, 1)
        self.deep_out2 = nn.Conv2d(base_ch * 2, out_channels, 1)
        
        self.outc = nn.Sequential(ConvBNReLU(base_ch, base_ch // 2), nn.Conv2d(base_ch // 2, out_channels, 1))
        
    def forward(self, x, return_deep: bool = False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.aspp(x5)
        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)
        out = self.outc(d1)
        return out

# =============================================================================
# 2. 推理引擎 (包含关键的预处理修复)
# =============================================================================

class SequenceInference:
    def __init__(self, model):
        self.model = model
        self.image_size = IMAGE_SIZE
    
    @torch.no_grad()
    def predict_sequence(self, frames: np.ndarray, first_mask: np.ndarray) -> np.ndarray:
        # Input frames: (T, H, W)
        T, H, W = frames.shape
        target_size = self.image_size
        
        # --- [CRITICAL FIX] 1. 预处理: 百分位归一化 ---
        # 必须与 trackrad_unet_v2.py (lines 301-305) 完全一致
        frames_norm = frames.astype(np.float32)
        for t in range(T):
            frame = frames_norm[t]
            non_zero = frame[frame > 0]
            
            # 使用 percentile 避免极亮噪点导致的"黑图"
            if len(non_zero) > 100:
                p_low, p_high = np.percentile(non_zero, [1, 99])
                frame = np.clip(frame, p_low, p_high)
                frame = (frame - p_low) / (p_high - p_low + 1e-8)
            else:
                frame = frame / (frame.max() + 1e-8)
            frames_norm[t] = frame
        
        # --- 2. Resize: 使用 scipy.zoom (order=1) ---
        scale_h, scale_w = target_size / H, target_size / W
        frames_resized = np.zeros((T, target_size, target_size), dtype=np.float32)
        
        for t in range(T):
            frames_resized[t] = zoom(frames_norm[t], (scale_h, scale_w), order=1)
        
        first_mask_resized = zoom(first_mask.astype(float), (scale_h, scale_w), order=0)
        
        # --- 3. 序列预测循环 ---
        predictions = np.zeros((T, target_size, target_size), dtype=np.float32)
        predictions[0] = first_mask_resized
        
        prev_frame = frames_resized[0].copy()
        prev_mask = first_mask_resized.copy()
        
        for t in range(1, T):
            # 构造输入: [当前, 前一帧, 初始Mask, 前一Mask]
            input_tensor = np.stack([
                frames_resized[t], 
                prev_frame,
                first_mask_resized, 
                prev_mask
            ], axis=0)
            
            input_tensor = torch.from_numpy(input_tensor[np.newaxis]).float().to(DEVICE)
            
            # 推理
            output = self.model(input_tensor, return_deep=False)
            pred = (torch.sigmoid(output) > 0.5).cpu().numpy()[0, 0]
            
            # 后处理: 连通域 + 平滑
            pred = self._post_process(pred, prev_mask)
            
            predictions[t] = pred
            prev_frame = frames_resized[t].copy()
            prev_mask = pred.copy()
        
        # --- 4. 恢复尺寸 ---
        predictions_orig = np.zeros((T, H, W), dtype=np.uint8)
        for t in range(T):
            resized_back = zoom(predictions[t], (H / target_size, W / target_size), order=0)
            predictions_orig[t] = (resized_back > 0.5).astype(np.uint8)
        
        return predictions_orig
    
    def _post_process(self, pred: np.ndarray, prev_mask: np.ndarray) -> np.ndarray:
        if pred.sum() == 0:
            return prev_mask.copy()
        
        # 1. 保留最大连通域 (去除噪点)
        labeled, n = scipy_label(pred)
        if n > 0:
            sizes = scipy_sum(pred, labeled, range(1, n + 1))
            largest = np.argmax(sizes) + 1
            pred = (labeled == largest).astype(float)
        
        # 2. 时序平滑 (减少抖动)
        # 注意：如果发现拖影严重，可尝试调整系数为 0.9/0.1
        pred = 0.7 * pred + 0.3 * prev_mask
        pred = (pred > 0.5).astype(float)
        
        return pred

# =============================================================================
# 全局加载
# =============================================================================

_ENGINE = None

def _load_engine():
    model = UNetV2(in_channels=4, out_channels=1, base_ch=BASE_CHANNELS)
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found at {WEIGHTS_PATH}")
    
    print(f"Loading weights from {WEIGHTS_PATH}...")
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return SequenceInference(model)

# =============================================================================
# 比赛入口
# =============================================================================

def run_algorithm(frames: np.ndarray, target: np.ndarray, frame_rate: float, magnetic_field_strength: float, scanned_region: str) -> np.ndarray:
    """
    Grand Challenge Entry Point.
    Input: frames (W, H, T), target (W, H, 1)
    Output: (W, H, T) uint8
    """
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _load_engine()
        
    # [CRITICAL FIX] 维度转换: (W, H, T) -> (T, H, W)
    # PyTorch 需要 T 在第一维，且 W, H 需要转置以匹配 Numpy 视角
    frames_t = frames.transpose(2, 1, 0)
    target_first = target.transpose(2, 1, 0)[0]
    
    # 执行预测
    preds_t = _ENGINE.predict_sequence(frames_t, target_first)
    
    # 还原维度: (T, H, W) -> (W, H, T)
    final_output = preds_t.transpose(2, 1, 0)
    
    return final_output