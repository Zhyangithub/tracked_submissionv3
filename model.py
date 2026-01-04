import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom, label as scipy_label, sum as scipy_sum

# =============================================================================
# 配置与路径
# =============================================================================

# Grand Challenge 容器内的资源路径
RESOURCE_PATH = Path("resources")
WEIGHTS_PATH = RESOURCE_PATH / "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 图像尺寸必须与训练时一致 (Config.image_size = 256)
IMAGE_SIZE = 256
BASE_CHANNELS = 64  # Config.base_channels = 64

# =============================================================================
# 模型架构 (完全复制自 trackrad_unet_v2.py)
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
        
        # 即使推理时不返回deep output，这些层在权重文件中依然存在，必须定义
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
# 推理逻辑
# =============================================================================

class SequenceInference:
    """提取自训练代码的推理类"""
    def __init__(self, model):
        self.model = model
        self.image_size = IMAGE_SIZE
    
    @torch.no_grad()
    def predict_sequence(self, frames: np.ndarray, first_mask: np.ndarray) -> np.ndarray:
        # frames: (T, H, W)
        T, H, W = frames.shape
        target_size = self.image_size
        
        # --- 1. 预处理 (复用训练代码逻辑) ---
        frames_norm = frames.astype(np.float32)
        for t in range(T):
            frame = frames_norm[t]
            non_zero = frame[frame > 0]
            if len(non_zero) > 100:
                p_low, p_high = np.percentile(non_zero, [1, 99])
                frame = np.clip(frame, p_low, p_high)
                frame = (frame - p_low) / (p_high - p_low + 1e-8)
            else:
                frame = frame / (frame.max() + 1e-8)
            frames_norm[t] = frame
        
        # 调整尺寸 (使用scipy.zoom以匹配训练)
        scale_h, scale_w = target_size / H, target_size / W
        frames_resized = np.zeros((T, target_size, target_size), dtype=np.float32)
        
        for t in range(T):
            frames_resized[t] = zoom(frames_norm[t], (scale_h, scale_w), order=1)
        
        first_mask_resized = zoom(first_mask.astype(float), (scale_h, scale_w), order=0)
        
        # --- 2. 逐帧预测 ---
        predictions = np.zeros((T, target_size, target_size), dtype=np.float32)
        predictions[0] = first_mask_resized
        
        prev_frame = frames_resized[0].copy()
        prev_mask = first_mask_resized.copy()
        
        for t in range(1, T):
            # 构造4通道输入
            input_tensor = np.stack([
                frames_resized[t], 
                prev_frame,
                first_mask_resized, 
                prev_mask
            ], axis=0)
            
            # 转Tensor
            input_tensor = torch.from_numpy(input_tensor[np.newaxis]).float().to(DEVICE)
            
            # 推理
            output = self.model(input_tensor, return_deep=False)
            pred = (torch.sigmoid(output) > 0.5).cpu().numpy()[0, 0]
            
            # 后处理
            pred = self._post_process(pred, prev_mask)
            
            predictions[t] = pred
            prev_frame = frames_resized[t].copy()
            prev_mask = pred.copy()
        
        # --- 3. 恢复原始尺寸 ---
        predictions_orig = np.zeros((T, H, W), dtype=np.uint8)
        for t in range(T):
            resized_back = zoom(predictions[t], (H / target_size, W / target_size), order=0)
            predictions_orig[t] = (resized_back > 0.5).astype(np.uint8)
        
        return predictions_orig
    
    def _post_process(self, pred: np.ndarray, prev_mask: np.ndarray) -> np.ndarray:
        if pred.sum() == 0:
            return prev_mask.copy()
        
        # 保留最大连通域
        labeled, n = scipy_label(pred)
        if n > 0:
            sizes = scipy_sum(pred, labeled, range(1, n + 1))
            largest = np.argmax(sizes) + 1
            pred = (labeled == largest).astype(float)
        
        # 时序平滑
        pred = 0.7 * pred + 0.3 * prev_mask
        pred = (pred > 0.5).astype(float)
        
        return pred

# =============================================================================
# 全局模型加载 (Lazy Loading)
# =============================================================================

_ENGINE = None

def _load_engine():
    # 实例化模型，必须与训练配置一致 (base_ch=64)
    model = UNetV2(in_channels=4, out_channels=1, base_ch=BASE_CHANNELS)
    
    if not WEIGHTS_PATH.exists():
        # 用于本地调试的错误提示
        raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}")
    
    # 加载权重
    print(f"Loading weights from {WEIGHTS_PATH}...")
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    return SequenceInference(model)

# =============================================================================
# 比赛入口函数
# =============================================================================

def run_algorithm(frames: np.ndarray, target: np.ndarray, frame_rate: float, magnetic_field_strength: float, scanned_region: str) -> np.ndarray:
    """
    Grand Challenge Entry Point.
    Args:
        frames: (W, H, T)
        target: (W, H, 1)
    Returns:
        (W, H, T) uint8
    """
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _load_engine()
        
    # 1. 维度转换: (W, H, T) -> (T, H, W)
    # Grand Challenge 输入是 W,H,T，但 PyTorch/Scipy 通常处理 T,H,W (或 H,W)
    frames_t = frames.transpose(2, 1, 0)
    
    # 提取第一帧掩码: (W, H, 1) -> (1, H, W) -> 取[0]得到(H, W)
    target_first = target.transpose(2, 1, 0)[0]
    
    # 2. 执行推理
    # 结果 shape: (T, H, W)
    preds_t = _ENGINE.predict_sequence(frames_t, target_first)
    
    # 3. 维度还原: (T, H, W) -> (W, H, T)
    final_output = preds_t.transpose(2, 1, 0)
    
    return final_output