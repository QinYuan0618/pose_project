import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 - 减少参数量的关键"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class EMAAttention(nn.Module):
    """EMA注意力机制 - 提升特征表达能力"""
    def __init__(self, channels, factor=8):
        super().__init__()
        self.channels = channels
        self.factor = factor
        self.conv = nn.Conv2d(channels, channels, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y)
        y = torch.sigmoid(y)
        return x * y

class MobileNetV3Block(nn.Module):
    """MobileNetV3基础块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_ema=True):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, kernel_size//2)
        self.use_ema = use_ema
        if use_ema:
            self.ema = EMAAttention(out_channels)
            
    def forward(self, x):
        x = self.conv(x)
        if self.use_ema:
            x = self.ema(x)
        return x

class LightweightOpenPose(nn.Module):
    """轻量级OpenPose with MobileNetV3 + EMA"""
    def __init__(self, num_joints=18):
        super().__init__()
        
        # Stage 1: 初始特征提取（改进版）
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(3, 64, 3, 2, 1),
            DepthwiseSeparableConv(64, 128, 3, 1, 1),
            DepthwiseSeparableConv(128, 128, 3, 1, 1),
        )
        
        # Stage 2-6: 使用MobileNetV3块 + EMA
        self.stage2 = MobileNetV3Block(128, 256, 3, 1, True)
        self.stage3 = MobileNetV3Block(256, 256, 3, 1, True)
        self.stage4 = MobileNetV3Block(256, 512, 3, 1, True)
        self.stage5 = MobileNetV3Block(512, 512, 3, 1, True)
        self.stage6 = MobileNetV3Block(512, 512, 3, 1, True)
        
        # PAF阶段
        self.paf_stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 256, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 38, 1)  # 19个PAF x 2维向量
            ) for _ in range(2)
        ])
        
        # 关键点热图阶段
        self.heatmap_stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512 + 38, 256, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_joints + 1, 1)  # 18关键点+1背景
            ) for _ in range(2)
        ])
        
    def forward(self, x):
        # 特征提取
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        features = self.stage6(x)
        
        # PAF预测
        pafs = []
        paf = self.paf_stages[0](features)
        pafs.append(paf)
        
        for paf_stage in self.paf_stages[1:]:
            paf = paf_stage(features)
            pafs.append(paf)
        
        # 热图预测
        heatmaps = []
        concat_features = torch.cat([features, pafs[-1]], 1)
        
        for heatmap_stage in self.heatmap_stages:
            heatmap = heatmap_stage(concat_features)
            heatmaps.append(heatmap)
            
        return pafs, heatmaps

# 测试模型
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightOpenPose().to(device)
    x = torch.randn(1, 3, 368, 368).to(device)
    pafs, heatmaps = model(x)
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params/1e6:.2f}M")
    print(f"PAF输出: {[p.shape for p in pafs]}")
    print(f"Heatmap输出: {[h.shape for h in heatmaps]}")
    
    # 对比原始VGG-19参数量（约143M）
    print(f"参数量减少: {(143 - total_params/1e6)/143*100:.1f}%")
