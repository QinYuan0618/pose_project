import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilenetv3 import MobileNetV3_Small
from ema_attentions import EMA

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积 - 用于优化初始阶段"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class OptimizedInitialStage(nn.Module):
    """
    优化的初始阶段
    - 合并前3层3×3卷积
    - 使用深度可分离卷积
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 原本是3个3x3卷积，现在合并为1个深度可分离卷积
        self.conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels // 2, 3, 1, 1),
            DepthwiseSeparableConv(out_channels // 2, out_channels, 3, 1, 1),
        )
    
    def forward(self, x):
        return self.conv(x)

class OptimizedRefinementStage(nn.Module):
    """
    优化的细化阶段
    - 合并前5层7×7卷积
    - 用3个3×3卷积替换7×7卷积（感受野相同，参数更少）
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        
        # 用3个3x3替换7x7
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class PoseEstimationStage(nn.Module):
    """
    姿态估计阶段
    输出关键点热图(heatmap)和部分亲和场(PAF)
    """
    def __init__(self, in_channels, out_channels_heatmap=19, out_channels_paf=38):
        super().__init__()
        
        # 热图分支
        self.heatmap_conv = nn.Sequential(
            OptimizedRefinementStage(in_channels, 128),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels_heatmap, 1, 1, 0)
        )
        
        # PAF分支
        self.paf_conv = nn.Sequential(
            OptimizedRefinementStage(in_channels, 128),
            nn.Conv2d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels_paf, 1, 1, 0)
        )
    
    def forward(self, x):
        heatmap = self.heatmap_conv(x)
        paf = self.paf_conv(x)
        return heatmap, paf

class LightweightOpenPose(nn.Module):
    """
    轻量化OpenPose模型
    - 主干: MobileNetV3-Small + EMA
    - 初始阶段: 优化的深度可分离卷积
    - 细化阶段: 3x3替换7x7
    """
    def __init__(self, use_ema=True, num_stages=6):
        super().__init__()
        self.num_stages = num_stages
        
        # 主干网络
        self.backbone = MobileNetV3_Small(use_ema=use_ema)
        
        # 初始阶段（接收主干的最后一个特征）
        self.initial_stage = OptimizedInitialStage(96, 128)
        
        # 第一个预测阶段
        self.stage1 = PoseEstimationStage(128)
        
        # 后续细化阶段
        self.refinement_stages = nn.ModuleList([
            PoseEstimationStage(128 + 19 + 38)  # 前一阶段特征 + 前一阶段输出
            for _ in range(num_stages - 1)
        ])
    
    def forward(self, x):
        # 提取主干特征
        features = self.backbone(x)
        x = features[-1]  # 使用最深层特征
        
        # 初始阶段
        x = self.initial_stage(x)
        
        # 第一个预测阶段
        heatmap, paf = self.stage1(x)
        
        heatmaps = [heatmap]
        pafs = [paf]
        
        # 细化阶段
        for stage in self.refinement_stages:
            # 拼接特征
            x_concat = torch.cat([x, heatmap, paf], dim=1)
            heatmap, paf = stage(x_concat)
            heatmaps.append(heatmap)
            pafs.append(paf)
        
        return heatmaps[-1], pafs[-1]  # 返回最后一个阶段的输出

# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("轻量化OpenPose模型测试")
    print("="*60)
    
    # 创建模型
    model = LightweightOpenPose(use_ema=True, num_stages=6)
    
    # 测试输入
    x = torch.randn(1, 3, 368, 368)
    
    # 前向传播
    heatmap, paf = model(x)
    
    print(f"\n输入: {x.shape}")
    print(f"输出热图: {heatmap.shape}  (应该是 [1, 19, ?, ?])")
    print(f"输出PAF: {paf.shape}  (应该是 [1, 38, ?, ?])")
    
    # 计算参数量和模型大小
    params = sum(p.numel() for p in model.parameters())
    params_mb = params * 4 / (1024 ** 2)  # 假设float32
    
    print(f"\n模型统计:")
    print(f"  参数量: {params:,}")
    print(f"  模型大小: {params_mb:.2f} MB")
    
    # 对比原OpenPose
    original_params = 26000000  # 约26M
    compression_ratio = original_params / params
    
    print(f"\n与原OpenPose对比:")
    print(f"  原模型参数: {original_params:,}")
    print(f"  压缩比: {compression_ratio:.1f}x")