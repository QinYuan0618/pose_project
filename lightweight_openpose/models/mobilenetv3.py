import torch
import torch.nn as nn
import torch.nn.functional as F
from ema_attentions import EMA

class HSwish(nn.Module):
    """Hard Swish激活函数"""
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

class HSigmoid(nn.Module):
    """Hard Sigmoid激活函数"""
    def forward(self, x):
        return F.relu6(x + 3, inplace=True) / 6

class SEModule(nn.Module):
    """Squeeze-and-Excitation模块"""
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            HSigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = HSwish()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return x

class InvertedResidual(nn.Module):
    """MobileNetV3的倒残差模块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, use_se=False, use_hs=True):
        super(InvertedResidual, self).__init__()
        # hidden_dim = in_channels * expand_ratio
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # 扩展
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(HSwish() if use_hs else nn.ReLU(inplace=True))
        
        # 深度卷积
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            HSwish() if use_hs else nn.ReLU(inplace=True)
        ])
        
        # SE模块
        if use_se:
            layers.append(SEModule(hidden_dim))
        
        # 投影
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3_Small(nn.Module):
    """
    MobileNetV3-Small作为OpenPose主干
    适配姿态估计任务，输出多尺度特征
    """
    def __init__(self, use_ema=False):
        super(MobileNetV3_Small, self).__init__()
        self.use_ema = use_ema
        
        # 初始层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            HSwish()
        )
        
        # MobileNetV3 Blocks
        self.bneck = nn.Sequential(
            InvertedResidual(16, 16, 3, 2, 1, use_se=True),      # /4
            InvertedResidual(16, 24, 3, 2, 4.5, use_se=False),   # /8
            InvertedResidual(24, 24, 3, 1, 3.67, use_se=False),
            InvertedResidual(24, 40, 5, 2, 4, use_se=True),      # /16
            InvertedResidual(40, 40, 5, 1, 6, use_se=True),
            InvertedResidual(40, 40, 5, 1, 6, use_se=True),
            InvertedResidual(40, 48, 5, 1, 3, use_se=True),
            InvertedResidual(48, 48, 5, 1, 3, use_se=True),
            # InvertedResidual(48, 96, 5, 2, 6, use_se=True),      # /32
            InvertedResidual(48, 96, 5, 1, 6, use_se=True),      # /16 
            InvertedResidual(96, 96, 5, 1, 6, use_se=True),
            InvertedResidual(96, 96, 5, 1, 6, use_se=True),
        )
        
        # EMA注意力模块
        if use_ema:
            self.ema1 = EMA(24)
            self.ema2 = EMA(48)
            self.ema3 = EMA(96)
    
    def forward(self, x):
        features = []
        
        x = self.conv1(x)
        
        # 提取多尺度特征
        for i, block in enumerate(self.bneck):
            x = block(x)
            
            # 在关键层添加EMA
            if self.use_ema:
                if i == 2:  # /8特征
                    x = self.ema1(x)
                    features.append(x)
                elif i == 7:  # /16特征
                    x = self.ema2(x)
                    features.append(x)
                elif i == 10:  # /32特征
                    x = self.ema3(x)
                    features.append(x)
            else:
                if i in [2, 7, 10]:
                    features.append(x)
        
        return features

# 测试代码
if __name__ == "__main__":
    # 测试不带EMA的MobileNetV3
    model = MobileNetV3_Small(use_ema=False)
    x = torch.randn(1, 3, 368, 368)
    features = model(x)
    
    print("MobileNetV3-Small 输出:")
    for i, f in enumerate(features):
        print(f"  特征{i+1}: {f.shape}")
    
    # 计算参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"\n参数量: {params:,}")
    
    # 测试带EMA的版本
    print("\n" + "="*50)
    model_ema = MobileNetV3_Small(use_ema=True)
    features_ema = model_ema(x)
    
    print("MobileNetV3-Small + EMA 输出:")
    for i, f in enumerate(features_ema):
        print(f"  特征{i+1}: {f.shape}")
    
    params_ema = sum(p.numel() for p in model_ema.parameters())
    print(f"\n参数量: {params_ema:,}")