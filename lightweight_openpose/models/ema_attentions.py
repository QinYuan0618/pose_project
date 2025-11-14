import torch
import torch.nn as nn
import torch.nn.functional as F

class EMA(nn.Module):
    """
    Efficient Multi-Scale Attention (EMA) 模块
    用于增强特征表达能力和全局上下文建模
    """
    def __init__(self, channels, reduction=4, num_groups=8):
        super(EMA, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.num_groups = num_groups
        
        # 分组卷积用于特征压缩
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 通道注意力
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        # 空间注意力
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
        # 多尺度特征融合
        self.group_conv = nn.Conv2d(channels, channels, kernel_size=1, groups=num_groups, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        # 通道注意力分支
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x_channel = x * channel_att.expand_as(x)
        
        # 空间注意力分支
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid(self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1)))
        x_spatial = x * spatial_att
        
        # 多尺度特征融合
        x_group = self.group_conv(x)
        
        # 融合三个分支
        out = x_channel + x_spatial + x_group
        
        return out

# 测试代码
if __name__ == "__main__":
    # 测试EMA模块
    x = torch.randn(1, 64, 56, 56)
    ema = EMA(channels=64)
    out = ema(x)
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
    
    # 计算参数量
    params = sum(p.numel() for p in ema.parameters())
    print(f"参数量: {params:,}")