import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pytorch-openpose/src")))

from improved_models import ImprovedOpenPose

from model import bodypose_model

try:
    from thop import profile, clever_format
except ImportError:
    print("请先安装thop: pip install thop")
    exit()

def compute_flops(model, input_size=(1, 3, 368, 368)):
    """计算模型的GFLOPs和参数量"""
    model.eval()
    x = torch.randn(input_size)
    
    flops, params = profile(model, inputs=(x,), verbose=False)
    flops_g = flops / 1e9  # 转换为GFLOPs
    params_m = params / 1e6  # 转换为M
    
    return flops_g, params_m

print("="*70)
print("OpenPose模型复杂度对比")
print("="*70)

# 原始OpenPose
print("\n【原始OpenPose】")
original = bodypose_model()
flops_o, params_o = compute_flops(original)
print(f"  参数量: {params_o:.2f}M")
print(f"  GFLOPs: {flops_o:.2f}")

# 改进版本
print("\n【改进版OpenPose (v4)】")
improved = ImprovedOpenPose(use_mobilenet=True, use_depthwise=True, optimize_7x7=True)
flops_i, params_i = compute_flops(improved)
print(f"  参数量: {params_i:.2f}M")
print(f"  GFLOPs: {flops_i:.2f}")

# 对比
print("\n" + "="*70)
print("对比分析")
print("="*70)
print(f"参数量压缩: {(1 - params_i/params_o)*100:.1f}%")
print(f"计算量压缩: {(1 - flops_i/flops_o)*100:.1f}%")

# 与EffiPoseNet对比
print("\n与EffiPoseNet对比:")
print(f"  EffiPoseNet: 3.68M, 22.73 GFLOPs")
print(f"  你的模型:    {params_i:.2f}M, {flops_i:.2f} GFLOPs")