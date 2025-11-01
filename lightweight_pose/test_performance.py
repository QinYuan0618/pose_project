import torch
import time
from mobilenet_openpose import LightweightOpenPose

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = LightweightOpenPose().to(device)
model.load_state_dict(torch.load("lightweight_openpose_distilled.pth"))
model.eval()

print("="*50)
print("模型性能测试")
print("="*50)

# 不同输入尺寸测试
test_sizes = [(256, 256), (368, 368), (512, 512)]

for size in test_sizes:
    x = torch.randn(1, 3, size[0], size[1]).to(device)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    # 测速
    torch.cuda.synchronize()
    start = time.time()
    
    iterations = 100
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(x)
    
    torch.cuda.synchronize()
    end = time.time()
    
    fps = iterations / (end - start)
    ms_per_frame = 1000 / fps
    
    print(f"\n输入尺寸: {size}")
    print(f"  FPS: {fps:.2f}")
    print(f"  延迟: {ms_per_frame:.2f} ms/帧")
    print(f"  实时处理: {'✓ 可以' if fps > 25 else '✗ 不能'}")

# 计算模型大小
torch.save(model.state_dict(), "model_temp.pth")
import os
model_size = os.path.getsize("model_temp.pth") / (1024*1024)
print(f"\n模型文件大小: {model_size:.2f} MB")
os.remove("model_temp.pth")

print("\n对比原始OpenPose:")
print("  原始模型: ~200MB, ~15 FPS")
print("  我们的模型: {:.2f}MB, {:.2f} FPS".format(model_size, fps))
print("  改进: {:.1f}x 更小, {:.1f}x 更快".format(200/model_size, fps/15))
