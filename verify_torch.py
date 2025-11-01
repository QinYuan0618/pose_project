import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    # 测试GPU运算
    x = torch.randn(1000, 1000).cuda()
    print("GPU测试成功！")
