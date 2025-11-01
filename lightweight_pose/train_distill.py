import torch
import torch.nn as nn
import torch.optim as optim
from mobilenet_openpose import LightweightOpenPose
import numpy as np

class KnowledgeDistillationLoss(nn.Module):
    """知识蒸馏损失函数"""
    def __init__(self, alpha=0.7, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.mse = nn.MSELoss()
        
    def forward(self, student_pafs, student_heatmaps, teacher_pafs, teacher_heatmaps):
        loss = 0
        
        # PAF蒸馏
        for s_paf, t_paf in zip(student_pafs, teacher_pafs):
            loss += self.mse(s_paf / self.T, t_paf / self.T)
            
        # Heatmap蒸馏
        for s_heat, t_heat in zip(student_heatmaps, teacher_heatmaps):
            loss += self.mse(s_heat / self.T, t_heat / self.T)
            
        return loss * (self.T ** 2)

def train_step(model, teacher_model, data, optimizer, criterion):
    """单步训练"""
    model.train()
    teacher_model.eval()
    
    images = data
    
    # 教师模型预测
    with torch.no_grad():
        teacher_pafs, teacher_heatmaps = teacher_model(images)
    
    # 学生模型预测
    student_pafs, student_heatmaps = model(images)
    
    # 计算损失
    loss = criterion(student_pafs, student_heatmaps, teacher_pafs, teacher_heatmaps)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# 主训练代码
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化模型
    student_model = LightweightOpenPose().to(device)
    
    # 暂时用同样的模型做教师（实际应该用HRNet）
    teacher_model = LightweightOpenPose().to(device)
    teacher_model.eval()
    
    # 优化器和损失函数
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)
    criterion = KnowledgeDistillationLoss()
    
    # 模拟训练
    print("开始训练...")
    for epoch in range(5):
        dummy_data = torch.randn(2, 3, 368, 368).to(device)
        
        loss = train_step(student_model, teacher_model, dummy_data, optimizer, criterion)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    # 保存模型
    torch.save(student_model.state_dict(), "lightweight_openpose_distilled.pth")
    print("模型已保存!")
