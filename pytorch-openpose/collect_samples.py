import cv2
import numpy as np
from src.body import Body
from src import util
import os
import random

# 加载模型
body_estimation = Body('model/body_pose_model.pth')

# 4个视频路径
VIDEO_DIR = r"D:\VSCode\PYCode\pose_project\videos"
video_files = [f"{i}.mp4" for i in range(1, 5)]

# 清空旧样本（可选，如果想保留旧的就注释掉下面两行）
import shutil
shutil.rmtree("posture_samples", ignore_errors=True)

os.makedirs("posture_samples", exist_ok=True)

sample_count = 0
per_video = 25  # 每个视频抽25个样本
frame_interval = 100  # 每100帧抽一次

print("从4个视频中提取样本...")

for video_file in video_files:
    video_path = os.path.join(VIDEO_DIR, video_file)
    print(f"\n处理: {video_file}")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 生成采样帧号（均匀分布）
    sample_frames = list(range(0, total_frames, frame_interval))
    random.shuffle(sample_frames)  # 随机打乱
    sample_frames = sample_frames[:per_video]  # 只取25个
    
    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 检测
        candidate, subset = body_estimation(frame)
        
        # 画骨架
        canvas = util.draw_bodypose(frame, candidate, subset)
        
        # 保存
        img_path = f"posture_samples/sample_{sample_count:03d}_v{video_file[0]}.jpg"
        data_path = f"posture_samples/sample_{sample_count:03d}_v{video_file[0]}.npy"
        
        cv2.imwrite(img_path, canvas)
        np.save(data_path, {'candidate': candidate, 'subset': subset})
        
        sample_count += 1
        print(f"  已保存 {sample_count}/100")
    
    cap.release()

print(f"\n完成！共 {sample_count} 个样本")