import cv2
import numpy as np
import joblib
from src.body import Body
from src import util
import sys
sys.path.append('.')
from design_features import extract_geometric_features

# 加载模型
print("加载模型...")
body_estimation = Body('model/body_pose_model.pth')
clf = joblib.load('posture_classifier_final.pkl')
print("✓ 模型加载完成\n")

# 视频路径
VIDEO_PATH = r"D:\VSCode\PYCode\pose_project\videos\1.mp4"

# 打开视频
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"视频信息: {width}x{height}, {fps}FPS, {total_frames}帧")

# 输出视频
out = cv2.VideoWriter('output_posture.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                      fps, (width, height))

# 颜色映射
color_map = {
    'Standard': (0, 255, 0),
    'Slouching': (0, 165, 255),
    'Severe Forward/Lying': (0, 0, 255),
    'Leaning': (255, 255, 0)
}

frame_count = 0
print("\n开始处理...")

while frame_count < 100:  # 先只处理前100帧测试
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count % 10 == 0:
        print(f"处理第 {frame_count} 帧...")
    
    # 检测骨架
    candidate, subset = body_estimation(frame)
    
    # 画骨架
    canvas = util.draw_bodypose(frame, candidate, subset)
    
    # 分类并标注
    for person in subset.astype(int):
        keypoints = {}
        for j in range(8):
            idx = person[j]
            if idx != -1:
                keypoints[j] = candidate[idx][:2]
        
        features = extract_geometric_features(keypoints)
        if features is None:
            continue
        
        features = np.array(features).reshape(1, -1)
        posture = clf.predict(features)[0]
        
        # 标注
        nose_idx = person[0]
        if nose_idx != -1:
            x, y = int(candidate[nose_idx][0]), int(candidate[nose_idx][1])
            color = color_map.get(posture, (255, 255, 255))
            cv2.putText(canvas, posture, (x-60, y-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    out.write(canvas)

cap.release()
out.release()

print(f"\n完成！处理了 {frame_count} 帧")
print("输出文件: output_posture.mp4")