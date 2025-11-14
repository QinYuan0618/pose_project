import cv2
import numpy as np
import joblib
from src.body import Body
from src import util
import sys
sys.path.append('.')
from design_features import extract_geometric_features

# 加载模型和分类器
print("加载模型...")
body_estimation = Body('model/body_pose_model.pth')
clf = joblib.load('posture_classifier_final.pkl')
print("✓ 加载完成\n")

# 读取测试视频的一帧
VIDEO_PATH = r"D:\VSCode\PYCode\pose_project\videos\1.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
cap.release()

# 检测骨架
print("检测骨架...")
candidate, subset = body_estimation(frame)
print(f"检测到 {len(subset)} 个人\n")

# 画骨架
canvas = util.draw_bodypose(frame, candidate, subset)

# 定义颜色映射
color_map = {
    'Standard': (0, 255, 0),           # 绿色
    'Slouching': (0, 165, 255),        # 橙色
    'Severe Forward/Lying': (0, 0, 255),  # 红色
    'Leaning': (255, 255, 0)           # 青色
}

# 对每个人进行分类并标注
for i, person in enumerate(subset.astype(int)):
    # 提取关键点
    keypoints = {}
    for j in range(8):
        idx = person[j]
        if idx != -1:
            keypoints[j] = candidate[idx][:2]
    
    # 提取特征
    features = extract_geometric_features(keypoints)
    if features is None:
        continue
    
    # 预测坐姿
    features = np.array(features).reshape(1, -1)
    posture = clf.predict(features)[0]
    
    # 获取头部位置用于标注
    nose_idx = person[0]
    if nose_idx != -1:
        x, y = int(candidate[nose_idx][0]), int(candidate[nose_idx][1])
        color = color_map.get(posture, (255, 255, 255))
        
        # 显示分类结果
        cv2.putText(canvas, posture, (x-60, y-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 保存结果
cv2.imwrite('posture_classification_result.jpg', canvas)
print("结果已保存到 posture_classification_result.jpg")