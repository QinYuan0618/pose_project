import numpy as np
import math

def extract_geometric_features(keypoints):
    """从8个关键点提取完整特征"""
    features = []
    
    # 获取所有关键点
    nose = keypoints.get(0)
    neck = keypoints.get(1)
    r_shoulder = keypoints.get(2)
    r_elbow = keypoints.get(3)
    r_wrist = keypoints.get(4)
    l_shoulder = keypoints.get(5)
    l_elbow = keypoints.get(6)
    l_wrist = keypoints.get(7)
    
    # 检查必要点
    if nose is None or neck is None or r_shoulder is None or l_shoulder is None:
        return None
    
    # 计算肩宽作为归一化基准
    shoulder_width = math.sqrt((r_shoulder[0]-l_shoulder[0])**2 + (r_shoulder[1]-l_shoulder[1])**2)
    if shoulder_width < 10:
        return None
    
    # === 头颈特征（4个）===
    
    # 1. 头颈水平偏移比
    head_offset_x = abs(nose[0] - neck[0]) / shoulder_width
    features.append(head_offset_x)
    
    # 2. 头颈垂直偏移比（俯视角度重要）
    head_offset_y = (nose[1] - neck[1]) / shoulder_width
    features.append(head_offset_y)
    
    # 3. 头颈距离比
    head_neck_dist = math.sqrt((nose[0]-neck[0])**2 + (nose[1]-neck[1])**2) / shoulder_width
    features.append(head_neck_dist)
    
    # 4. 头颈角度
    head_angle = math.atan2(nose[1]-neck[1], nose[0]-neck[0])
    features.append(head_angle)
    
    # === 肩部特征（3个）===
    
    # 5. 肩膀高度差比
    shoulder_height_diff = abs(r_shoulder[1] - l_shoulder[1]) / shoulder_width
    features.append(shoulder_height_diff)
    
    # 6. 肩膀倾斜角度
    shoulder_angle = math.atan2(r_shoulder[1]-l_shoulder[1], r_shoulder[0]-l_shoulder[0])
    features.append(shoulder_angle)
    
    # 7. 颈部相对肩膀中心的前后位置（Y方向）
    shoulder_center_y = (r_shoulder[1] + l_shoulder[1]) / 2
    neck_forward = (neck[1] - shoulder_center_y) / shoulder_width
    features.append(neck_forward)
    
    # === 手臂特征（8个）===
    
    # 8-9. 肩-肘距离（左右）
    if r_elbow is not None:
        r_shoulder_elbow_dist = math.sqrt((r_shoulder[0]-r_elbow[0])**2 + (r_shoulder[1]-r_elbow[1])**2) / shoulder_width
        features.append(r_shoulder_elbow_dist)
    else:
        features.append(0)
    
    if l_elbow is not None:
        l_shoulder_elbow_dist = math.sqrt((l_shoulder[0]-l_elbow[0])**2 + (l_shoulder[1]-l_elbow[1])**2) / shoulder_width
        features.append(l_shoulder_elbow_dist)
    else:
        features.append(0)
    
    # 10-11. 肘部前移程度（Y方向相对肩膀）
    if r_elbow is not None:
        r_elbow_forward = (r_elbow[1] - r_shoulder[1]) / shoulder_width
        features.append(r_elbow_forward)
    else:
        features.append(0)
    
    if l_elbow is not None:
        l_elbow_forward = (l_elbow[1] - l_shoulder[1]) / shoulder_width
        features.append(l_elbow_forward)
    else:
        features.append(0)
    
    # 12-13. 肘-腕距离（检测手臂伸展度）
    if r_elbow is not None and r_wrist is not None:
        r_elbow_wrist_dist = math.sqrt((r_elbow[0]-r_wrist[0])**2 + (r_elbow[1]-r_wrist[1])**2) / shoulder_width
        features.append(r_elbow_wrist_dist)
    else:
        features.append(0)
    
    if l_elbow is not None and l_wrist is not None:
        l_elbow_wrist_dist = math.sqrt((l_elbow[0]-l_wrist[0])**2 + (l_elbow[1]-l_wrist[1])**2) / shoulder_width
        features.append(l_elbow_wrist_dist)
    else:
        features.append(0)
    
    # 14-15. 肩-肘-腕角度（检测手臂弯曲）
    if r_shoulder is not None and r_elbow is not None and r_wrist is not None:
        vec1 = [r_shoulder[0]-r_elbow[0], r_shoulder[1]-r_elbow[1]]
        vec2 = [r_wrist[0]-r_elbow[0], r_wrist[1]-r_elbow[1]]
        dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]
        det = vec1[0]*vec2[1] - vec1[1]*vec2[0]
        r_arm_angle = math.atan2(det, dot)
        features.append(r_arm_angle)
    else:
        features.append(0)
    
    if l_shoulder is not None and l_elbow is not None and l_wrist is not None:
        vec1 = [l_shoulder[0]-l_elbow[0], l_shoulder[1]-l_elbow[1]]
        vec2 = [l_wrist[0]-l_elbow[0], l_wrist[1]-l_elbow[1]]
        dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]
        det = vec1[0]*vec2[1] - vec1[1]*vec2[0]
        l_arm_angle = math.atan2(det, dot)
        features.append(l_arm_angle)
    else:
        features.append(0)
    
    # === 对称性特征（2个）===
    
    # 16. 左右肩-肘距离差异（检测不对称）
    if r_elbow is not None and l_elbow is not None:
        r_se_dist = math.sqrt((r_shoulder[0]-r_elbow[0])**2 + (r_shoulder[1]-r_elbow[1])**2)
        l_se_dist = math.sqrt((l_shoulder[0]-l_elbow[0])**2 + (l_shoulder[1]-l_elbow[1])**2)
        arm_symmetry = abs(r_se_dist - l_se_dist) / shoulder_width
        features.append(arm_symmetry)
    else:
        features.append(0)
    
    # 17. 左右肘部前移差异（检测侧倾）
    if r_elbow is not None and l_elbow is not None:
        r_forward = r_elbow[1] - r_shoulder[1]
        l_forward = l_elbow[1] - l_shoulder[1]
        forward_diff = abs(r_forward - l_forward) / shoulder_width
        features.append(forward_diff)
    else:
        features.append(0)
    
    return features

# 加载数据
data = np.load('training_data.npy', allow_pickle=True).item()
keypoints_list = data['features']
labels_list = data['labels']

# 提取特征
X = []
y = []

for keypoints, label in zip(keypoints_list, labels_list):
    feat = extract_geometric_features(keypoints)
    if feat is not None:
        X.append(feat)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"特征矩阵: {X.shape}")
print(f"标签数量: {len(y)}")
print(f"\n特征说明（共{X.shape[1]}维）:")
print("1-4: 头颈特征")
print("5-7: 肩部特征")
print("8-15: 手臂特征")
print("16-17: 对称性特征")

# 保存
np.save('features_labels.npy', {'X': X, 'y': y})
print("\n已保存到 features_labels.npy")