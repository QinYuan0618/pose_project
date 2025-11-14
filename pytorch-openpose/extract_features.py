# 根据标注后的json文件提取特征数据
import json
import numpy as np
import math

# 读取标注
json_path = r"D:\VSCode\PYCode\pose_project\pytorch-openpose\project-1-at-2025-11-04-22-30-f9e2c372.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 准备存储
features_list = []
labels_list = []

for task in data:
    if 'annotations' not in task or len(task['annotations']) == 0:
        continue
    
    # 获取对应的关键点数据
    img_name = task['data']['image'].split('/')[-1]
    # 去掉Label Studio添加的UUID前缀（格式：xxxxxxxx-原始文件名）
    if '-' in img_name:
        img_name = img_name.split('-', 1)[1]  # 只去掉第一个-之前的部分
    npy_name = img_name.replace('.jpg', '.npy')
    npy_path = f"posture_samples/{npy_name}"
    
    try:
        keypoint_data = np.load(npy_path, allow_pickle=True).item()
        candidate = keypoint_data['candidate']
        subset = keypoint_data['subset']
    except:
        print(f"跳过: {img_name}")
        continue
    
    # 遍历该图片的所有标注框
    annotations = task['annotations'][0]['result']
    
    for ann in annotations:
        label = ann['value']['rectanglelabels'][0]
        
        # 获取边框中心点（用于匹配最近的检测人体）
        box = ann['value']
        box_center_x = (box['x'] + box['width'] / 2) * task['annotations'][0]['result'][0]['original_width'] / 100
        box_center_y = (box['y'] + box['height'] / 2) * task['annotations'][0]['result'][0]['original_height'] / 100
        
        # 找到最近的人体（通过鼻子位置）
        min_dist = float('inf')
        best_person = None
        
        for person in subset.astype(int):
            nose_idx = person[0]
            if nose_idx == -1:
                continue
            nose_x, nose_y = candidate[nose_idx][:2]
            dist = math.sqrt((nose_x - box_center_x)**2 + (nose_y - box_center_y)**2)
            if dist < min_dist:
                min_dist = dist
                best_person = person
        
        if best_person is None or min_dist > 200:  # 距离太远就跳过
            continue
        
        # 提取关键点
        person = best_person
        keypoints = {}
        for i in range(8):  # 只用上半身0-7
            idx = person[i]
            if idx != -1:
                keypoints[i] = candidate[idx][:2]
        
        # 检查必要关键点是否存在
        if 1 not in keypoints or 2 not in keypoints or 5 not in keypoints:
            continue
        
        features_list.append(keypoints)
        labels_list.append(label)

print(f"成功提取 {len(features_list)} 个样本")
print(f"各类别数量: {len(set(labels_list))}")

# 保存
np.save('training_data.npy', {
    'features': features_list,
    'labels': labels_list
})
print("已保存到 training_data.npy")