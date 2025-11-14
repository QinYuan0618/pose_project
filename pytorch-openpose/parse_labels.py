# 显示label studio导出的json文件的标注信息统计结果
import json
import numpy as np

# 读取标注文件
json_path = r"D:\VSCode\PYCode\pose_project\pytorch-openpose\project-1-at-2025-11-04-16-13-5b7437ea.json"

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"总任务数: {len(data)}")

# 统计标注信息
label_counts = {}
total_annotations = 0

for task in data:
    if 'annotations' not in task or len(task['annotations']) == 0:
        continue
    
    annotations = task['annotations'][0]['result']
    total_annotations += len(annotations)
    
    for ann in annotations:
        label = ann['value']['rectanglelabels'][0]
        label_counts[label] = label_counts.get(label, 0) + 1

print(f"\n总标注框数: {total_annotations}")
print("\n各类别数量:")
for label, count in sorted(label_counts.items()):
    print(f"  {label}: {count}")