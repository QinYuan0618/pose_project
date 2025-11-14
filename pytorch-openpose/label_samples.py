# 样本标注工具
import cv2
import os
import json

# 定义类别
CATEGORIES = {
    '1': '正常坐姿',
    '2': '头部前倾',
    '3': '趴桌子',
    '4': '歪头',
    '0': '跳过（检测失败）'
}

# 加载已有标注（如果存在）
label_file = "posture_samples/labels.json"
if os.path.exists(label_file):
    with open(label_file, 'r', encoding='utf-8') as f:
        labels = json.load(f)
else:
    labels = {}

print("="*50)
print("坐姿标注工具")
print("="*50)
for key, value in CATEGORIES.items():
    print(f"按 {key} - {value}")
print("按 q - 退出并保存")
print("="*50)

# 获取所有样本
samples = sorted([f for f in os.listdir("posture_samples") if f.endswith('.jpg')])
start_idx = len(labels)  # 从未标注的开始

for i in range(start_idx, len(samples)):
    img_name = samples[i]
    img_path = os.path.join("posture_samples", img_name)
    
    img = cv2.imread(img_path)
    cv2.imshow('Label Tool', img)
    
    print(f"\n[{i+1}/{len(samples)}] {img_name}")
    
    key = cv2.waitKey(0) & 0xFF
    key_char = chr(key)
    
    if key_char == 'q':
        print("退出标注")
        break
    
    if key_char in CATEGORIES:
        labels[img_name] = {
            'category': key_char,
            'name': CATEGORIES[key_char]
        }
        print(f"标注为: {CATEGORIES[key_char]}")
    else:
        print("无效按键，跳过")

cv2.destroyAllWindows()

# 保存标注
with open(label_file, 'w', encoding='utf-8') as f:
    json.dump(labels, f, ensure_ascii=False, indent=2)

print(f"\n已保存 {len(labels)} 个标注到 {label_file}")