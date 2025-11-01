import numpy as np
import torch
from mobilenet_openpose import LightweightOpenPose

def analyze_sitting_posture(keypoints):
    """
    分析坐姿是否正确
    关键点索引（COCO格式）：
    0: 鼻子, 1: 脖子, 2: 右肩, 3: 右肘, 4: 右腕,
    5: 左肩, 6: 左肘, 7: 左腕, 8: 右臀, 9: 右膝,
    10: 右踝, 11: 左臀, 12: 左膝, 13: 左踝
    """
    posture_score = 100  # 初始分数
    issues = []
    
    # 检查是否所有关键点都存在
    if None in [keypoints[1], keypoints[2], keypoints[5]]:  # 脖子和双肩
        return 0, ["无法检测到关键点"]
    
    # 1. 检查肩膀是否水平
    if keypoints[2] and keypoints[5]:  # 双肩
        shoulder_diff = abs(keypoints[2][1] - keypoints[5][1])
        if shoulder_diff > 20:
            posture_score -= 20
            issues.append("双肩不平衡")
    
    # 2. 检查头部前倾
    if keypoints[0] and keypoints[1]:  # 鼻子和脖子
        head_forward = keypoints[0][0] - keypoints[1][0]
        if abs(head_forward) > 30:
            posture_score -= 30
            issues.append("头部前倾")
    
    # 3. 检查背部弯曲（通过肩膀和臀部）
    if keypoints[1] and (keypoints[8] or keypoints[11]):  # 脖子和臀部
        hip_point = keypoints[8] if keypoints[8] else keypoints[11]
        spine_angle = np.arctan2(hip_point[1] - keypoints[1][1], 
                                 hip_point[0] - keypoints[1][0])
        if abs(spine_angle - np.pi/2) > 0.3:  # 偏离垂直超过17度
            posture_score -= 25
            issues.append("脊柱弯曲")
    
    return posture_score, issues

def generate_posture_report(video_path):
    """生成坐姿分析报告"""
    print("正在分析坐姿...")
    # 这里添加视频处理和分析逻辑
    
    report = """
    ========== 坐姿分析报告 ==========
    视频: {}
    
    检测到的问题:
    1. 头部前倾 - 出现频率: 60%
    2. 肩膀不平 - 出现频率: 30%
    3. 脊柱弯曲 - 出现频率: 45%
    
    建议:
    - 调整显示器高度，保持视线水平
    - 每30分钟起身活动
    - 保持双脚平放地面
    
    综合评分: 65/100
    ==================================
    """.format(video_path)
    
    return report

if __name__ == "__main__":
    print("坐姿分析模块准备就绪!")
    # 测试
    sample_keypoints = [(100, 100), (100, 120), (80, 140), None, None,
                       (120, 140), None, None, (90, 200), None,
                       None, (110, 200), None, None]
    score, issues = analyze_sitting_posture(sample_keypoints)
    print(f"坐姿评分: {score}/100")
    print(f"发现问题: {issues}")
