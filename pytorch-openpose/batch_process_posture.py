# 处理4个视频（每个300帧）并生成统计报告。
import cv2
import numpy as np
import joblib
from src.body import Body
from src import util
import sys
sys.path.append('.')
from design_features import extract_geometric_features
from datetime import datetime

# 加载模型
print("加载模型...")
body_estimation = Body('model/body_pose_model.pth')
clf = joblib.load('posture_classifier_final.pkl')
print("✓ 模型加载完成\n")

# 视频路径
VIDEO_DIR = r"D:\VSCode\PYCode\pose_project\videos"
video_files = ["1.mp4", "2.mp4", "3.mp4", "4.mp4"]

# 颜色映射
color_map = {
    'Standard': (0, 255, 0),
    'Slouching': (0, 165, 255),
    'Severe Forward/Lying': (0, 0, 255),
    'Leaning': (255, 255, 0)
}

all_results = []

for video_file in video_files:
    video_path = f"{VIDEO_DIR}\\{video_file}"
    print(f"="*50)
    print(f"处理: {video_file}")
    print("="*50)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height}, {fps}FPS, {total_frames}帧")
    
    # 输出视频
    output_path = f"output_{video_file}"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                          fps, (width, height))
    
    # 统计数据
    frame_stats = []
    frame_count = 0
    max_frames = 300  # 每个视频处理300帧
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  进度: {frame_count}/{max_frames}")
        
        # 检测和分类
        candidate, subset = body_estimation(frame)
        canvas = util.draw_bodypose(frame, candidate, subset)
        
        # 当前帧统计
        frame_postures = {'Standard': 0, 'Slouching': 0, 
                         'Severe Forward/Lying': 0, 'Leaning': 0}
        
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
            frame_postures[posture] += 1
            
            # 标注
            nose_idx = person[0]
            if nose_idx != -1:
                x, y = int(candidate[nose_idx][0]), int(candidate[nose_idx][1])
                color = color_map[posture]
                cv2.putText(canvas, posture, (x-60, y-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        frame_stats.append(frame_postures)
        out.write(canvas)
    
    cap.release()
    out.release()
    
    # 计算视频统计
    total_counts = {'Standard': 0, 'Slouching': 0, 
                   'Severe Forward/Lying': 0, 'Leaning': 0}
    for stats in frame_stats:
        for key in total_counts:
            total_counts[key] += stats[key]
    
    total_students = sum(total_counts.values())
    
    result = {
        'video': video_file,
        'frames': frame_count,
        'total_detections': total_students,
        'counts': total_counts,
        'output': output_path
    }
    all_results.append(result)
    
    print(f"  完成! 输出: {output_path}\n")

# 生成报告
print("\n" + "="*60)
print("坐姿分析报告")
print("="*60)
print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

for i, result in enumerate(all_results, 1):
    print(f"【视频{i}】{result['video']}")
    print(f"  处理帧数: {result['frames']}")
    print(f"  检测人次: {result['total_detections']}")
    print(f"  坐姿分布:")
    
    for posture, count in result['counts'].items():
        percentage = count / result['total_detections'] * 100 if result['total_detections'] > 0 else 0
        print(f"    {posture}: {count} ({percentage:.1f}%)")
    
    print(f"  输出文件: {result['output']}\n")

print("="*60)

# 调用可视化
from visualize_report import generate_report_charts
generate_report_charts(all_results)