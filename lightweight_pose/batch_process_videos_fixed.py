import cv2
import torch
import numpy as np
from mobilenet_openpose import LightweightOpenPose
import os
from datetime import datetime
import json

class BatchVideoProcessor:
    def __init__(self):
        # 智能选择设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        if not torch.cuda.is_available():
            print("警告: GPU不可用，使用CPU处理（速度会较慢）")
        
        # 加载模型 - 修复加载问题
        self.model = LightweightOpenPose().to(self.device)
        
        # 关键修复：添加map_location参数
        checkpoint = torch.load("lightweight_openpose_distilled.pth", 
                               map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("模型加载成功\n")
        
        # COCO关键点和骨架定义
        self.SKELETON = [
            [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
            [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13]
        ]
    
    def extract_keypoints(self, heatmap, width, height, threshold=0.3):
        """从热图提取关键点"""
        keypoints = {}
        for i in range(min(18, heatmap.shape[0]-1)):
            h = heatmap[i]
            if h.max() > threshold:
                y, x = np.unravel_index(h.argmax(), h.shape)
                keypoints[i] = (
                    int(x * width / h.shape[1]),
                    int(y * height / h.shape[0])
                )
        return keypoints
    
    def analyze_sitting_posture(self, keypoints):
        """分析坐姿评分"""
        score = 100
        issues = []
        
        # 检查必要关键点
        neck = keypoints.get(1)
        r_shoulder = keypoints.get(2)
        l_shoulder = keypoints.get(5)
        r_hip = keypoints.get(8)
        l_hip = keypoints.get(11)
        
        if neck and r_shoulder and l_shoulder:
            # 1. 肩膀平衡检查
            shoulder_diff = abs(r_shoulder[1] - l_shoulder[1])
            if shoulder_diff > 20:
                score -= 20
                issues.append("肩膀不平衡")
            elif shoulder_diff > 10:
                score -= 10
                issues.append("肩膀轻微倾斜")
            
            # 2. 头部前倾检查
            shoulder_center_x = (r_shoulder[0] + l_shoulder[0]) / 2
            head_forward = abs(neck[0] - shoulder_center_x)
            if head_forward > 40:
                score -= 25
                issues.append("头部严重前倾")
            elif head_forward > 20:
                score -= 15
                issues.append("头部轻微前倾")
            
            # 3. 驼背检查（如果有髋部数据）
            if r_hip or l_hip:
                hip = r_hip if r_hip else l_hip
                spine_alignment = abs(neck[0] - hip[0])
                if spine_alignment > 50:
                    score -= 20
                    issues.append("脊柱弯曲")
        else:
            score = 0
            issues.append("关键点检测不足")
        
        return max(0, score), issues
    
    def process_single_video(self, video_path, video_name):
        """处理单个视频"""
        print(f"处理视频: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  视频信息: {width}x{height}, {fps}FPS, {total_frames}帧")
        
        # 输出视频
        output_path = f"outputs/{video_name}_analyzed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 统计数据
        frame_scores = []
        all_issues = []
        frame_count = 0
        
        # 处理帧 - 为了加速，可以跳帧处理
        skip_frames = 2 if self.device.type == 'cpu' else 1  # CPU时跳帧
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 跳帧处理
            if frame_count % skip_frames != 0:
                continue
            
            # 每30帧显示进度
            if frame_count % 30 == 0:
                progress = frame_count / total_frames * 100
                print(f"    进度: {progress:.1f}%")
            
            # 预处理 - 降低分辨率以加速
            resize_dim = 256 if self.device.type == 'cpu' else 368
            img = cv2.resize(frame, (resize_dim, resize_dim))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(self.device)
            
            # 推理
            with torch.no_grad():
                pafs, heatmaps = self.model(img_tensor)
            
            heatmap = heatmaps[-1][0].cpu().numpy()
            
            # 提取关键点
            keypoints = self.extract_keypoints(heatmap, width, height)
            
            # 绘制骨架
            for connection in self.SKELETON:
                if connection[0] in keypoints and connection[1] in keypoints:
                    pt1 = keypoints[connection[0]]
                    pt2 = keypoints[connection[1]]
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            # 绘制关键点
            for idx, point in keypoints.items():
                cv2.circle(frame, point, 4, (0, 0, 255), -1)
            
            # 分析坐姿
            score, issues = self.analyze_sitting_posture(keypoints)
            frame_scores.append(score)
            if issues:
                all_issues.extend(issues)
            
            # 显示评分
            color = (0, 255, 0) if score >= 80 else (0, 165, 255) if score >= 60 else (0, 0, 255)
            cv2.putText(frame, f"Score: {score}/100", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # 显示问题
            for i, issue in enumerate(issues[:2]):
                cv2.putText(frame, issue, (10, 60 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        # 计算统计数据
        avg_score = np.mean(frame_scores) if frame_scores else 0
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        result = {
            'video_name': video_name,
            'total_frames': frame_count,
            'processed_frames': len(frame_scores),
            'avg_score': avg_score,
            'min_score': min(frame_scores) if frame_scores else 0,
            'max_score': max(frame_scores) if frame_scores else 0,
            'issues': issue_counts,
            'output_path': output_path
        }
        
        print(f"  完成! 平均分: {avg_score:.1f}/100\n")
        
        return result
    
    def generate_comprehensive_report(self, all_results):
        """生成综合报告"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
========================================================
            中小学生坐姿识别分析报告
========================================================
生成时间: {timestamp}
分析模型: 轻量级OpenPose (MobileNetV3 + EMA)
模型参数: 3.46M (相比原版减少97.6%)
运行设备: {self.device}

========================================================
                    视频分析结果
========================================================
"""
        
        # 各视频详细结果
        for i, result in enumerate(all_results, 1):
            report += f"""
【视频 {i}】{result['video_name']}
----------------------------------------
总帧数: {result['total_frames']}
处理帧数: {result['processed_frames']}
平均得分: {result['avg_score']:.1f}/100
最低得分: {result['min_score']}/100
最高得分: {result['max_score']}/100

主要问题:"""
            
            if result['issues']:
                for issue, count in sorted(result['issues'].items(), 
                                          key=lambda x: x[1], reverse=True):
                    percentage = count / result['processed_frames'] * 100
                    report += f"\n  - {issue}: {percentage:.1f}% ({count}帧)"
            else:
                report += "\n  - 无明显问题"
            
            # 评级
            avg = result['avg_score']
            if avg >= 85:
                grade = "优秀"
                suggestion = "坐姿非常标准，请继续保持"
            elif avg >= 70:
                grade = "良好"
                suggestion = "坐姿基本正确，注意细节改进"
            elif avg >= 60:
                grade = "一般"
                suggestion = "坐姿需要改善，建议调整坐姿习惯"
            else:
                grade = "较差"
                suggestion = "坐姿问题较多，强烈建议纠正"
            
            report += f"\n\n评级: {grade}\n建议: {suggestion}\n"
        
        # 综合统计
        report += """
========================================================
                    综合统计分析
========================================================
"""
        all_scores = [r['avg_score'] for r in all_results]
        report += f"""
视频数量: {len(all_results)}
总体平均分: {np.mean(all_scores):.1f}/100
最佳表现: 视频{np.argmax(all_scores)+1} ({max(all_scores):.1f}分)
最需改进: 视频{np.argmin(all_scores)+1} ({min(all_scores):.1f}分)

常见问题排序:"""
        
        # 合并所有问题统计
        total_issues = {}
        total_frames = sum(r['processed_frames'] for r in all_results)
        for result in all_results:
            for issue, count in result['issues'].items():
                total_issues[issue] = total_issues.get(issue, 0) + count
        
        if total_issues:
            for issue, count in sorted(total_issues.items(), 
                                      key=lambda x: x[1], reverse=True):
                percentage = count / total_frames * 100
                report += f"\n  {issue}: {percentage:.1f}%"
        else:
            report += "\n  无明显问题"
        
        # 改善建议
        report += """

========================================================
                    改善建议
========================================================
"""
        if "头部严重前倾" in total_issues or "头部轻微前倾" in total_issues:
            report += """
1. 头部姿势调整:
   - 调整显示器高度，使视线与屏幕顶部平齐
   - 屏幕距离保持50-70cm
   - 每30分钟做颈部放松运动
"""
        
        if "肩膀不平衡" in total_issues or "肩膀轻微倾斜" in total_issues:
            report += """
2. 肩膀姿势纠正:
   - 确保椅子高度合适，双脚平放地面
   - 键盘鼠标位置适中，避免单侧用力
   - 定期做肩部拉伸运动
"""
        
        if "脊柱弯曲" in total_issues:
            report += """
3. 脊柱健康维护:
   - 使用有腰部支撑的椅子
   - 保持背部挺直，不要弓背
   - 加强核心肌群锻炼
"""
        
        report += """
========================================================
                    输出文件列表
========================================================
"""
        for result in all_results:
            report += f"\n{result['video_name']} -> {result['output_path']}"
        
        report += "\n\n========================================================"
        
        return report

# 主程序
if __name__ == "__main__":
    # 先检查GPU状态
    print("系统信息:")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print("-"*50)
    
    processor = BatchVideoProcessor()
    
    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)
    
    # 处理4个视频
    video_files = ["1.mp4", "2.mp4", "3.mp4", "4.mp4"]
    all_results = []
    
    print("="*50)
    print("开始批量处理视频")
    print("="*50)
    
    for video_file in video_files:
        video_path = f"/root/pose_project/videos/{video_file}"
        if os.path.exists(video_path):
            try:
                result = processor.process_single_video(video_path, video_file.split('.')[0])
                all_results.append(result)
            except Exception as e:
                print(f"处理视频 {video_file} 时出错: {e}")
        else:
            print(f"警告: 找不到视频 {video_path}")
    
    # 生成综合报告
    if all_results:
        report = processor.generate_comprehensive_report(all_results)
        
        # 保存报告
        report_path = f"outputs/comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"\n报告已保存到: {report_path}")
        
        # 保存JSON格式结果（便于后续分析）
        json_path = "outputs/analysis_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"JSON数据已保存到: {json_path}")
    else:
        print("没有成功处理的视频")
