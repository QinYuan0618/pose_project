import matplotlib.pyplot as plt
import numpy as np

# 模拟数据（等真实数据出来后替换）
def generate_report_charts(video_results):
    """生成可视化报告"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Classroom Posture Analysis Report', fontsize=16, fontweight='bold')
    
    # 1. 各视频坐姿分布（堆叠柱状图）
    ax1 = axes[0, 0]
    videos = [r['video'] for r in video_results]
    categories = ['Standard', 'Slouching', 'Severe Forward/Lying', 'Leaning']
    colors = ['#00ff00', '#ffa500', '#ff0000', '#ffff00']
    
    data = np.array([[r['counts'][cat] for cat in categories] for r in video_results])
    bottom = np.zeros(len(videos))
    
    for i, cat in enumerate(categories):
        ax1.bar(videos, data[:, i], bottom=bottom, label=cat, color=colors[i])
        bottom += data[:, i]
    
    ax1.set_ylabel('Detection Count')
    ax1.set_title('Posture Distribution by Video')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. 整体坐姿比例（饼图）
    ax2 = axes[0, 1]
    total_counts = {cat: sum(r['counts'][cat] for r in video_results) for cat in categories}
    ax2.pie(total_counts.values(), labels=total_counts.keys(), autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax2.set_title('Overall Posture Distribution')
    
    # 3. 不良坐姿趋势（折线图）
    ax3 = axes[1, 0]
    poor_postures = ['Slouching', 'Severe Forward/Lying', 'Leaning']
    for cat in poor_postures:
        values = [r['counts'][cat] / r['total_detections'] * 100 for r in video_results]
        ax3.plot(videos, values, marker='o', label=cat)
    
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('Poor Posture Trends')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. 统计摘要（文本）
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    total_detections = sum(r['total_detections'] for r in video_results)
    good_posture = sum(r['counts']['Standard'] for r in video_results)
    poor_posture = total_detections - good_posture
    
    # 排除标准坐姿来计算最常见的“问题坐姿”
    non_standard_counts = {cat: sum(r['counts'][cat] for r in video_results) for cat in poor_postures}
    
    # 计算最常见问题坐姿
    most_common_issue = max(non_standard_counts.items(), key=lambda x: x[1])[0]
    
    # 获取最常见的问题坐姿类别
    recommendation = most_common_issue
    
    summary_text = f"""
    Summary Statistics
    {'='*30}
    Total Videos: {len(video_results)}
    Total Detections: {total_detections}
    
    Good Posture: {good_posture} ({good_posture/total_detections*100:.1f}%)
    Poor Posture: {poor_posture} ({poor_posture/total_detections*100:.1f}%)
    
    Most Common Issue (excluding 'Standard'):
    {most_common_issue}
    
    Recommendation:
    Focus on correcting
    {recommendation}
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('posture_report.png', dpi=300, bbox_inches='tight')
    print("报告已保存到 posture_report.png")

# 测试用模拟数据
if __name__ == "__main__":
    mock_results = [
        {'video': 'Video1', 'total_detections': 1000, 
         'counts': {'Standard': 500, 'Slouching': 300, 'Severe Forward/Lying': 150, 'Leaning': 50}},
        {'video': 'Video2', 'total_detections': 1100,
         'counts': {'Standard': 600, 'Slouching': 250, 'Severe Forward/Lying': 200, 'Leaning': 50}},
        {'video': 'Video3', 'total_detections': 950,
         'counts': {'Standard': 450, 'Slouching': 350, 'Severe Forward/Lying': 100, 'Leaning': 50}},
        {'video': 'Video4', 'total_detections': 1050,
         'counts': {'Standard': 550, 'Slouching': 300, 'Severe Forward/Lying': 150, 'Leaning': 50}}
    ]
    
    generate_report_charts(mock_results)
