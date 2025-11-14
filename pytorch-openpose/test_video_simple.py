# 视频测试，这个文件用我私有数据集（视频）测试openpose模型是否OK,只测试4.mp4前面的50帧
import cv2
import copy
from src.body import Body
from src import util

# 修改这里为你的视频路径
VIDEO_PATH = r"D:\VSCode\PYCode\pose_project\videos\4.mp4"

# 加载模型
print("加载模型...")
body_estimation = Body('model/body_pose_model.pth')

# 打开视频
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"视频: {width}x{height}, {fps}FPS")

# 输出视频
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
while frame_count < 50:  # 只测试前50帧
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"处理第 {frame_count} 帧...")
    
    # 检测并绘制
    candidate, subset = body_estimation(frame)
    canvas = copy.deepcopy(frame)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    
    out.write(canvas)

cap.release()
out.release()
print(f"✓ 完成！处理了 {frame_count} 帧，保存到 output.mp4")