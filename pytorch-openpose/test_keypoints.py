import cv2
from src.body import Body

# 加载模型
body_estimation = Body('model/body_pose_model.pth')

# 读取一帧
cap = cv2.VideoCapture(r"D:\VSCode\PYCode\pose_project\videos\1.mp4")
ret, frame = cap.read()
cap.release()

# 检测
candidate, subset = body_estimation(frame)

# 画第一个学生的关键点，带编号
if len(subset) > 0:
    person = subset[0].astype(int)
    for i in range(8):  # 上半身0-7
        idx = person[i]
        if idx != -1:
            x, y = candidate[idx][:2]
            x, y = int(x), int(y)
            # 画点
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            # 画编号
            cv2.putText(frame, str(i), (x+15, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"{i}: ({x}, {y})")

cv2.imwrite('keypoints_check.jpg', frame)
print("已保存到 keypoints_check.jpg，打开查看")