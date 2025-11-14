import cv2
from src.body import Body
from src import util

# 检测是否头部前倾：鼻子和脖子的像素< 50 则为良好，否则为前倾
def check_head_forward(candidate, person):
    """检查头部前倾"""
    nose_idx = person[0]
    neck_idx = person[1]
    
    if nose_idx == -1 or neck_idx == -1:
        return None, "No detect"
    
    nose_x = candidate[nose_idx][0]
    neck_x = candidate[neck_idx][0]
    forward_dist = abs(nose_x - neck_x)
    
    if forward_dist > 50:
        return False, f"Forward {forward_dist:.0f}"
    else:
        return True, f"Good {forward_dist:.0f}"

def check_lying_on_desk(candidate, person):
    """检查是否趴桌子"""
    nose_idx = person[0]
    neck_idx = person[1]
    
    if nose_idx == -1 or neck_idx == -1:
        return None, "No detect"
    
    nose_y = candidate[nose_idx][1]
    neck_y = candidate[neck_idx][1]
    
    # 头部和脖子的垂直距离（俯视角度下，趴桌子时头会低于脖子很多）
    vertical_diff = nose_y - neck_y
    
    if vertical_diff < -40:  # 头比脖子低很多
        return False, f"Lying {abs(vertical_diff):.0f}"
    else:
        return True, f"OK {abs(vertical_diff):.0f}"

# 加载模型和视频
body_estimation = Body('model/body_pose_model.pth')
cap = cv2.VideoCapture(r"D:\VSCode\PYCode\pose_project\videos\1.mp4")
ret, frame = cap.read()
cap.release()

# 检测
candidate, subset = body_estimation(frame)

# 画骨架
canvas = util.draw_bodypose(frame, candidate, subset)

# 在每个学生头部上方标注结果
for i, person in enumerate(subset.astype(int)):
    # 检查头部前倾
    result1, msg1 = check_head_forward(candidate, person)
    # 检查是否趴桌子
    result2, msg2 = check_lying_on_desk(candidate, person)
    
    nose_idx = person[0]
    if nose_idx != -1:
        x, y = int(candidate[nose_idx][0]), int(candidate[nose_idx][1])
        
        # 显示两行结果
        color1 = (0, 255, 0) if result1 else (0, 0, 255)
        color2 = (0, 255, 0) if result2 else (0, 0, 255)
        
        cv2.putText(canvas, msg1, (x-50, y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 2)
        cv2.putText(canvas, msg2, (x-50, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)

cv2.imwrite('posture_result.jpg', canvas)
print("Saved to posture_result.jpg")