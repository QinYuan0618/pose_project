# 这个文件用于测试Openpose能否正常检测骨架
import cv2
import copy
from src.body import Body
from src import util

# 加载模型
print("加载模型...")
body_estimation = Body('model/body_pose_model.pth')
print("✓ 模型加载完成")

# 读取测试图片
img = cv2.imread('images/demo.jpg')
print(f"图片尺寸: {img.shape}")

# 检测骨架
print("检测骨架...")
candidate, subset = body_estimation(img)
print(f"✓ 检测到 {len(subset)} 个人")

# 绘制骨架
canvas = copy.deepcopy(img)
canvas = util.draw_bodypose(canvas, candidate, subset)

# 保存结果
cv2.imwrite('result.jpg', canvas)
print("✓ 结果已保存到 result.jpg")