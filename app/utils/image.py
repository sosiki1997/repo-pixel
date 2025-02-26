import cv2
import numpy as np
from PIL import Image

def process_sketch(
    image: Image.Image,
    low_threshold: int = 100,
    high_threshold: int = 200,
    bg_color: int = 255
) -> Image.Image:
    """处理草图，提取 Canny 边缘"""
    # 转换为 numpy 数组
    img_array = np.array(image)
    
    # 转换为灰度图
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 应用 Canny 边缘检测
    edges = cv2.Canny(
        blurred,
        threshold1=low_threshold,
        threshold2=high_threshold
    )
    
    # 创建白色背景
    result = np.full_like(edges, bg_color, dtype=np.uint8)
    
    # 将边缘设为黑色
    result[edges > 0] = 0
    
    # 转回 PIL Image
    return Image.fromarray(result) 