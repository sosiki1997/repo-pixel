import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from segment_anything import SamPredictor, sam_model_registry
import os

class ImageProcessor:
    def __init__(self):
        # 加载SAM模型用于图像分割
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 更新为你下载的模型实际路径
        model_path =  "./models/sam_vit_h_4b8939.pth" # 修改为你的实际路径
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SAM模型文件未找到: {model_path}")
        print(f"加载SAM模型: {model_path}")
        self.sam = sam_model_registry["vit_h"](checkpoint=model_path)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)
        
    def extract_subject(self, image_path):
        """提取图像中的主体对象"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 设置SAM预测器
        self.predictor.set_image(image_rgb)
        
        # 使用图像中心点作为提示
        h, w = image.shape[:2]
        center_point = np.array([[w//2, h//2]])
        
        # 获取掩码
        masks, _, _ = self.predictor.predict(
            point_coords=center_point,
            point_labels=np.array([1]),  # 1表示前景
            multimask_output=False
        )
        
        # 获取最佳掩码
        mask = masks[0]
        
        # 保存掩码为图像
        mask_path = os.path.splitext(image_path)[0] + '_mask.png'
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        
        return image_path, mask
    
    def pixelate_subject(self, image_path, mask, pixel_size=20):
        """只对主体部分进行像素化，保持透明背景"""
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 创建一个透明背景的RGBA图像
        result = np.zeros((h, w, 4), dtype=np.uint8)
        
        # 获取主体区域的边界框
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return image_path  # 如果没有检测到主体，返回原图
            
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # 提取主体区域
        subject = image_rgb[y_min:y_max, x_min:x_max]
        
        # 对主体进行像素化
        h_sub, w_sub = subject.shape[:2]
        temp = cv2.resize(subject, (w_sub//pixel_size, h_sub//pixel_size), 
                         interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (w_sub, h_sub), 
                              interpolation=cv2.INTER_NEAREST)
        
        # 创建主体区域的掩码
        subject_mask = mask[y_min:y_max, x_min:x_max]
        
        # 将像素化后的主体放回原位置，只在掩码区域
        for c in range(3):  # RGB通道
            result[y_min:y_max, x_min:x_max, c] = pixelated[:,:,c] * subject_mask
        
        # 设置Alpha通道为掩码
        result[:,:,3] = mask * 255
        
        # 保存结果为PNG（支持透明度）
        result_path = os.path.splitext(image_path)[0] + '_pixelated.png'
        
        # 使用PIL保存RGBA图像
        pil_image = Image.fromarray(result)
        pil_image.save(result_path, format='PNG')
        
        return result_path 