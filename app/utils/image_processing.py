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
        model_path = "./models/sam_vit_h_4b8939.pth"  # 修改为你的实际路径
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SAM模型文件未找到: {model_path}")
            
        print(f"加载SAM模型: {model_path}")
        self.sam = sam_model_registry["vit_h"](checkpoint=model_path)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)
        
    def extract_subject(self, image_path):
        """提取图像中的主体对象，使用多点提示和更强的分割策略"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 设置SAM预测器
        self.predictor.set_image(image_rgb)
        
        # 使用多个点作为提示，覆盖图像的不同区域
        points = np.array([
            [w//2, h//2],      # 中心
            [w//4, h//4],      # 左上
            [3*w//4, h//4],    # 右上
            [w//4, 3*h//4],    # 左下
            [3*w//4, 3*h//4],  # 右下
        ])
        
        print(f"使用多点提示: {points}")
        
        # 所有点都标记为前景
        labels = np.ones(len(points))
        
        # 获取掩码
        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True  # 生成多个掩码
        )
        
        # 选择得分最高的掩码
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]
        
        print(f"选择得分最高的掩码: {scores[best_mask_idx]}")
        
        # 检查掩码覆盖面积
        mask_area = np.sum(mask)
        image_area = h * w
        coverage = mask_area / image_area
        print(f"掩码覆盖率: {coverage:.2%}")
        
        # 如果掩码覆盖率太小，尝试使用更简单的方法
        if coverage < 0.05:  # 如果覆盖率小于5%
            print("掩码覆盖率太小，尝试使用颜色阈值分割")
            
            # 使用颜色阈值分割
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 定义橙色范围 (松鼠的主要颜色)
            lower_orange = np.array([10, 100, 100])
            upper_orange = np.array([25, 255, 255])
            
            # 创建掩码
            color_mask = cv2.inRange(hsv, lower_orange, upper_orange)
            
            # 应用形态学操作清理掩码
            kernel = np.ones((5,5), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            
            # 转换为布尔掩码
            mask = color_mask > 0
            
            # 再次检查掩码覆盖率
            mask_area = np.sum(mask)
            coverage = mask_area / image_area
            print(f"颜色阈值分割后的掩码覆盖率: {coverage:.2%}")
            
            # 如果仍然太小，使用简单的矩形区域
            if coverage < 0.05:
                print("颜色阈值分割仍然不理想，使用中心区域")
                mask = np.zeros((h, w), dtype=bool)
                # 使用图像中心的60%区域
                h_start, h_end = int(h*0.2), int(h*0.8)
                w_start, w_end = int(w*0.2), int(w*0.8)
                mask[h_start:h_end, w_start:w_end] = True
        
        # 保存掩码为图像
        mask_path = os.path.splitext(image_path)[0] + '_mask.png'
        cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
        
        return image_path, mask
    
    def pixelate_subject(self, image_path, mask, pixel_size=20):
        """对主体进行像素化，并确保边缘也完全像素化"""
        print("开始像素化处理...")
        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        print("图像读取成功，开始处理...")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # 创建一个透明背景的RGBA图像
        result = np.zeros((h, w, 4), dtype=np.uint8)
        
        # 获取主体区域的边界框
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            print("未检测到主体，返回原图")
            return image_path  # 如果没有检测到主体，返回原图
            
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        print(f"主体边界框: ({x_min}, {y_min}) - ({x_max}, {y_max})")
        
        # 提取主体区域
        subject = image_rgb[y_min:y_max, x_min:x_max]
        subject_mask = mask[y_min:y_max, x_min:x_max]
        
        # 创建一个完整的RGBA图像，包含主体和透明背景
        subject_rgba = np.zeros((subject.shape[0], subject.shape[1], 4), dtype=np.uint8)
        subject_rgba[:,:,:3] = subject
        subject_rgba[:,:,3] = (subject_mask * 255).astype(np.uint8)
        
        print("对整个主体(包括边缘)进行像素化...")
        # 对整个RGBA图像进行像素化处理
        h_sub, w_sub = subject_rgba.shape[:2]
        print(f"主体尺寸: {w_sub}x{h_sub}, 像素大小: {pixel_size}")
        
        # 确保目标尺寸至少为1x1
        target_w = max(1, w_sub // pixel_size)
        target_h = max(1, h_sub // pixel_size)
        print(f"目标尺寸: {target_w}x{target_h}")
        
        # 先缩小
        temp = cv2.resize(subject_rgba, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        # 再放大，使用最近邻插值以保持像素化效果
        pixelated_rgba = cv2.resize(temp, (w_sub, h_sub), interpolation=cv2.INTER_NEAREST)
        
        print("将处理后的主体放回原位置...")
        # 将处理后的主体放回原位置
        result[y_min:y_max, x_min:x_max] = pixelated_rgba
        
        # 保存结果为PNG（支持透明度）
        result_path = os.path.splitext(image_path)[0] + '_pixelated.png'
        
        print(f"保存结果到: {result_path}")
        # 使用PIL保存RGBA图像
        pil_image = Image.fromarray(result)
        pil_image.save(result_path, format='PNG')
        
        print("像素化处理完成!")
        return result_path 