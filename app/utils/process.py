from PIL import Image
# 确保导入路径正确
from .image_processing import ImageProcessor
import time
import gradio as gr
import traceback

# 初始化图像处理器
try:
    print("初始化 ImageProcessor...")
    image_processor = ImageProcessor()
    print("ImageProcessor 初始化成功")
except Exception as e:
    print(f"初始化 ImageProcessor 失败: {str(e)}")
    traceback.print_exc()

def process_image(input_image, *args, **kwargs):
    progress = gr.Progress()
    
    print("开始处理输入图片...")
    progress(0, desc="开始处理...")
    
    # 保存输入图像到临时文件
    temp_input_path = "temp_input.png"
    input_image.save(temp_input_path)
    
    try:
        print("使用SAM提取主体...")
        progress(0.3, desc="提取主体...")
        # 使用SAM提取主体
        _, mask = image_processor.extract_subject(temp_input_path)
        
        print("对主体进行像素化...")
        progress(0.6, desc="像素化处理...")
        # 对主体进行像素化
        pixel_size = kwargs.get('pixel_size', 20)
        result_path = image_processor.pixelate_subject(temp_input_path, mask, pixel_size)
        
        # 读取结果图像
        progress(0.9, desc="生成最终图像...")
        result_image = Image.open(result_path)
        
        print("完成主体提取和像素化!")
        progress(1.0, desc="处理完成!")
        return result_image
    except Exception as e:
        traceback.print_exc()
        print(f"处理失败: {str(e)}")
        progress(1.0, desc="处理失败")
        
        # 如果失败，继续使用原来的处理流程
        print("回退到原始处理流程...")
        
        print("提取边缘...")
        # 原来的边缘提取代码
        
        print("开始生成图片...")
        # 原来的图片生成代码
        
        # ... 其余原始处理代码 ...
        return None 