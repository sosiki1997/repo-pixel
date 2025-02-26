from PIL import Image
from .image_processing import ImageProcessor

# 初始化图像处理器
image_processor = ImageProcessor()

def process_image(input_image, *args, **kwargs):
    print("开始处理输入图片...")
    
    # 保存输入图像到临时文件
    temp_input_path = "temp_input.png"
    input_image.save(temp_input_path)
    
    try:
        print("使用SAM提取主体...")
        # 使用SAM提取主体
        _, mask = image_processor.extract_subject(temp_input_path)
        
        print("对主体进行像素化...")
        # 对主体进行像素化
        pixel_size = kwargs.get('pixel_size', 20)
        result_path = image_processor.pixelate_subject(temp_input_path, mask, pixel_size)
        
        # 读取结果图像
        result_image = Image.open(result_path)
        
        print("完成主体提取和像素化!")
        return result_image
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"处理失败: {str(e)}")
        
        # 如果失败，继续使用原来的处理流程
        print("回退到原始处理流程...")
        
        print("提取边缘...")
        # 原来的边缘提取代码
        
        print("开始生成图片...")
        # 原来的图片生成代码
        
        # ... 其余原始处理代码 ...
        return None 