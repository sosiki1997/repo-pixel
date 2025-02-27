import torch
from PIL import Image
import io
from diffusers import StableDiffusionImg2ImgPipeline
import numpy as np
import gc

class PixelArtGenerator:
    def __init__(self):
        """初始化像素艺术生成器"""
        # 检查是否有可用的GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载Stable Diffusion模型
        try:
            print("加载Stable Diffusion模型...")
            # 使用更小的模型
            model_id = "CompVis/stable-diffusion-v1-4"  # 比v1-5小，内存占用更少
            
            # 使用8位精度加载模型以减少内存使用
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                revision="fp16" if self.device.type == "cuda" else "main",
                safety_checker=None,  # 禁用安全检查器以节省内存
                requires_safety_checker=False
            )
            
            # 启用内存优化
            self.pipe.enable_attention_slicing()  # 减少内存使用
            
            # 移动到设备
            self.pipe = self.pipe.to(self.device)
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            # 创建一个空的管道，以便应用程序仍然可以运行
            self.pipe = None
    
    def preprocess_image(self, image_data):
        """预处理图像数据"""
        if isinstance(image_data, bytes):
            # 如果是字节数据，转换为PIL图像
            image = Image.open(io.BytesIO(image_data))
        else:
            # 如果已经是PIL图像，直接使用
            image = image_data
        
        # 确保图像是RGB模式
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # 调整图像大小为较小的尺寸以减少内存使用
        if image.size != (384, 384):  # 使用较小的尺寸
            image = image.resize((384, 384))
        
        return image
    
    def generate(self, image_data, prompt, guidance_scale=7.5, strength=0.75):
        """生成像素艺术"""
        if self.pipe is None:
            raise ValueError("Stable Diffusion模型未成功加载")
        
        try:
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # 预处理图像
            init_image = self.preprocess_image(image_data)
            
            # 增强提示词，强调像素艺术风格
            if "pixel" not in prompt.lower():
                full_prompt = f"{prompt}, pixel art style, 16-bit, retro game art"
            else:
                full_prompt = prompt
            
            print(f"使用提示词: {full_prompt}")
            
            # 生成图像，使用较少的推理步骤
            with torch.no_grad():
                result = self.pipe(
                    prompt=full_prompt,
                    image=init_image,
                    strength=strength,  # 控制原始图像的保留程度
                    guidance_scale=guidance_scale,  # 控制文本提示的影响力
                    num_inference_steps=20,  # 减少步数以减少内存使用
                    height=384,  # 使用较小的输出尺寸
                    width=384
                ).images[0]
                
                # 如果需要，可以将结果调整回512x512
                result = result.resize((512, 512), Image.LANCZOS)
            
            # 再次清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return result
        except Exception as e:
            print(f"生成失败: {str(e)}")
            # 如果失败，返回一个简单的错误图像
            error_img = Image.new('RGB', (512, 512), color=(255, 0, 0))
            return error_img 