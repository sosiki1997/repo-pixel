import torch
from PIL import Image
import numpy as np
import io
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from ..utils.image import process_sketch
import huggingface_hub
import requests
import cv2
from diffusers.utils import load_image

def process_sketch(image):
    # 转换为OpenCV格式
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # 边缘检测
    edges = cv2.Canny(img_cv, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # 确保是RGB格式
    
    # 转回PIL格式
    edges_pil = Image.fromarray(edges)
    
    return edges_pil

class PixelArtGenerator:
    def __init__(self):
        # 初始化模型
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float32,
            local_files_only=True  # 使用本地文件
        )
        
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float32,
            local_files_only=True  # 使用本地文件
        )
        
        # 移动到CPU
        self.pipe.to("cpu")
        print("模型已加载到 CPU")

    def generate(self, sketch_image: Image.Image, prompt: str) -> Image.Image:
        try:
            print("开始处理输入图片...")
            sketch_image = sketch_image.convert("RGB")
            sketch_image = sketch_image.resize((512, 512), Image.LANCZOS)
            
            print("提取边缘...")
            canny_image = process_sketch(sketch_image)
            
            print("开始生成图片...")
            full_prompt = f"{prompt}, pixel art, 8-bit style, pixelated"
            negative_prompt = "blurry, smooth, realistic, 3D"
            
            # 设置随机种子以保持一致性
            generator = torch.Generator().manual_seed(42)
            
            output = self.pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                image=canny_image,
                num_inference_steps=20,
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0,
                generator=generator
            ).images[0]
            
            print("后处理...")
            output = output.resize((48, 48), Image.NEAREST)
            output = output.resize((400, 400), Image.NEAREST)
            
            print("完成！")
            return output
            
        except Exception as e:
            print(f"生成错误: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None 