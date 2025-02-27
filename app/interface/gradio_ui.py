import gradio as gr
import os
import sys
from PIL import Image
import numpy as np
import io

# 添加父目录到路径，以便导入 utils 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_processing import ImageProcessor

# 导入我们的处理函数
from ..utils.process import process_image

def create_gradio_interface(generator):
    """创建Gradio界面"""
    
    def handle_image_opencv(input_image, pixel_size=20):
        """使用OpenCV处理上传的图像"""
        if input_image is None:
            return None, "请上传或绘制图像"
        
        try:
            # 调用OpenCV处理函数
            result_image = process_image(input_image, pixel_size=pixel_size)
            return result_image, "处理成功"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"处理失败: {str(e)}"
    
    def handle_image_diffusion(input_image, prompt, guidance_scale=7.5):
        """使用Stable Diffusion处理上传的图像"""
        if input_image is None:
            return None, "请上传图像"
        
        try:
            # 将PIL图像转换为字节
            img_byte_arr = io.BytesIO()
            input_image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # 调用generator的generate函数
            result_image = generator.generate(img_byte_arr, prompt, guidance_scale=guidance_scale)
            
            return result_image, "生成成功"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"生成失败: {str(e)}"
    
    # 创建Gradio界面
    with gr.Blocks(title="像素画生成器") as demo:
        gr.Markdown("# 像素画生成器")
        
        with gr.Tabs():
            with gr.TabItem("OpenCV像素化"):
                with gr.Row():
                    with gr.Column():
                        # 输入区域 - 设置固定大小
                        input_image_opencv = gr.Image(
                            label="上传草图", 
                            type="pil",
                            height=512,  # 设置固定高度
                            width=512,   # 设置固定宽度
                            container=True,  # 使用容器包裹
                            show_download_button=False,  # 不显示下载按钮
                            show_label=True,  # 显示标签
                        )
                        
                        pixel_size = gr.Slider(minimum=5, maximum=50, value=20, step=1, 
                                            label="像素大小")
                        
                        process_btn_opencv = gr.Button("生成像素画 (OpenCV)")
                    
                    with gr.Column():
                        # 输出区域 - 也设置固定大小
                        output_image_opencv = gr.Image(
                            label="生成结果",
                            height=512,  # 设置固定高度
                            width=512,   # 设置固定宽度
                            container=True,  # 使用容器包裹
                            show_download_button=True,  # 显示下载按钮
                        )
                        output_message_opencv = gr.Textbox(label="状态")
            
            with gr.TabItem("Stable Diffusion生成"):
                with gr.Row():
                    with gr.Column():
                        # 输入区域
                        input_image_diffusion = gr.Image(
                            label="上传参考图", 
                            type="pil",
                            height=512,
                            width=512,
                            container=True,
                            show_download_button=False,
                        )
                        
                        prompt = gr.Textbox(
                            label="提示词", 
                            placeholder="描述你想要的像素艺术风格...",
                            value="pixel art style, 16-bit, retro game art"
                        )
                        
                        guidance_scale = gr.Slider(
                            minimum=1.0, 
                            maximum=15.0, 
                            value=7.5, 
                            step=0.5,
                            label="引导系数 (越高越遵循提示词)"
                        )
                        
                        process_btn_diffusion = gr.Button("生成像素画 (Stable Diffusion)")
                    
                    with gr.Column():
                        # 输出区域
                        output_image_diffusion = gr.Image(
                            label="生成结果",
                            height=512,
                            width=512,
                            container=True,
                            show_download_button=True,
                        )
                        output_message_diffusion = gr.Textbox(label="状态")
        
        # 连接OpenCV处理按钮和函数
        process_btn_opencv.click(
            fn=handle_image_opencv,
            inputs=[input_image_opencv, pixel_size],
            outputs=[output_image_opencv, output_message_opencv]
        )
        
        # 连接Stable Diffusion处理按钮和函数
        process_btn_diffusion.click(
            fn=handle_image_diffusion,
            inputs=[input_image_diffusion, prompt, guidance_scale],
            outputs=[output_image_diffusion, output_message_diffusion]
        )
    
    return demo

# 启动应用
if __name__ == "__main__":
    demo.launch()
