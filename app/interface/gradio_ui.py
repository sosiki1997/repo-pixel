import gradio as gr
import os
import sys
from PIL import Image
import numpy as np

# 添加父目录到路径，以便导入 utils 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_processing import ImageProcessor

# 导入我们的处理函数
from ..utils.process import process_image

def create_gradio_interface(generator):
    """创建Gradio界面"""
    
    def handle_image(input_image, pixel_size=20):
        """处理上传的图像或绘制的图像"""
        if input_image is None:
            return None, "请上传或绘制图像"
        
        try:
            # 调用我们的新处理函数
            result_image = process_image(input_image, pixel_size=pixel_size)
            return result_image, "处理成功"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"处理失败: {str(e)}"
    
    # 创建Gradio界面
    with gr.Blocks(title="像素画生成器") as demo:
        gr.Markdown("# 像素画生成器")
        
        with gr.Row():
            with gr.Column():
                # 输入区域 - 只使用上传功能，移除绘图功能
                input_image = gr.Image(label="上传草图", type="pil")
                
                pixel_size = gr.Slider(minimum=5, maximum=50, value=20, step=1, 
                                      label="像素大小")
                
                process_btn = gr.Button("生成像素画")
            
            with gr.Column():
                # 输出区域
                output_image = gr.Image(label="生成结果")
                output_message = gr.Textbox(label="状态")
        
        # 连接按钮和处理函数
        process_btn.click(
            fn=handle_image,
            inputs=[input_image, pixel_size],
            outputs=[output_image, output_message]
        )
    
    return demo

# 启动应用
if __name__ == "__main__":
    demo.launch()
