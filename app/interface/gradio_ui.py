import gradio as gr
import time

def create_gradio_interface(generator):
    def generate_fn(image, prompt):
        if image is None:
            return None
        
        try:
            result = generator.generate(image, prompt)
            print('Generation response:', result)
            return result
            
        except Exception as e:
            print(f"错误: {str(e)}")
            print('Generation error:', e)
            return None

    # 创建 Gradio 界面
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 像素画生成器")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="上传草图",
                    type="pil",
                    height=400,
                    width=400,
                    image_mode="RGB",
                    sources=["upload", "webcam", "clipboard"]
                )
                # 使用 gr.Sketchpad 组件来代替手绘草图
                sketchpad = gr.Sketchpad(
                    label="绘图",
                    height=400,
                    width=400
                )
                prompt = gr.Textbox(
                    label="描述",
                    placeholder="描述你想要的像素画风格...",
                    value="pixel art of a cute character"
                )
                generate_btn = gr.Button("生成", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(
                    label="生成结果",
                    height=400,
                    width=400
                )

        gr.Markdown("""
        ### 支持的图片格式：
        - PNG (.png)
        - JPEG/JPG (.jpg, .jpeg)
        
        ### 提示：
        - 可以直接上传图片
        - 也可以使用内置绘图工具绘制
        - 或从剪贴板粘贴图片
        """)

        # 绑定生成按钮
        generate_btn.click(
            fn=generate_fn,
            inputs=[input_image, prompt],
            outputs=output_image,
            api_name=None,
            show_progress="full"
        )

    return interface
