from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .models.generator import PixelArtGenerator
from .interface.gradio_ui import create_gradio_interface
import gradio as gr
import uvicorn
import webbrowser
import time
import threading

# 不再需要直接导入 ImageProcessor，因为我们现在通过 process.py 使用它
# from utils.image_processing import ImageProcessor

app = FastAPI(title="Pixel Art Generator")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化生成器
generator = PixelArtGenerator()

# 不再需要直接初始化 ImageProcessor
# image_processor = ImageProcessor()

@app.post("/generate")
async def generate_pixel_art(
    sketch: UploadFile = File(...),
    prompt: str = "pixel art style"
):
    """处理上传的草图并生成像素画"""
    try:
        # 读取上传的图片
        image_data = await sketch.read()
        # 生成像素画
        result = generator.generate(image_data, prompt)
        return {"status": "success", "image": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 创建 Gradio 界面
interface = create_gradio_interface(generator)

# 挂载 Gradio 到 FastAPI
app = gr.mount_gradio_app(app, interface, path="/")

def open_browser():
    """延迟2秒后打开浏览器"""
    time.sleep(2)  # 等待服务器完全启动
    webbrowser.open("http://127.0.0.1:8000")

# 不再需要在这里定义 process_image 函数，因为它已经移到了 utils/process.py

if __name__ == "__main__":
    print("正在启动服务器...")
    print("请在浏览器中访问: http://127.0.0.1:8000")
    print("生成过程可能需要几秒钟，请耐心等待")
    
    # 添加一些调试信息
    print("检查模块导入...")
    try:
        from .utils.process import process_image
        print("成功导入 process_image 函数")
    except Exception as e:
        print(f"导入 process_image 函数失败: {str(e)}")
    
    threading.Thread(target=open_browser).start()
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )