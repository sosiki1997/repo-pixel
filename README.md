
Pixel Art Generator
一个基于 AI 的像素画生成器，可以将手绘草图转换为像素风格的艺术作品。
功能特点
将手绘草图转换为像素风格图片
支持自定义文本描述
实时生成进度显示
简洁的用户界面
安装步骤
克隆项目：
git clone [项目地址]
cd pixel-art-generator
创建并激活 conda 环境：
conda create -n pixel python=3.9
conda activate pixel
安装依赖：
pip install -r requirements.txt
使用方法
启动服务器：
python -m app.main
打开浏览器访问：
http://localhost:8000
使用步骤：
上传或绘制草图
输入描述文字
点击生成按钮
等待10-20秒生成结果
系统要求
Python 3.9+
CUDA 支持（可选，用于 GPU 加速）
8GB+ RAM
现代浏览器（Chrome, Firefox, Safari 等）
注意事项
生成过程可能需要10-20秒
建议使用简单的线稿作为输入
生成过程中请勿刷新页面
技术栈
FastAPI
Gradio
Diffusers
Transformers
PyTorch
ControlNet
许可证
MIT License