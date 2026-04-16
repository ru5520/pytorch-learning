"""
FastAPI 部署 MNIST 模型
=======================

把 PyTorch CNN 模型部署成 API 接口
启动方式: uvicorn 01_mnist_api:app --reload --port 8000
访问地址: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from PIL import Image
import io
import numpy as np
import os

# ===== 1. 定义 CNN 模型（和训练时一样）=====
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv(x)
        x = self.fc(x)
        return x

# ===== 2. 加载模型 =====
app = FastAPI(title="MNIST 手写数字识别 API", version="1.0")

# 加载模型
model = CNN()

# 尝试加载权重（如果没有权重文件，先用随机初始化）
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "mnist_cnn.pth")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    print("✅ 模型权重加载成功")
else:
    print("⚠️ 未找到模型权重，使用随机初始化的模型")

model.eval()

# ===== 3. 预处理图片 =====
def preprocess_image(file_bytes) -> torch.Tensor:
    """把上传的图片转成 28x28 张量"""
    image = Image.open(io.BytesIO(file_bytes)).convert("L")  # 转灰度
    image = image.resize((28, 28))                           # 缩放到 28x28
    image = np.array(image).astype(np.float32) / 255.0       # 归一化
    tensor = torch.from_numpy(image).float()                 # 转张量
    return tensor

# ===== 4. 定义 API 接口 =====

@app.get("/")
def root():
    """首页"""
    return {
        "message": "MNIST 手写数字识别 API",
        "version": "1.0",
        "docs": "http://127.0.0.1:8000/docs"
    }

@app.get("/health")
def health():
    """健康检查"""
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    预测接口

    上传一张手写数字图片，返回预测结果

    返回格式:
    {
        "digit": 5,
        "confidence": 0.9987,
        "all_probs": [0.0, 0.0, 0.0, 0.0, 0.0, 0.9987, 0.0013, ...]
    }
    """
    # 读取图片
    contents = await file.read()
    image_tensor = preprocess_image(contents)

    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)[0].numpy()
        pred_digit = int(output.argmax(1).item())
        confidence = float(probs[pred_digit])

    return {
        "digit": pred_digit,
        "confidence": round(confidence, 4),
        "all_probs": [round(float(p), 4) for p in probs]
    }

# ===== 5. 启动提示 =====
print("\n" + "=" * 50)
print("🚀 MNIST API 已启动！")
print("=" * 50)
print("API 文档: http://127.0.0.1:8000/docs")
print("预测接口: POST http://127.0.0.1:8000/predict")
print("启动命令: uvicorn 01_mnist_api:app --reload --port 8000")
print("=" * 50)