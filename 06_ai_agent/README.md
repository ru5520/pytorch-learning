# 🧠 PyTorch Learning

一个从零开始的深度学习学习项目，涵盖神经网络基础、CNN、ResNet、Kaggle 竞赛和 AI Agent。

---

## 📁 项目结构


pytorch-learning/
├── 01_basics/ # 神经网络基础
│ └── neural_network.py # 从零实现神经网络
├── 02_nn_framework/ # PyTorch 框架
│ └── pytorch_nn.py # 用 nn.Module 实现
├── 03_mnist/ # MNIST 手写数字识别
│ ├── 01_mnist.py # 全连接网络 (96.76%)
│ ├── 02_cnn.py # CNN 卷积网络
│ └── 03_resnet.py # ResNet 残差网络 (99.02%)
├── 05_kaggle/ # Kaggle 比赛
│ ├── digit_recognizer/ # 手写数字识别
│ ├── store_sales/ # 商店销售预测
│ └── titanic/ # 泰坦尼克号生存预测
└── 06_ai_agent/ # AI Agent 开发
├── 01_first_agent.py # Agent 基础概念
└── 02_agent_tools.py # 带工具的 Agent


---

## 📊 项目成果

| 项目 | 模型 | 结果 |
|------|------|------|
| MNIST 手写数字 | 全连接网络 | 96.76% |
| MNIST 手写数字 | CNN | 99.19% |
| **Kaggle Digit Recognizer** | **CNN** | **🏆 第 431 名 (0.99042)** |
| MNIST 手写数字 | ResNet | 99.02% |
| Titanic 生存预测 | 神经网络 | 81.56% |
| Titanic 生存预测 | 随机森林 | 79.11% |
| 商店销售预测 | 神经网络 | RMSE 0.6441 |
| 商店销售预测 | 随机森林 | RMSE 0.5397 |

---

## 🔧 技术栈

**深度学习**
- PyTorch
- 神经网络（全连接、CNN、ResNet）
- 优化器（SGD、Adam）
- 正则化（Dropout、BatchNorm）

**机器学习**
- scikit-learn（Ridge、随机森林、逻辑回归）
- 数据预处理（标准化、编码、缺失值处理）

**AI Agent**
- 大语言模型（LLM）
- AI Agent 核心概念
- 工具调用、思维链（ReAct）

**数据处理**
- Pandas、NumPy
- Matplotlib

---

## 📚 学习路径


第一阶段：神经网络基础
→ 理解前向传播、反向传播、梯度下降

第二阶段：PyTorch 框架
→ 用 nn.Module 重构代码

第三阶段：CNN 图像识别
→ MNIST → Kaggle 比赛

第四阶段：ResNet 残差网络
→ 理解残差连接、深层网络训练

第五阶段：AI Agent
→ Agent 核心概念、工具调用、思维链


---

## 🚀 运行方式

```bash
# 安装依赖
pip install torch torchvision pandas numpy scikit-learn

# 运行示例
python 03_mnist/01_mnist.py
python 03_mnist/03_resnet.py



