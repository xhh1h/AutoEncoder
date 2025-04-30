# 隐私保护人脸匿名化系统

这个项目实现了一个基于深度学习的人脸匿名化系统，通过差分隐私技术保护用户隐私。

## 功能特点

- 自动检测图像中的人脸
- 使用编码器-解码器架构处理人脸图像
- 应用差分隐私技术增强隐私保护
- 生成匿名化后的图像

## 系统要求

- Python 3.6+
- PyTorch
- CUDA支持（推荐但非必需）
- FaceNet PyTorch
- Pillow
- NumPy

## 安装

```bash
pip install torch torchvision facenet-pytorch pillow numpy
```

## 使用方法

1. 将测试图像放在项目根目录下，命名为`test.jpg`
2. 运行主程序：

```bash
python main.py
```

3. 匿名化后的图像将保存在`outputs/anonymized.png`

## 项目结构

- `main.py`: 主程序，处理图像和匿名化流程
- `models.py`: 包含编码器和解码器模型定义
- `dp_utils.py`: 差分隐私相关工具函数
- `face_utils.py`: 人脸检测工具函数
- `checkpoints/`: 保存预训练模型权重
- `outputs/`: 保存生成的匿名化图像

## 工作原理

1. 使用MTCNN检测图像中的人脸
2. 将每个人脸区域裁剪并输入到编码器
3. 编码器将人脸压缩为潜在特征表示
4. 应用差分隐私噪声到潜在特征
5. 解码器重建带有隐私保护的人脸
6. 将重建的人脸放回原始图像位置

## 许可证
