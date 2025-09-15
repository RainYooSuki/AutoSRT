# AutoSRT🚀

AutoSRT 是一个基于语音识别的自动化字幕生成工具，可将音频内容转换为结构化字幕文件（如 SRT 格式），支持中文繁简转换，提升视频字幕制作效率。

## 功能特性

- **音频处理**：使用 FFmpeg 进行音频格式转换与提取
- **语音识别**：采用 faster-whisper 模型进行高精度语音转文本
- **字幕生成**：自动生成标准 SRT 格式字幕文件
- **中文支持**：自动检测并转换繁体中文为简体中文
- **GPU 加速**：支持 CUDA 加速，大幅提升处理速度
- **批量处理**：支持多线程并发处理多个音频文件

## 技术架构

- **核心语言**：Python 3.8+
- **语音识别**：faster-whisper >= 0.6.0
- **音频处理**：FFmpeg
- **AI 框架**：PyTorch (支持 CUDA 11.3+)
- **中文转换**：zhconv >= 1.4.0

## 目录结构

```
AutoSRT/
├── models/                 # faster-whisper 模型文件
│   └── your-faster-whisper-model/
│               └── model.bin
│               └── ...
├── SrtFiles/               # 输入输出文件夹
│   ├── input/              # 音频输入文件夹
│   └── output/             # 字幕输出文件夹
├── module.py               # 核心功能模块
├── srt-generation.py       # 主程序入口
└── requirements.txt        # 项目依赖列表
```

## 环境要求

- Python 3.8 或更高版本
- CUDA 11.3+（推荐，用于 GPU 加速）

## 安装步骤

1. 克隆或下载本项目到本地
2. 安装 Python 3.8+
3. 安装项目依赖：
   ```bash
   pip install -r requirements.txt
   ```
4. 下载 faster-whisper 模型文件并放置到 [models](./models) 目录
   例如：[Faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3)

## 使用方法

1. 将需要转换的音频文件放入 [SrtFiles/input](./SrtFiles/input) 目录
2. 运行主程序：
   ```bash
   python srt-generation.py
   ```
3. 程序将自动处理所有音频文件并将生成的字幕保存到 [SrtFiles/output](./SrtFiles/output) 目录

## 处理流程

1. 程序启动时自动检测可用的计算设备（GPU/CPU）
2. 自动将非 WAV 格式的音频文件转换为 WAV 格式
3. 使用 faster-whisper 模型对音频进行语音识别
4. 自动生成 SRT 格式字幕文件
5. 对于中文内容，自动进行繁简转换
6. 将结果保存到输出目录

## 注意事项

- 推荐使用 GPU 运行以获得最佳性能
- 大型模型对内存要求较高，请确保有足够的系统内存
- 模型文件较大，请确保有足够的存储空间
- 确保模型版本与 faster-whisper 版本兼容

## 许可证

本项目采用 MIT 许可证，详情请见 [LICENSE](./LICENSE) 文件。
