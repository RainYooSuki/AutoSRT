# AutoSRT🚀

AutoSRT 是一个基于语音识别的多语种自动化字幕生成工具，可将音频内容转换为SRT字幕文件，支持中文繁简转换，提升视频字幕制作效率。

## 功能特性

- **音频处理**：使用 FFmpeg 进行音频格式转换与提取
- **语音识别**：采用 faster-whisper 模型进行高精度语音转文本
- **字幕生成**：自动生成标准 SRT 格式字幕文件
- **中文支持**：自动检测并转换繁体中文为简体中文
- **GPU 加速**：支持 CUDA 加速，大幅提升处理速度
- **批量处理**：支持多线程并发处理多个音频文件
- **灵活配置**：通过 YAML 配置文件管理所有参数

## 技术架构

- **核心语言**：Python 3.8+
- **语音识别**：faster-whisper >= 0.6.0
- **音频处理**：FFmpeg
- **AI 框架**：PyTorch (支持 CUDA 11.3+)
- **中文转换**：zhconv >= 1.4.0
- **配置管理**：PyYAML

## 目录结构

```
AutoSRT/
├── config/                 # 配置文件目录
│   └── config.yaml         # 项目配置文件
├── inputs/                 # 音频输入文件夹
├── outputs/                # 字幕输出文件夹
├── models/                 # faster-whisper 模型文件
│   └── your-faster-whisper-model/
│               └── model.bin
│               └── ...
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

   ```bash
   # 单独下载torch
   # CUDA 11.8 as example
   pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118

   ```
4. （可选）根据需要编辑 [config/config.yaml](./config/config.yaml) 文件以自定义配置
5. 下载 faster-whisper 模型文件并放置到 [models](./models) 目录
   例如：
   [Faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3)
   [Faster-whisper-large-v3-turbo-ct2](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo)

## 配置说明

项目的所有配置都集中管理在 [config/config.yaml](./config/config.yaml) 文件中，主要包括以下几个部分：

- **paths**: 定义输入输出目录和模型路径（默认为 [./inputs](./inputs) 和 [./outputs](./outputs)）
- **audio**: 支持的音频格式扩展名
- **model**: 模型相关参数，包括GPU/CPU计算类型、工作线程数等
- **processing**: 处理相关的配置参数

可以根据需要编辑此文件来自定义程序行为。

## 使用方法

1. 将需要转换的音频文件放入 [inputs](./inputs) 目录
2. 运行主程序：
   ```bash
   python srt-generation.py
   ```
3. 程序将自动处理所有音频文件并将生成的字幕保存到 [outputs](./outputs) 目录

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


## [Web界面](./WEBUI_README.md)

本项目提供了一个基于Gradio的Web界面，让您可以通过浏览器轻松使用AutoSRT功能：

1. 安装依赖：
   ```bash
   pip install gradio
   ```

2. 启动WebUI：
   ```bash
   python webui.py
   ```


3. 在浏览器中访问 `http://127.0.0.1:7860`

## 许可证

本项目采用 MIT 许可证，详情请见 [LICENSE](./LICENSE) 文件。
