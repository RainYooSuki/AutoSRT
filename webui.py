import gradio as gr
import os
import yaml
from module import device_detect, audio2wav, process_audio
from faster_whisper import WhisperModel
import threading
import time
from pathlib import Path
import shutil
import tempfile
import math
import zhconv
import torch
import ffmpeg as ff


def load_config():
    """加载配置文件"""
    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        # 默认配置
        return {
            'paths': {
                'input_dir': './inputs',
                'output_dir': './outputs',
                'model_path': 'models/faster-whisper-large-v3-turbo-ct2'
            },
            'model': {
                'compute_types_gpu': ['bfloat16', 'float16', 'float32'],
                'compute_type_cpu': 'int8',
                'num_workers': 12,
                'beam_size': 5,
                'chunk_length': 30,
                'vad_filter': True,
                'min_silence_duration_ms': 500
            },
            'audio': {
                'supported_extensions': ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.opus']
            }
        }


def format_time(seconds):
    """格式化时间戳为SRT格式"""
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:01d},{milliseconds:03d}"
    return formatted_time


def transcribe_audio_single(input_audio_path, output_dir, progress=gr.Progress(track_tqdm=True)):
    """
    转录音频文件的函数（单个文件处理）
    """
    # 加载配置
    config = load_config()
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保临时目录存在
    tmp_dir = "./tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_audio_path):
        return f"错误: 输入音频文件不存在: {input_audio_path}", output_dir
    
    # 检查文件扩展名是否受支持
    _, ext = os.path.splitext(input_audio_path)
    if ext.lower() not in config['audio']['supported_extensions']:
        return f"错误: 不支持的音频格式: {ext}", output_dir
    
    try:
        # 如果不是WAV格式，则转换为WAV
        temp_wav_path = None
        if ext.lower() != '.wav':
            # 创建临时WAV文件
            temp_wav_path = os.path.join("./tmp", 
                                         f"{os.path.splitext(os.path.basename(input_audio_path))[0]}_{int(time.time())}.wav")
            ff.input(input_audio_path).output(temp_wav_path, y='-y').run()
            audio_for_processing = temp_wav_path
        else:
            audio_for_processing = input_audio_path
        
        # 检测设备
        device = device_detect()
        
        # 加载模型
        whisper_model = None
        model_path = config['paths']['model_path']
        
        if device == 'cuda':
            # 尝试按照优先级顺序加载GPU模型
            for compute_type in config['model']['compute_types_gpu']:
                try:
                    whisper_model = WhisperModel(
                        model_size_or_path=model_path,
                        device=device,
                        local_files_only=True,
                        compute_type=compute_type,
                        num_workers=config['model'].get('num_workers', 12)
                    )
                    print(f'成功加载模型到GPU: {compute_type.upper()}')
                    break
                except Exception as e:
                    print(f'无法使用 {compute_type} 加载模型到GPU，尝试下一个...')
                    continue
            
            # 如果GPU上所有类型都失败，则使用CPU
            if whisper_model is None:
                whisper_model = WhisperModel(
                    model_size_or_path=model_path,
                    device="cpu",
                    local_files_only=True,
                    compute_type=config['model']['compute_type_cpu'],
                    num_workers=config['model'].get('num_workers', 12)
                )
                print(f'使用CPU加载模型: {config["model"]["compute_type_cpu"]}')
        else:
            # CPU设备
            whisper_model = WhisperModel(
                model_size_or_path=model_path,
                device=device,
                local_files_only=True,
                compute_type=config['model']['compute_type_cpu'],
                num_workers=config['model'].get('num_workers', 12)
            )
            print(f'使用CPU加载模型: {config["model"]["compute_type_cpu"]}')
        
        print(f"开始处理音频：{audio_for_processing}")
        
        # 转录
        segments, info = whisper_model.transcribe(
            audio=audio_for_processing,
            beam_size=config['model']['beam_size'],
            vad_filter=config['model']['vad_filter'],
            chunk_length=config['model']['chunk_length'],
            vad_parameters=dict(min_silence_duration_ms=config['model']['min_silence_duration_ms'])
        )
        
        # 生成SRT文件
        base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
        subtitle_file = f"{output_dir}/{base_name}-sub-{info.language}.srt"
        
        print(f"[音频：{os.path.basename(input_audio_path)}]:[语言: '{info.language}' ],[识别准确度: {info.language_probability}]")
        
        # 生成字幕内容
        whisper_message = ''
        for segment in segments:
            count_id = segment.id
            segment_start = format_time(segment.start)
            segment_end = format_time(segment.end)
            segment_text = segment.text
            whisper_message += f"{count_id}\n{segment_start} --> {segment_end}\n{segment_text}\n\n"

        # 繁体中文转简体
        if info.language == "zh":
            print('检测到可能的繁体字，将进行繁简转换，结果将保存为简体字')
            whisper_message = zhconv.convert(whisper_message, locale='zh-hans')
            print('繁简转换完成')

        # 保存字幕文件
        with open(subtitle_file, "w", encoding='utf-8') as f:
            f.write(whisper_message)
        print(f"音频已保存到：{subtitle_file}")
        
        # 清理临时文件
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        
        success_msg = f"✓ {os.path.basename(input_audio_path)}: 处理成功，已保存到 {subtitle_file}"
        
        return success_msg, output_dir
        
    except Exception as e:
        # 清理临时文件
        if 'temp_wav_path' in locals() and temp_wav_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        error_msg = f"✗ {os.path.basename(input_audio_path)}: 处理出错 - {str(e)}"
        return error_msg, output_dir


def process_audio_files(audio_files, output_dir, progress=gr.Progress(track_tqdm=True)):
    """
    处理音频文件的函数
    """
    if not audio_files:
        return "请至少上传一个音频文件", ""
    
    if not output_dir.strip():
        output_dir = "./outputs"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for idx, audio_file in enumerate(audio_files):
        progress(idx / len(audio_files), desc=f"处理第 {idx+1}/{len(audio_files)} 个文件...")
        
        try:
            # 调用单个音频处理函数
            result, _ = transcribe_audio_single(audio_file.name, output_dir, progress)
            results.append(result)
        except Exception as e:
            results.append(f"✗ {os.path.basename(audio_file.name)}: 处理出错 - {str(e)}")
    
    progress(1.0, desc="处理完成!")
    return "\n\n".join(results), output_dir


def create_gradio_interface():
    """创建Gradio界面"""
    config = load_config()
    
    with gr.Blocks(
        title="AutoSRT - 自动字幕生成工具"
    ) as demo:
        gr.Markdown("# AutoSRT - 自动字幕生成工具")
        gr.Markdown("将音频文件转换为SRT格式的字幕文件")
        
        with gr.Row():
            with gr.Column(scale=2):
                # 文件上传区域
                audio_files = gr.File(
                    label="上传音频文件",
                    file_count="multiple",
                    file_types=["audio"],
                    elem_id="audio_upload"
                )
                
                # 输出目录设置
                output_dir = gr.Textbox(
                    label="输出目录",
                    value=config['paths'].get('output_dir', './outputs'),
                    placeholder="例如: ./outputs 或 D:/my_subtitles"
                )
                
                # 处理按钮
                submit_btn = gr.Button("开始处理", variant="primary")
                
                # 预设模型路径信息
                gr.Markdown(f"**当前模型路径**: `{config['paths'].get('model_path', 'models/faster-whisper-large-v3-turbo-ct2')}`")
                
            with gr.Column(scale=3):
                # 结果显示区域
                result_text = gr.Textbox(
                    label="处理结果",
                    lines=15,
                    interactive=False
                )
                
                # 输出目录显示
                output_folder = gr.Textbox(
                    label="输出文件夹",
                    interactive=False
                )
        
        # 事件绑定
        submit_btn.click(
            fn=process_audio_files,
            inputs=[audio_files, output_dir],
            outputs=[result_text, output_folder]
        )
        
        # 示例说明
        gr.Markdown("""
        **提示**: 在 `inputs` 文件夹中放置音频文件以供处理，支持的格式包括: MP3, WAV, M4A, FLAC, AAC, OGG, WMA, OPUS
        """)
        
        # 页脚信息
        gr.Markdown("""
        ---
        **使用说明**:
        1. 上传一个或多个音频文件
        2. 设置输出目录（可选，默认为 ./outputs）
        3. 点击"开始处理"按钮
        4. 等待处理完成，结果将在下方显示
        """)
    
    return demo


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(
        share=True,  # 生成公共链接
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,  # 端口
        inbrowser=True,  # 自动打开浏览器
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
            spacing_size="md",
            radius_size="lg"
        )
    )