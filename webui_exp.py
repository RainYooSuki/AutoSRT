import gradio as gr
import os
import yaml
from srt_generation_exp import SRTGenerator
import threading
import time
from pathlib import Path
import shutil
import tempfile
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
                'temp_dir': './tmp',
                'model_path': 'models/faster-whisper-large-v3-turbo-ct2'
            },
            'model': {
                'compute_types_gpu': ['bfloat16', 'float16', 'int8', 'float32'],
                'compute_type_cpu': 'int8',
                'num_workers': 12,
                'beam_size': 5,
                'chunk_length': 30,
                'vad_filter': True,
                'min_silence_duration_ms': 500,
                'max_instances': 2  # 新增：最大实例数
            },
            'audio': {
                'supported_extensions': ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma', '.opus']
            }
        }


def process_audio_files_exp(audio_files, output_dir, max_instances, progress=gr.Progress(track_tqdm=True)):
    """
    使用实验性多GPU、多实例功能处理音频文件
    """
    if not audio_files:
        return "请至少上传一个音频文件", ""
    
    if not output_dir.strip():
        output_dir = "./outputs"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建临时输入目录并复制上传的文件
    temp_input_dir = "./temp_web_inputs"
    os.makedirs(temp_input_dir, exist_ok=True)
    
    copied_files = []
    try:
        # 复制上传的文件到临时输入目录
        for audio_file in audio_files:
            original_path = audio_file.name
            filename = os.path.basename(original_path)
            temp_path = os.path.join(temp_input_dir, filename)
            
            # 复制文件
            shutil.copy2(original_path, temp_path)
            copied_files.append(temp_path)
        
        # 创建配置文件的副本并更新max_instances
        config = load_config()
        config['model']['max_instances'] = max_instances
        
        # 创建临时配置文件
        temp_config_path = "./temp_config.yaml"
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # 创建SRTGenerator实例并处理音频
        generator = SRTGenerator(config_path=temp_config_path)
        
        # 处理音频列表
        generator.process_audio_list(copied_files)
        
        # 清理临时配置文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        
        # 移动输出文件到指定输出目录
        output_files = os.listdir("./outputs")
        for output_file in output_files:
            if output_file.endswith(".srt"):
                src_path = os.path.join("./outputs", output_file)
                dst_path = os.path.join(output_dir, output_file)
                if src_path != dst_path:
                    shutil.move(src_path, dst_path)
        
        # 返回成功消息
        success_msg = f"✓ 成功处理 {len(audio_files)} 个音频文件，使用了 {min(len(audio_files), max_instances)} 个模型实例，输出保存到: {output_dir}"
        return success_msg, output_dir
        
    except Exception as e:
        error_msg = f"处理过程中出错: {str(e)}"
        return error_msg, output_dir
    finally:
        # 清理临时输入目录
        if os.path.exists(temp_input_dir):
            shutil.rmtree(temp_input_dir, ignore_errors=True)


def create_gradio_interface():
    """创建Gradio界面"""
    config = load_config()
    
    with gr.Blocks(
        title="AutoSRT - WebUI - EXP"
    ) as demo:
        gr.Markdown("# AutoSRT - WebUI - EXP")
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
                
                # 最大实例数设置
                max_instances = gr.Slider(
                    label="最大并发实例数",
                    minimum=1,
                    maximum=8,
                    value=config['model'].get('max_instances', 2),
                    step=1,
                    info="设置同时运行的模型实例数量，多GPU环境下可设置更高值"
                )
                
                # GPU数量显示
                gpu_count = torch.cuda.device_count()
                gr.Markdown(f"**检测到 {gpu_count} 个GPU设备**")
                
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
            fn=process_audio_files_exp,
            inputs=[audio_files, output_dir, max_instances],
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
        3. 调整最大并发实例数（根据GPU数量和内存调整）
        4. 点击"开始多GPU处理"按钮
        5. 等待处理完成，结果将在下方显示
        """)
    
    return demo


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(
        share=False,  # 生成公共链接
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7861,  # 使用不同端口避免冲突
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
            spacing_size="md",
            radius_size="lg"
        )
    )