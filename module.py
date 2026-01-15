# coding: utf-8

import glob
import os
import yaml
import torch
import ffmpeg as ff
import zhconv

def load_config(config_path='./config/config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# Load configuration
config = load_config()

# Global variables from config
SRTwhisper_model_path = config['paths']['model_path']
INPUT_DIR = config['paths']['input_dir']
OUTPUT_DIR = config['paths']['output_dir']
TEMP_DIR = config['paths']['temp_dir']

# Initialize input files (these will be updated as needed in srt-generation.py)
srtallinput = glob.glob(f'{INPUT_DIR}/*')
srtwavs = glob.glob(f'{INPUT_DIR}/*.wav')


def device_detect():
    if torch.cuda.is_available():
        print("检测到可用的NVIDIA GPU，模型将被加载到NVIDIA GPU")
        device = "cuda"
    else:
        print("未检测到可用的NVIDIA GPU，模型将被加载到CPU")
        device = "cpu"

    return device


def audio2wav(audio):
    """将音频文件转换为WAV格式"""
    try:
        # 使用 os.path.join 来正确构建路径，兼容不同操作系统
        output_path = os.path.join(os.path.dirname(audio), f"{os.path.splitext(os.path.basename(audio))[0]}.wav")
        
        # 运行ffmpeg命令进行转换
        ff.input(audio).output(output_path, y='-y').run(overwrite_output=True)
        
        print(f"已将{os.path.basename(audio)}转换为{os.path.basename(output_path)}")
        return output_path
    except ff.Error as e:
        print(f"FFmpeg错误，无法转换音频文件 {os.path.basename(audio)}: {e.stderr.decode()}")
        raise e
    except Exception as e:
        print(f"无法转换音频文件 {os.path.basename(audio)}: {str(e)}")
        raise e


def format_time(seconds):
    """将秒数格式化为 SRT 字幕时间格式 (HH:MM:SS,mmm)"""
    # 使用 divmod 函数减少重复计算，提高性能
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    
    # 计算毫秒部分
    milliseconds = int((secs - int(secs)) * 1000)
    
    # 确保毫秒不超过999，防止四舍五入导致进位
    if milliseconds >= 1000:
        milliseconds = 999
        secs = int(secs) + 1
        if secs >= 60:
            secs = 0
            minutes += 1
            if minutes >= 60:
                minutes = 0
                hours += 1
    else:
        secs = int(secs)
    
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:02d},{milliseconds:03d}"


def process_audio(index, wav, total_count, whisper_model, model_lock):
    """处理单个音频文件，生成对应的 SRT 字幕文件"""
    try:
        wavpath = os.path.join(os.path.dirname(wav), os.path.basename(wav))
        print(f"开始处理第{index + 1}项音频：{wavpath}，总已开始处理进度:{index + 1}/{total_count}\n")
        
        # 使用线程锁保护模型访问
        with model_lock:
            segments, info = whisper_model.transcribe(
                audio=wav, 
                beam_size=config['model']['beam_size'], 
                vad_filter=config['model']['vad_filter'],
                chunk_length=config['model']['chunk_length'],
                vad_parameters=dict(min_silence_duration_ms=config['model']['min_silence_duration_ms'])
            )
            language = info.language  # 获取语言信息
            language_probability = getattr(info, 'language_probability', 'N/A')  # 获取语言识别准确度

        # 构建输出文件路径
        output_filename = f"{os.path.splitext(os.path.basename(wav))[0]}-sub-{language}.srt"
        subtitle_file = os.path.join(OUTPUT_DIR, output_filename)
        
        print(f"[第{index + 1}项音频：{os.path.basename(wav)}]:[语言: '{language}' ],[识别准确度: {language_probability}]")
        
        # 生成字幕内容
        whisper_message = generate_subtitle_content(segments)
        
        # 如果是中文，进行繁简转换
        if language == "zh":
            print('检测到可能的繁体字，将进行繁简转换，结果将保存为简体字')
            whisper_message = zhconv.convert(whisper_message, locale='zh-hans')
            print('繁简转换完成')

        # 保存字幕文件
        save_subtitle_file(subtitle_file, whisper_message)
        print(f"第{index + 1}项音频已保存到：{subtitle_file}")
        return True
    except Exception as e:
        print(f"Error processing {wav}: {e}")
        return False


def generate_subtitle_content(segments):
    """根据转录结果生成字幕内容"""
    # 使用列表推导式和生成器表达式提高性能
    content_parts = [
        f"{segment.id}\n{format_time(segment.start)} --> {format_time(segment.end)}\n{segment.text.strip()}\n"
        for segment in segments
    ]
    
    # 使用换行符连接所有部分，并在最后添加换行符以符合SRT格式
    return "\n".join(content_parts) + "\n"


def save_subtitle_file(file_path, content):
    """保存字幕文件"""
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(content)





