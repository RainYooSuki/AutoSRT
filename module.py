# coding: utf-8

import glob
import math
import os
import yaml
import torch
import ffmpeg as ff
import zhconv

# Load configuration
with open('./config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Global variables from config
SRTwhisper_model_path = config['paths']['model_path']
INPUT_DIR = config['paths']['input_dir']
OUTPUT_DIR = config['paths']['output_dir']

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
    try:
        ff.input(audio).output(f"{os.path.dirname(audio)}/{os.path.splitext(os.path.basename(audio))[0]}.wav", y='-y').run()
        print(f"已将{os.path.basename(audio)}转换为{os.path.splitext(os.path.basename(audio))[0]}.wav")
    except Exception as e:
        print(f"无法转换音频文件 {os.path.basename(audio)}: {e}")
        raise e


def format_time(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600  # 用取余操作保留不足1小时的剩余秒数（即去掉整小时部分）
    minutes = math.floor(seconds / 60)
    seconds %= 60  # 再次取余，保留不足1分钟的秒数(含小数部分)
    milliseconds = round((seconds - math.floor(seconds)) * 1000)  # 不足1分钟的秒数中的小数部分转毫秒
    seconds = math.floor(seconds)  # 不足1分钟的秒数中的整数部分
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:01d},{milliseconds:03d}"

    return formatted_time


def process_audio(index, wav, total_count, whisper_model, model_lock):
    wavpath = os.path.dirname(wav) + '/' + os.path.basename(wav)
    try:
        print(f"开始处理第{index + 1}项音频：{wavpath}，总已开始处理进度:{index + 1}/{total_count}\n")
        # 转录
        whisper_message = ''

        # 使用线程锁保护模型访问
        with model_lock:
            segments, info = whisper_model.transcribe(
                audio=wav, 
                beam_size=config['model']['beam_size'], 
                vad_filter=config['model']['vad_filter'],
                chunk_length=config['model']['chunk_length'],
                vad_parameters=dict(min_silence_duration_ms=config['model']['min_silence_duration_ms'])
            )

        subtitle_file = f"{OUTPUT_DIR}/{os.path.splitext(os.path.basename(wav))[0]}-sub-{info.language}.srt"
        print(f"[第{index + 1}项音频：{os.path.basename(wav)}]:[语言: '%s' ],[识别准确度: %f]" % (
            info.language, info.language_probability))
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

        # 结果文本保存
        with open(subtitle_file, "w", encoding='utf-8') as f:
            f.write(whisper_message)
        print(f"第{index + 1}项音频已保存到：{subtitle_file}")
        return True
    except Exception as e:
        print(f"Error processing {wav}: {e}")
        return False





