# coding: utf-8

import glob
import math
import os
import torch
import ffmpeg as ff
import zhconv
# 全局变量
ffmpeg_path = './ffmpeg'
srtallinput = glob.glob('./SrtFiles/input/*')
srtwavs = glob.glob('./SrtFiles/input/*.wav')
SRTwhisper_model_path = 'models/faster-whisper-large-v3-turbo-ct2'
os.environ['FFMPEG_PATH'] = ffmpeg_path


def audio2wav(audio):
    ff.input(audio).output(f"{os.path.dirname(audio)}/{os.path.basename(audio).split('.')[0]}.wav", y='-y').run()
    print(f"已将{os.path.basename(audio)}转换为{os.path.basename(audio).split('.')[0]}.wav")


def device_detect():
    if torch.cuda.is_available():
        print("检测到可用的GPU，模型将被加载到GPU")
        device = "cuda"
    else:
        print("未检测到可用的GPU，模型将被加载到CPU")
        device = "cpu"

    return device


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
            segments, info = whisper_model.transcribe(audio=wav, beam_size=5, vad_filter=True,
                                                      chunk_length=30,
                                                      vad_parameters=dict(min_silence_duration_ms=500))

        subtitle_file = f"./SrtFiles/output/{os.path.basename(wav).split('.')[0]}-sub-{info.language}.srt"
        print(f"识别到第{index + 1}项音频：{wavpath},其语言为 '%s' ,本次识别的准确度为 %f" % (info.language, info.language_probability))
        print('开始转录')
        for segment in segments:
            count_id = segment.id
            segment_start = format_time(segment.start)
            segment_end = format_time(segment.end)
            segment_text = segment.text
            whisper_message += str(count_id) + "\n"
            whisper_message += f"{segment_start} --> {segment_end} \n"
            whisper_message += f"{segment_text} \n"
            whisper_message += "\n"

        # 繁体中文转简体
        if info.language == "zh":
            print('检测到可能的繁体字，将进行繁简转换，结果将保存为简体字')
            whisper_message = zhconv.convert(whisper_message, locale='zh-hans')
            print('繁简转换完成')

        # 结果文本保存
        print('所有转录已结束,开始保存转录文本')
        f = open(subtitle_file, "w", encoding='utf-8')
        f.write(whisper_message)
        f.close()
        print('srt文件已被保存在： ' + subtitle_file)
        return True
    except Exception as e:
        print(f"Error processing {wav}: {e}")
        return False
