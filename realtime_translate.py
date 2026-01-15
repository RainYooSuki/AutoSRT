# coding: utf-8
"""
AutoSRT Realtime Translate
"""

import os
import sys
import threading
import queue
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import tempfile
import torch
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# 导入 module.py 中的功能
sys.path.append('.')
from module import device_detect, format_time, process_audio, load_config, config, generate_subtitle_content,TEMP_DIR,OUTPUT_DIR

# 全局变量
SAMPLE_RATE = 16000  # 采样率，Whisper推荐使用16kHz
CHUNK_DURATION = 5  # 每个音频块的持续时间（秒）
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION  # 每个音频块的样本数


# 确保临时目录存在
os.makedirs(TEMP_DIR, exist_ok=True)

class RealtimeTranslator:
    def __init__(self):
        """
        初始化实时翻译器
        """
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.model = None
        self.model_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=3)  # 线程池，最多3个并发转录任务
        self.temp_files = []  # 存储临时文件路径
        self.results_queue = queue.Queue()  # 用于存储处理结果，保持时间顺序
        self.processed_ids = set()  # 已处理的ID集合，防止重复
        self.global_start_time = time.time()  # 记录程序开始时间
        self.total_processed_duration = 0.0  # 总共已处理的音频时长
        
        # 检测设备
        self.device = device_detect()
        
        # 加载模型
        self.load_whisper_model()
        
    def load_whisper_model(self):
        """加载Faster-Whisper模型"""
        try:
            from faster_whisper import WhisperModel
            model_path = config['paths']['model_path']
            
            # 根据设备类型加载模型
            if self.device == 'cuda':
                # 从配置中获取GPU支持的计算类型
                gpu_compute_types = config['model']['compute_types_gpu']
                
                # 尝试每种计算类型直到成功
                model_loaded = False
                for compute_type in gpu_compute_types:
                    try:
                        # 尝试加载模型到GPU
                        self.model = WhisperModel(
                            model_size_or_path=model_path,
                            device=self.device,
                            local_files_only=True,
                            compute_type=compute_type,
                            num_workers=12)  # 实时处理不需要太多worker
                        print(f'成功加载模型到GPU，计算类型: {compute_type.upper()}')
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f'无法使用 {compute_type} 加载模型到GPU，尝试下一个...')
                        continue
                
                if not model_loaded:
                    # 如果所有GPU精度都失败，则抛出错误
                    raise RuntimeError("无法在GPU上加载模型，所有精度类型均失败")
            else:
                # 以CPU配置加载模型
                self.model = WhisperModel(
                    model_size_or_path=model_path,
                    device=self.device,
                    local_files_only=True,
                    compute_type=config['model']['compute_type_cpu'],
                    num_workers=1)  # 实时处理不需要太多worker
                print(f'成功加载模型到CPU，计算类型: {config["model"]["compute_type_cpu"].upper()}')
            
            print("Faster-Whisper模型加载完成")
        except Exception as e:
            print(f"加载Faster-Whisper模型失败: {e}")
            raise e

    def find_audio_device(self):
        """
        查找合适的麦克风设备
        """
        devices = sd.query_devices()
        print("可用的音频设备:")
        for i, device in enumerate(devices):
            print(f"{i}: {device['name']} (输入通道: {device['max_input_channels']}, 输出通道: {device['max_output_channels']})")
        
        # 严格查找麦克风设备
        for i, device in enumerate(devices):
            if device['max_input_channels'] >= 1:
                device_name = device['name'].lower()
                # 专门查找麦克风设备，排除系统音频设备
                mic_keywords = ['mic', 'microphone', '麦克风', 'input']
                exclude_keywords = ['stereo mix', '立体声混音', 'loopback', 'output', 'desktop', 'what u hear', 'wave', 'bus', 'virtual', 'cable', '扬声器']
                
                # 检查是否包含麦克风关键词，但不包含系统音频关键词
                has_mic_keyword = any(keyword in device_name for keyword in mic_keywords)
                has_exclude_keyword = any(keyword in device_name for keyword in exclude_keywords)
                
                if has_mic_keyword and not has_exclude_keyword:
                    print(f"找到麦克风设备: {device['name']} (ID: {i})")
                    return i
        
        # 如果没找到明确的麦克风设备，再查找包含"input"但不含排除词的设备
        for i, device in enumerate(devices):
            if device['max_input_channels'] >= 1:
                device_name = device['name'].lower()
                has_input = 'input' in device_name
                has_exclude_keyword = any(keyword in device_name for keyword in exclude_keywords)
                
                if has_input and not has_exclude_keyword:
                    print(f"找到输入设备: {device['name']} (ID: {i})")
                    return i
        
        # 如果仍然找不到设备，返回第一个有输入通道的设备（仅当没有其他选项时）
        for i, device in enumerate(devices):
            if device['max_input_channels'] >= 1:
                device_name = device['name'].lower()
                # 排除明显的系统音频设备
                exclude_keywords = ['stereo mix', '立体声混音', 'loopback', 'output', 'desktop', 'what u hear', 'wave', 'bus', 'virtual', 'cable', '扬声器']
                has_exclude_keyword = any(keyword in device_name for keyword in exclude_keywords)
                if has_exclude_keyword:
                    continue  # 跳过系统音频设备
                
                print(f"使用默认输入设备: {device['name']} (ID: {i})")
                return i
        
        raise Exception("找不到支持音频输入的设备")

    def audio_callback(self, indata, frames, time, status):
        """音频回调函数，用于捕获音频数据"""
        if status:
            print(f"音频状态: {status}")
        
        # 将音频数据复制到队列中
        audio_data = indata.copy()
        self.audio_queue.put(audio_data)

    def start_listening(self):
        """开始监听麦克风音频"""
        try:
            device_id = self.find_audio_device()
            self.is_recording = True
            
            print(f"开始监听麦克风，设备ID: {device_id}")
            
            # 启动音频流
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                device=device_id,
                channels=1,
                dtype='float32',
                callback=self.audio_callback
            )
            
            with stream:
                print(f"麦克风监听已启动...")
                self.process_audio_stream()
                
        except Exception as e:
            print(f"监听麦克风时发生错误: {e}")
            self.is_recording = False

    def process_audio_stream(self):
        """处理音频流，使用缓冲机制累积音频数据后再转录"""
        accumulated_audio = np.array([], dtype=np.float32)  # 累积音频数据
        silence_threshold = 0.001  # 静音阈值，降低以提高对小声音的敏感度
        min_speech_duration = 1.5  # 最小语音持续时间（秒），避免过短的语音片段
        max_buffer_duration = 10.0  # 最大缓冲持续时间（秒），避免过长的累积
        
        try:
            while self.is_recording:
                try:
                    # 从队列获取音频数据
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                    
                    if audio_chunk is not None:
                        # 累积音频数据
                        accumulated_audio = np.concatenate([accumulated_audio, audio_chunk[:, 0]])
                        
                        # 计算当前累积音频的持续时间
                        current_duration = len(accumulated_audio) / SAMPLE_RATE
                        
                        # 计算音频能量（RMS）
                        rms = np.sqrt(np.mean(audio_chunk[:, 0] ** 2))
                        
                        # 如果当前累积音频超过最大缓冲时间，或者检测到相对长时间的静音，则进行转录
                        if (current_duration >= max_buffer_duration or 
                           (rms < silence_threshold and current_duration >= min_speech_duration)):
                            
                            if current_duration >= min_speech_duration:
                                # 将累积的音频数据写入临时文件
                                temp_filename = os.path.join(TEMP_DIR, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav")
                                
                                # 转换音频数据格式
                                audio_int16 = (accumulated_audio * 32767).astype(np.int16)
                                wav_write(temp_filename, SAMPLE_RATE, audio_int16)
                                
                                self.temp_files.append(temp_filename)
                                
                                # 限制临时文件数量，避免占用过多磁盘空间
                                if len(self.temp_files) > 10:
                                    old_file = self.temp_files.pop(0)
                                    if os.path.exists(old_file):
                                        os.remove(old_file)
                                
                                # 提交转录任务到线程池
                                future = self.executor.submit(self.transcribe_audio, temp_filename)
                            
                            # 重置累积音频数据
                            accumulated_audio = np.array([], dtype=np.float32)
                        # 如果检测到静音但音频太短，继续累积
                        elif rms < silence_threshold and current_duration < min_speech_duration:
                            continue
                        # 否则继续累积音频数据
                        else:
                            continue
                        
                except queue.Empty:
                    # 即使队列为空，也检查是否积累了足够的音频数据
                    if len(accumulated_audio) > 0:
                        current_duration = len(accumulated_audio) / SAMPLE_RATE
                        if current_duration >= min_speech_duration:
                            # 将累积的音频数据写入临时文件
                            temp_filename = os.path.join(TEMP_DIR, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav")
                            
                            # 转换音频数据格式
                            audio_int16 = (accumulated_audio * 32767).astype(np.int16)
                            wav_write(temp_filename, SAMPLE_RATE, audio_int16)
                            
                            self.temp_files.append(temp_filename)
                            
                            # 限制临时文件数量，避免占用过多磁盘空间
                            if len(self.temp_files) > 10:
                                old_file = self.temp_files.pop(0)
                                if os.path.exists(old_file):
                                    os.remove(old_file)
                            
                            # 提交转录任务到线程池
                            future = self.executor.submit(self.transcribe_audio, temp_filename)
                            
                            # 重置累积音频数据
                            accumulated_audio = np.array([], dtype=np.float32)
                    
                    continue
                    
        except KeyboardInterrupt:
            print("用户中断监听")
            # 如果有累积的音频数据，在退出前进行转录
            if len(accumulated_audio) > 0:
                current_duration = len(accumulated_audio) / SAMPLE_RATE
                if current_duration >= min_speech_duration:
                    temp_filename = os.path.join(TEMP_DIR, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav")
                    
                    # 转换音频数据格式
                    audio_int16 = (accumulated_audio * 32767).astype(np.int16)
                    wav_write(temp_filename, SAMPLE_RATE, audio_int16)
                    
                    self.temp_files.append(temp_filename)
                    future = self.executor.submit(self.transcribe_audio, temp_filename)
        finally:
            self.is_recording = False
            # 关闭线程池并等待所有任务完成
            self.executor.shutdown(wait=True)
            # 清理临时文件
            for temp_file in self.temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    def transcribe_audio(self, audio_path):
        """转录音频文件"""
        try:
            print(f"开始转录音频: {audio_path}")
            
            # 获取音频文件的持续时间
            import wave
            with wave.open(audio_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                audio_duration = frames / float(rate)
            
            # 使用模型锁保护模型访问
            with self.model_lock:
                # 在实时转录中使用适中的静音检测参数，平衡语音截断和小声音识别
                segments, info = self.model.transcribe(
                    audio_path,
                    beam_size=config['model']['beam_size'],
                    vad_filter=config['model']['vad_filter'],
                    chunk_length=config['model']['chunk_length'],
                    word_timestamps=False,  # 不需要词级别时间戳
                    vad_parameters=dict(min_silence_duration_ms=1000)  # 使用1秒静音检测阈值，平衡截断和小声音识别
                )
                
                # 将segments转换为列表以便处理
                segment_list = list(segments)
                
                # 只处理有内容的段
                if not segment_list:
                    print("转录内容: 未检测到语音")
                    # 即使没有检测到语音，也需要更新总处理时长
                    with threading.Lock():  # 使用锁确保线程安全
                        self.total_processed_duration += audio_duration
                    return
                
                # 创建模拟的segments对象以匹配generate_subtitle_content函数的期望
                import types
                processed_segments = []
                for i, segment in enumerate(segment_list):
                    seg_obj = types.SimpleNamespace()
                    seg_obj.id = i + 1
                    seg_obj.start = segment.start
                    seg_obj.end = segment.end
                    seg_obj.text = segment.text
                    processed_segments.append(seg_obj)
                
                # 将本地时间戳转换为全局时间戳（加上已处理的总时长作为偏移量）
                global_offset = self.total_processed_duration
                for segment in processed_segments:
                    segment.start += global_offset
                    segment.end += global_offset
                
                # 处理字幕段，确保它们之间有适当的间隔
                processed_segments_with_gaps = []
                for i, segment in enumerate(processed_segments):
                    start_time = segment.start
                    end_time = segment.end
                    
                    # 如果不是第一个段且存在前一段，则调整开始时间以确保最小间隔
                    if i > 0:
                        prev_segment = processed_segments_with_gaps[-1]  # 使用已经处理过的段
                        prev_end_time = prev_segment.end
                        
                        # 如果当前段开始时间太接近前一段的结束时间，则调整
                        if start_time - prev_end_time < 0.3:  # 0.3秒最小间隔
                            start_time = prev_end_time + 0.3
                            
                            # 同样确保结束时间不早于新的开始时间
                            if end_time < start_time:
                                end_time = start_time + max(segment.end - segment.start, 0.5)  # 确保至少0.5秒的持续时间
                    else:
                        # 对于第一个段，确保有合理的持续时间
                        if end_time - start_time < 0.5:  # 最小0.5秒持续时间
                            end_time = start_time + 0.5
                    
                    # 创建新的段落对象，具有调整后的时间
                    adjusted_segment = type('Segment', (), {
                        'start': start_time,
                        'end': end_time,
                        'text': segment.text,
                        'id': segment.id
                    })()
                    
                    processed_segments_with_gaps.append(adjusted_segment)
                
                # 生成字幕内容
                subtitle_content = generate_subtitle_content(processed_segments_with_gaps)
                
                # 使用基于日期的单一文件名
                date_str = datetime.now().strftime('%Y-%m-%d')
                output_filename = f"realtime_transcription_{date_str}.srt"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                # 获取当前文件的最大ID，用于连续编号
                current_max_id = 0
                if os.path.exists(output_path):
                    with open(output_path, "r", encoding='utf-8') as f:
                        content = f.read()
                        # 解析现有内容，找出最大的ID
                        import re
                        ids = re.findall(r'^(\d+)$', content, re.MULTILINE)
                        if ids:
                            current_max_id = max(int(id_str) for id_str in ids)
                
                # 调整ID以确保连续性
                for segment in processed_segments_with_gaps:
                    segment.id += current_max_id
                
                # 重新生成字幕内容以反映新的ID
                subtitle_content = generate_subtitle_content(processed_segments_with_gaps)
                
                # 追加到输出文件
                with open(output_path, "a", encoding='utf-8') as f:
                    f.write(subtitle_content)
                
                print(f"转录完成，结果已追加到: {output_path}")
                
                # 打印最新转录内容
                combined_text = " ".join([seg.text for seg in segment_list])
                print(f"转录内容: {combined_text[:200]}...")
                
                # 更新总处理时长（使用实际处理的最后一个段的结束时间，而不是原始音频时长）
                if processed_segments_with_gaps:
                    last_end_time = max(seg.end for seg in processed_segments_with_gaps)
                    with threading.Lock():  # 使用锁确保线程安全
                        self.total_processed_duration = max(self.total_processed_duration, last_end_time)
                else:
                    # 如果没有处理任何段，至少要更新时长
                    with threading.Lock():  # 使用锁确保线程安全
                        self.total_processed_duration += audio_duration
                
        except Exception as e:
            print(f"转录音频时发生错误: {e}")
        finally:
            # 删除临时音频文件
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def stop_listening(self):
        """停止监听"""
        self.is_recording = False
        print("已停止监听系统音频输出")


def main():
    """主函数"""
    translator = RealtimeTranslator()
    
    try:
        print(f"\n开始实时翻译程序 (麦克风模式)...")
        print("按 Ctrl+C 停止程序")
        translator.start_listening()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        translator.stop_listening()
    except Exception as e:
        print(f"程序运行时发生错误: {e}")
        translator.stop_listening()


if __name__ == "__main__":
    main()