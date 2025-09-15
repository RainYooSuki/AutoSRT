# coding: utf-8

import os
import module
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


device = module.device_detect()
whisper_model_path = module.SRTwhisper_model_path

# 线程锁，用于保护模型访问
model_lock = threading.Lock()


def process_audio(index, wav, total_count, whisper_model):
    return module.process_audio(index, wav, total_count, whisper_model, model_lock)


if __name__ == '__main__':

    # 模型加载
    if device == 'cuda':
        # noinspection PyBroadException
        try:
            # 以fp16加载模型到GPU
            whisper_model = WhisperModel(
                model_size_or_path=whisper_model_path,
                device=device,
                local_files_only=True,
                compute_type='float16',
                num_workers=12)
            print('加载模型到GPU:FP16')
        except Exception as e:
            # 以fp32加载模型到GPU
            whisper_model = WhisperModel(
                model_size_or_path=whisper_model_path,
                device=device,
                local_files_only=True,
                compute_type='float32',
                num_workers=12)
            print('加载模型到GPU:FP32')
    else:
        # 以int8加载模型到CPU
        whisper_model = WhisperModel(
            model_size_or_path=whisper_model_path,
            device=device,
            local_files_only=True,
            compute_type='int8',
            num_workers=12)
        print('加载模型到CPU:IN8')
    print("初始化模型完成")

    # 音频格式修正为wav
    audio_converted = False
    with ThreadPoolExecutor(max_workers=12) as executor:
        # 提交所有需要转换的音频文件到线程池
        future_to_audio = {}
        for srtinput in module.srtallinput:
            if os.path.basename(srtinput).split('.')[1] != 'wav':
                future = executor.submit(module.audio2wav, srtinput)
                future_to_audio[future] = srtinput
        
        # 等待所有转换任务完成
        for future in as_completed(future_to_audio):
            srtinput = future_to_audio[future]
            try:
                future.result()
                audio_converted = True
            except Exception as e:
                print(f"Error processing {srtinput}: {e}")
    
    # 等待所有音频转换完成后再更新wav音频列表
    if audio_converted:
        module.srtwavs = module.glob.glob('./SrtFiles/input/*.wav')
    print(f"检测到{len(module.srtwavs)}项wav音频，开始尝试处理")

    # 批量处理音频 - 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=12) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(process_audio, index, wav, len(module.srtwavs), whisper_model): index
            for index, wav in enumerate(module.srtwavs)
        }

        # 处理完成的任务
        completed_count = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                completed_count += 1
                if result:
                    print(f"第{index + 1}项音频处理完成 ({completed_count}/{len(module.srtwavs)})")
                else:
                    print(f"第{index + 1}项音频处理失败 ({completed_count}/{len(module.srtwavs)})")
            except Exception as e:
                completed_count += 1
                print(f"处理第{index + 1}项音频时发生异常: {e} ({completed_count}/{len(module.srtwavs)})")
