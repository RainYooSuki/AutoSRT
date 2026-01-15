# coding: utf-8

import os
import yaml
import module
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load configuration
with open('./config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Get supported audio extensions from config
SUPPORTED_AUDIO_EXTENSIONS = set(config['audio']['supported_extensions'])


device = module.device_detect()
whisper_model_path = module.SRTwhisper_model_path

# 线程锁，用于保护模型访问
model_lock = threading.Lock()


def process_audio(index, wav, total_count, whisper_model):
    return module.process_audio(index, wav, total_count, whisper_model, model_lock)


if __name__ == '__main__':

    # 模型加载
    if device == 'cuda':
        # Get GPU compute types from config
        gpu_compute_types = config['model']['compute_types_gpu']
        
        # Try each compute type in order
        model_loaded = False
        for compute_type in gpu_compute_types:
            try:
                # 尝试加载模型到GPU
                whisper_model = WhisperModel(
                    model_size_or_path=whisper_model_path,
                    device=device,
                    local_files_only=True,
                    compute_type=compute_type,
                    num_workers=config['model']['num_workers'])
                print(f'加载模型到GPU:{compute_type.upper()}')
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
        whisper_model = WhisperModel(
            model_size_or_path=whisper_model_path,
            device=device,
            local_files_only=True,
            compute_type=config['model']['compute_type_cpu'],
            num_workers=config['model']['num_workers'])
        print(f'加载模型到CPU:{config["model"]["compute_type_cpu"].upper()}')
    print("初始化模型完成")

    # 音频格式修正为wav
    audio_converted = False
    with ThreadPoolExecutor(max_workers=12) as executor:
        # 提交所有需要转换的音频文件到线程池
        future_to_audio = {}
        for srtinput in module.srtallinput:
            _, ext = os.path.splitext(os.path.basename(srtinput))
            # 只处理支持的音频格式文件
            if ext.lower() in SUPPORTED_AUDIO_EXTENSIONS or ext.lower() == '.wav':
                if ext.lower() != '.wav':
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
        module.srtwavs = module.glob.glob(f'{module.INPUT_DIR}/*.wav')
    
    # 确保只处理实际的音频文件（wav文件默认认为是有效的音频文件）
    valid_wav_files = []
    for wav_file in module.srtwavs:
        if os.path.isfile(wav_file):  # 确保文件存在
            valid_wav_files.append(wav_file)
    module.srtwavs = valid_wav_files
    
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
