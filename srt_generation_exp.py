# coding: utf-8

import os
import yaml
import module
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import time
import torch


class SRTGenerator:
    def __init__(self, config_path='./config/config.yaml'):
        """初始化SRT生成器"""
        self.config = self.load_config(config_path)
        self.model_instances = []  # 存储模型实例
        self.instance_locks = []   # 存储模型实例对应的可重入锁
        self.max_instances = self.config['model'].get('max_instances', 1)  # 从配置中读取最大实例数
        self.audio_queue = Queue()  # 音频处理队列
        self.results = {}  # 存储处理结果
        self.gpu_count = torch.cuda.device_count()  # 检测GPU数量
        print(f"检测到 {self.gpu_count} 个GPU设备")
        
    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def initialize_models(self):
        """初始化多个模型实例，支持多GPU分配"""
        print(f"初始化 {self.max_instances} 个模型实例...")
        
        # 根据GPU数量决定分配策略
        if self.gpu_count > 1:
            print(f"检测到多GPU环境 ({self.gpu_count} 张卡)，将模型实例分布到不同GPU上")
            self._initialize_multigpu_models()
        else:
            print(f"单GPU或无GPU环境，使用传统方式初始化模型实例")
            self._initialize_single_gpu_models()
    
    def _initialize_multigpu_models(self):
        """多GPU环境下的模型初始化"""
        # 确定最佳compute_type - 从第一个实例在GPU 0上确定
        first_model = None
        best_compute_type = None
        
        gpu_compute_types = self.config['model'].get('compute_types_gpu', ['float16'])
        print(f"在GPU 0 上测试compute_types: {gpu_compute_types}")
        
        for ct in gpu_compute_types:
            try:
                print(f"  尝试 compute_type: {ct}")
                first_model = WhisperModel(
                    self.config['paths']['model_path'],
                    device="cuda:0",  # 使用第一张GPU
                    local_files_only=True,
                    compute_type=ct,
                    num_workers=self.config['model'].get('num_workers', 4)
                )
                best_compute_type = ct
                print(f"  成功使用 compute_type: {ct} 在GPU 0")
                break
            except Exception as e:
                print(f"  compute_type {ct} 在GPU 0 初始化失败: {e}")
                continue
        
        # 如果GPU 0 初始化失败，则使用CPU
        if first_model is None:
            cpu_compute_type = self.config['model'].get('compute_type_cpu', 'int8')
            print(f"GPU 0 初始化失败，使用CPU，compute_type: {cpu_compute_type}")
            try:
                first_model = WhisperModel(
                    self.config['paths']['model_path'],
                    device="cpu",
                    local_files_only=True,
                    compute_type=cpu_compute_type,
                    num_workers=self.config['model'].get('num_workers', 4)
                )
                best_compute_type = cpu_compute_type
                print(f"  CPU模式初始化成功，compute_type: {cpu_compute_type}")
            except Exception as e:
                print(f"  CPU模式初始化失败: {e}")
        
        # 将第一个模型实例添加到列表
        self.model_instances.append(first_model)
        self.instance_locks.append(threading.RLock())
        
        if first_model is not None:
            print(f"模型实例 1/{self.max_instances} 初始化完成，使用的compute_type: {best_compute_type}")
            
            # 使用确定的最佳compute_type初始化其余实例，并分配到不同GPU
            for i in range(1, self.max_instances):
                gpu_id = i % self.gpu_count  # 循环分配到不同GPU
                try:
                    if self.gpu_count > 0:
                        device = f"cuda:{gpu_id}"
                    else:
                        device = "cpu"
                    
                    additional_model = WhisperModel(
                        self.config['paths']['model_path'],
                        device=device,
                        local_files_only=True,
                        compute_type=best_compute_type,
                        num_workers=self.config['model'].get('num_workers', 4)
                    )
                    self.model_instances.append(additional_model)
                    self.instance_locks.append(threading.RLock())
                    print(f"模型实例 {i+1}/{self.max_instances} 初始化完成，部署在 {device}，使用的compute_type: {best_compute_type}")
                except Exception as e:
                    print(f"模型实例 {i+1} 在 {device} 初始化失败: {e}")
                    # 如果额外实例初始化失败，添加None占位符
                    self.model_instances.append(None)
                    self.instance_locks.append(threading.RLock())
        else:
            # 第一个模型就失败，那么全部失败
            print(f"模型实例 1/{self.max_instances} 初始化失败")
            for i in range(1, self.max_instances):
                self.model_instances.append(None)
                self.instance_locks.append(threading.RLock())
                print(f"模型实例 {i+1}/{self.max_instances} 初始化失败")
    
    def _initialize_single_gpu_models(self):
        """单GPU或CPU环境下的模型初始化"""
        # 首先确定最佳compute_type - 从第一个实例确定
        first_model = None
        best_compute_type = None
        
        # 对第一个模型实例尝试所有可能的compute_type
        if 'cuda' in self.config['model']['device']:
            gpu_compute_types = self.config['model'].get('compute_types_gpu', ['float16'])
            print(f"尝试使用GPU初始化第一个模型实例，测试compute_types: {gpu_compute_types}")
            
            for ct in gpu_compute_types:
                try:
                    print(f"  尝试 compute_type: {ct}")
                    first_model = WhisperModel(
                        self.config['paths']['model_path'],
                        device=self.config['model']['device'],
                        local_files_only=True,  # 添加这个参数以匹配原文件
                        compute_type=ct,
                        num_workers=self.config['model'].get('num_workers', 4)
                    )
                    best_compute_type = ct
                    print(f"  成功使用 compute_type: {ct}")
                    break
                except Exception as e:
                    print(f"  compute_type {ct} 初始化失败: {e}")
                    continue
        
        # 如果GPU初始化失败或未指定GPU，则尝试CPU
        if first_model is None:
            cpu_compute_type = self.config['model'].get('compute_type_cpu', 'int8')
            print(f"GPU初始化失败，尝试使用CPU，compute_type: {cpu_compute_type}")
            try:
                first_model = WhisperModel(
                    self.config['paths']['model_path'],
                    device="cpu",
                    local_files_only=True,  # 添加这个参数以匹配原文件
                    compute_type=cpu_compute_type,
                    num_workers=self.config['model'].get('num_workers', 4)
                )
                best_compute_type = cpu_compute_type
                print(f"  CPU模式初始化成功，compute_type: {cpu_compute_type}")
            except Exception as e:
                print(f"  CPU模式初始化失败: {e}")
        
        # 将第一个模型实例添加到列表
        self.model_instances.append(first_model)
        self.instance_locks.append(threading.RLock())
        
        if first_model is not None:
            print(f"模型实例 1/{self.max_instances} 初始化完成，使用的compute_type: {best_compute_type}")
            
            # 使用确定的最佳compute_type初始化其余实例
            for i in range(1, self.max_instances):
                try:
                    additional_model = WhisperModel(
                        self.config['paths']['model_path'],
                        device=self.config['model']['device'] if 'cuda' in self.config['model']['device'] else "cpu",
                        local_files_only=True,
                        compute_type=best_compute_type,
                        num_workers=self.config['model'].get('num_workers', 4)
                    )
                    self.model_instances.append(additional_model)
                    self.instance_locks.append(threading.RLock())
                    print(f"模型实例 {i+1}/{self.max_instances} 初始化完成，使用的compute_type: {best_compute_type}")
                except Exception as e:
                    print(f"模型实例 {i+1} 初始化失败: {e}")
                    # 如果额外实例初始化失败，添加None占位符
                    self.model_instances.append(None)
                    self.instance_locks.append(threading.RLock())
        else:
            # 第一个模型就失败，那么全部失败
            print(f"模型实例 1/{self.max_instances} 初始化失败")
            for i in range(1, self.max_instances):
                self.model_instances.append(None)
                self.instance_locks.append(threading.RLock())
                print(f"模型实例 {i+1}/{self.max_instances} 初始化失败")
    
    def process_single_audio(self, audio_index, audio_path, total_count):
        """使用指定模型实例处理单个音频文件"""
        # 循环分配模型实例，跳过不可用的实例
        instance_idx = audio_index % len(self.model_instances)
        model_instance = self.model_instances[instance_idx]
        
        if model_instance is None:
            print(f"警告: 模型实例 {instance_idx} 不可用，跳过音频 {audio_path}")
            return False
            
        model_lock = self.instance_locks[instance_idx]
        
        try:
            print(f"使用模型实例 {instance_idx+1} 处理第 {audio_index+1} 个音频: {os.path.basename(audio_path)}")
            
            # 使用对应的模型实例锁调用 module 中的处理函数
            # 使用 RLock 可以避免同一线程重复获取锁造成的死锁
            result = module.process_audio(audio_index, audio_path, total_count, model_instance, model_lock)
            
            return result
        except Exception as e:
            print(f"处理音频 {audio_path} 时出错: {e}")
            return False
    
    def process_audio_list(self, audio_list):
        """处理音频列表"""
        if not audio_list:
            print("没有音频文件需要处理")
            return
        
        # 动态调整模型实例数量为音频数量和最大实例数的较小值
        actual_instance_count = min(self.max_instances, len(audio_list))
        print(f"开始处理 {len(audio_list)} 个音频文件，计划创建 {actual_instance_count} 个模型实例")
        
        # 更新要创建的实例数量
        original_max_instances = self.max_instances
        self.max_instances = actual_instance_count
        
        # 初始化模型实例
        self.initialize_models()
        
        # 恢复原始的最大实例数设置（以防后续需要）
        self.max_instances = original_max_instances
        
        # 限制并发数不超过实际创建的模型实例数和音频总数的最小值
        max_workers = min(actual_instance_count, len(audio_list))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有音频处理任务
            future_to_index = {
                executor.submit(self.process_single_audio, idx, audio_path, len(audio_list)): idx 
                for idx, audio_path in enumerate(audio_list)
            }
            
            # 等待所有任务完成
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result:
                        print(f"第 {index+1} 个音频处理成功")
                    else:
                        print(f"第 {index+1} 个音频处理失败")
                except Exception as e:
                    print(f"第 {index+1} 个音频处理过程中出现异常: {e}")
        
        print("所有音频处理完成")
    
    def cleanup(self):
        """清理资源"""
        print("清理模型实例...")
        for i, model in enumerate(self.model_instances):
            if model is not None:
                # 如果模型有清理方法，调用它
                del model
        self.model_instances.clear()
        print("资源清理完成")


def main():
    """主函数"""
    generator = SRTGenerator()
    
    # 获取配置
    config = generator.load_config('./config/config.yaml')
    
    # 获取输入目录中的所有音频文件
    input_dir = config['paths']['input_dir']
    supported_extensions = set(config['audio']['supported_extensions'])
    
    audio_files = []
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in supported_extensions):
            audio_files.append(os.path.join(input_dir, file))
    
    print(f"检测到 {len(audio_files)} 个音频文件: {[os.path.basename(f) for f in audio_files]}")
    
    if audio_files:
        generator.process_audio_list(audio_files)
    
    # 清理资源
    generator.cleanup()


if __name__ == "__main__":
    main()