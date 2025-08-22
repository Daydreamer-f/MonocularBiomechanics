#!/usr/bin/env python3
"""
脚本用于检查 _fitted_model.npz 文件的详细内容
"""

import numpy as np
import os
import glob
from pathlib import Path

def inspect_fitted_model(file_path):
    """检查单个fitted_model.npz文件的内容"""
    print(f"\n{'='*60}")
    print(f"检查文件: {file_path}")
    print(f"{'='*60}")
    
    try:
        # 加载npz文件
        with np.load(file_path, allow_pickle=True) as data:
            print(f"文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            print(f"包含的数组数量: {len(data.files)}")
            print("\n数组详细信息:")
            print("-" * 50)
            
            for key in data.files:
                array = data[key]
                print(f"键名: {key}")
                print(f"  - 数据类型: {array.dtype}")
                print(f"  - 形状: {array.shape}")
                print(f"  - 维度: {array.ndim}")
                print(f"  - 总元素数: {array.size}")
                print(f"  - 内存大小: {array.nbytes / 1024:.2f} KB")
                
                # 显示数值范围
                if array.size > 0 and np.issubdtype(array.dtype, np.number):
                    print(f"  - 数值范围: [{np.min(array):.4f}, {np.max(array):.4f}]")
                    print(f"  - 均值: {np.mean(array):.4f}")
                    print(f"  - 标准差: {np.std(array):.4f}")
                
                # 显示前几个值的示例
                if array.size > 0:
                    if array.ndim == 1:
                        print(f"  - 前5个值: {array[:5]}")
                    elif array.ndim == 2:
                        print(f"  - 前3行前3列:\n{array[:3, :3]}")
                    elif array.ndim == 3:
                        print(f"  - 第1帧前3行前3列:\n{array[0, :3, :3]}")
                
                print()
                
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}")
        print(f"错误信息: {e}")

def find_fitted_models():
    """查找当前目录下的所有fitted_model.npz文件"""
    pattern = "*_fitted_model.npz"
    files = glob.glob(pattern)
    return files

def main():
    print("🔍 查找 _fitted_model.npz 文件...")
    
    # 查找所有fitted model文件
    fitted_files = find_fitted_models()
    
    if not fitted_files:
        print("❌ 当前目录下没有找到 _fitted_model.npz 文件")
        print("\n💡 提示：")
        print("1. 请确保您在正确的目录下运行此脚本")
        print("2. 请先运行 python main.py 并处理一个视频以生成fitted model文件")
        return
    
    print(f"✅ 找到 {len(fitted_files)} 个 fitted model 文件:")
    for i, file in enumerate(fitted_files, 1):
        print(f"  {i}. {file}")
    
    # 检查每个文件
    for file_path in fitted_files:
        inspect_fitted_model(file_path)
    
    # 总结信息
    print(f"\n{'='*60}")
    print("📊 总结")
    print(f"{'='*60}")
    print(f"总共检查了 {len(fitted_files)} 个文件")
    
    if fitted_files:
        # 分析第一个文件的关键信息
        with np.load(fitted_files[0], allow_pickle=True) as data:
            if 'qpos' in data:
                qpos = data['qpos']
                print(f"关节角度序列长度: {qpos.shape[0]} 帧")
                print(f"关节数量: {qpos.shape[1]} 个")
            
            if 'timestamps' in data:
                timestamps = data['timestamps']
                duration = timestamps[-1] - timestamps[0]
                print(f"视频时长: {duration:.2f} 秒")
                print(f"帧率: {len(timestamps)/duration:.1f} FPS")

if __name__ == "__main__":
    main() 