#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量运行所有模型
"""

import os
import sys
import subprocess

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型列表
MODELS = [
    {'name': 'main', 'path': 'models/main/run.py', 'description': '主模型 (829期完整数据)'},
    {'name': 'early80', 'path': 'models/early80/run.py', 'description': '前80期数据模型'},
    {'name': 'later80', 'path': 'models/later80/run.py', 'description': '后80期数据模型'},
    {'name': 'full', 'path': 'models/full/run.py', 'description': '完整数据集模型'},
    {'name': 'optimized', 'path': 'models/optimized/run.py', 'description': '优化参数模型'},
    {'name': 'legacy', 'path': 'models/legacy/run.py', 'description': '遗留模型'},
    {'name': 'xboyi', 'path': 'models/xboyi/run.py', 'description': 'Xboyi模型'},
]

def run_model(model_info):
    """运行单个模型"""
    print(f"\n{'='*70}")
    print(f"正在运行: {model_info['description']}")
    print(f"{'='*70}")
    
    script_path = os.path.join(BASE_DIR, model_info['path'])
    model_dir = os.path.dirname(script_path)
    
    try:
        result = subprocess.run(
            [sys.executable, os.path.basename(script_path)],
            cwd=model_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print(f"警告/错误: {result.stderr}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"运行失败: {e}")
        return False

def main():
    """主函数"""
    print("="*70)
    print("生肖预测 - 批量运行所有模型")
    print("="*70)
    
    success_count = 0
    fail_count = 0
    results = []
    
    for model_info in MODELS:
        success = run_model(model_info)
        results.append({'model': model_info, 'success': success})
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # 总结
    print("\n" + "="*70)
    print("运行总结")
    print("="*70)
    for result in results:
        status = "✓ 成功" if result['success'] else "✗ 失败"
        print(f"{status}: {result['model']['description']}")
    
    print(f"\n总计: {len(MODELS)} 个模型")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")

if __name__ == "__main__":
    main()
