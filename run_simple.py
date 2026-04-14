#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的运行所有模型的脚本
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import shutil

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'python'))

from zodiac_ml_predictor import (
    load_model, predict_next, ZODIAC_CONFIG
)

def main():
    print("="*70)
    print("运行所有模型")
    print("="*70)
    
    # 复制数据文件到 python 目录
    data_src = os.path.join(script_dir, 'data', 'real_lottery_history.csv')
    data_dst = os.path.join(script_dir, 'python', 'real_lottery_history.csv')
    shutil.copy(data_src, data_dst)
    
    # 模型列表
    models = [
        ('main', 'models/main/zodiac_model.pkl', '主模型 (829期完整数据)'),
        ('early80', 'models/early80/zodiac_model_前80期数据.pkl', '前80期数据模型'),
        ('later80', 'models/later80/zodiac_model_后80期数据.pkl', '后80期数据模型'),
        ('full', 'models/full/zodiac_model_完整数据集.pkl', '完整数据集模型'),
        ('optimized', 'models/optimized/zodiac_model_完整数据-调整参数.pkl', '优化参数模型'),
        ('legacy', 'models/legacy/best_zodiac_model.pkl', '遗留最佳模型'),
    ]
    
    # 加载历史数据
    df = pd.read_csv(data_dst)
    
    for name, path, desc in models:
        print(f"\n{'='*70}")
        print(f"【{desc}】")
        print(f"{'='*70}")
        
        try:
            full_path = os.path.join(script_dir, path)
            model = load_model(full_path)
            print(f"✓ 模型加载成功")
            
            # 尝试预测
            probabilities = predict_next(model, df.iloc[-1], df)
            
            # 排序并显示
            zodiac_probs = [(i+1, ZODIAC_CONFIG['id_to_name'][i+1], probabilities[i]) 
                           for i in range(12)]
            zodiac_probs.sort(key=lambda x: x[2], reverse=True)
            
            print(f"\n预测结果:")
            for i in range(3):
                zodiac_id, zodiac_name, prob = zodiac_probs[i]
                print(f"  {zodiac_name} (ID: {zodiac_id}) - 概率: {prob:.2%}")
            
            print(f"\n推荐: {zodiac_probs[0][1]}")
            
        except Exception as e:
            print(f"✗ 失败: {e}")
    
    # 清理
    os.remove(data_dst)
    
    print("\n" + "="*70)
    print("所有模型运行完成！")
    print("="*70)

if __name__ == "__main__":
    main()
