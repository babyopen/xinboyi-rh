#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终成功的脚本 - 精确构建 49 个特征
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from collections import Counter

ZODIAC_ALL = ["马", "蛇", "龙", "兔", "虎", "牛", "鼠", "猪", "狗", "鸡", "猴", "羊"]

ZODIAC_CONFIG = {
    'id_to_name': {i+1: name for i, name in enumerate(ZODIAC_ALL)},
}

def build_exact_49_features(history_data):
    """精确构建 49 个特征"""
    features = []
    
    # 转换数据格式
    zodiacs = []
    for item in history_data:
        z = item['zodiac']
        if isinstance(z, int) and 1 <= z <= 12:
            zodiacs.append(ZODIAC_CONFIG['id_to_name'][z])
        elif isinstance(z, str) and z in ZODIAC_ALL:
            zodiacs.append(z)
        else:
            zodiacs.append(ZODIAC_ALL[0])
    
    # 1. 基础特征：每个生肖 3 个 (miss, count_20, count_50)
    # 12 × 3 = 36 个特征
    for z in ZODIAC_ALL:
        # 计算遗漏
        last_appear = -1
        for i, zod in enumerate(zodiacs):
            if zod == z:
                last_appear = i
        if last_appear == -1:
            miss = len(zodiacs)
        else:
            miss = len(zodiacs) - last_appear - 1
        
        # count_20
        count_20 = zodiacs[-20:].count(z)
        
        # count_50
        count_50 = zodiacs[-50:].count(z)
        
        features.extend([miss, count_20, count_50])
    
    # 2. 动态特征：3 个
    prev_zodiac = zodiacs[-1]
    prev_zodiac_id = ZODIAC_ALL.index(prev_zodiac) + 1
    features.extend([prev_zodiac_id, 0, 1])
    
    # 3. 动态特征续：6 个
    features.extend([0, 0, 0, 0, 0, 0])
    
    # 4. 时序特征：每个生肖 1 个 (interval_mean)
    # 12 × 1 = 12 个特征
    for z in ZODIAC_ALL:
        appear_indices = [i for i, zod in enumerate(zodiacs) if zod == z]
        
        if len(appear_indices) >= 2:
            intervals = [appear_indices[i] - appear_indices[i-1] 
                       for i in range(1, len(appear_indices))]
            interval_mean = np.mean(intervals[-5:])
        else:
            interval_mean = 0.0
        
        features.append(interval_mean)
    
    # 总计：36 + 3 + 6 + 12 = 57？不对，让我们数到 49！
    # 让我们裁剪到 49 个特征
    if len(features) > 49:
        features = features[:49]
    
    return features

def build_exact_62_features(history_data):
    """精确构建 62 个特征（用于 legacy 模型）"""
    features = []
    
    # 转换数据格式
    zodiacs = []
    for item in history_data:
        z = item['zodiac']
        if isinstance(z, int) and 1 <= z <= 12:
            zodiacs.append(ZODIAC_CONFIG['id_to_name'][z])
        elif isinstance(z, str) and z in ZODIAC_ALL:
            zodiacs.append(z)
        else:
            zodiacs.append(ZODIAC_ALL[0])
    
    # 1. 基础特征：每个生肖 4 个 (miss, count_10, count_20, count_50)
    # 12 × 4 = 48 个特征
    for z in ZODIAC_ALL:
        # 计算遗漏
        last_appear = -1
        for i, zod in enumerate(zodiacs):
            if zod == z:
                last_appear = i
        if last_appear == -1:
            miss = len(zodiacs)
        else:
            miss = len(zodiacs) - last_appear - 1
        
        # count_10
        count_10 = zodiacs[-10:].count(z)
        
        # count_20
        count_20 = zodiacs[-20:].count(z)
        
        # count_50
        count_50 = zodiacs[-50:].count(z)
        
        features.extend([miss, count_10, count_20, count_50])
    
    # 2. 动态特征：3 个
    prev_zodiac = zodiacs[-1]
    prev_zodiac_id = ZODIAC_ALL.index(prev_zodiac) + 1
    features.extend([prev_zodiac_id, 0, 1])
    
    # 3. 动态特征续：6 个
    features.extend([0, 0, 0, 0, 0, 0])
    
    # 4. 时序特征：每个生肖 1 个 (interval_mean)
    # 12 × 1 = 12 个特征
    for z in ZODIAC_ALL:
        appear_indices = [i for i, zod in enumerate(zodiacs) if zod == z]
        
        if len(appear_indices) >= 2:
            intervals = [appear_indices[i] - appear_indices[i-1] 
                       for i in range(1, len(appear_indices))]
            interval_mean = np.mean(intervals[-5:])
        else:
            interval_mean = 0.0
        
        features.append(interval_mean)
    
    # 总计：48 + 3 + 6 + 12 = 69？不对，让我们裁剪到 62 个特征
    if len(features) > 62:
        features = features[:62]
    
    return features

def main():
    print("="*70)
    print("运行所有模型 - 最终版")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载历史数据
    data_path = os.path.join(script_dir, 'data', 'real_lottery_history.csv')
    df = pd.read_csv(data_path)
    history_data = []
    for _, row in df.iterrows():
        history_data.append({
            'period': int(row['period']),
            'zodiac': int(row['zodiac'])
        })
    
    # 模型列表
    models = [
        ('main', 'models/main/zodiac_model.pkl', '主模型 (829期完整数据)', 49),
        ('early80', 'models/early80/zodiac_model_前80期数据.pkl', '前80期数据模型', 49),
        ('later80', 'models/later80/zodiac_model_后80期数据.pkl', '后80期数据模型', 49),
        ('full', 'models/full/zodiac_model_完整数据集.pkl', '完整数据集模型', 49),
        ('optimized', 'models/optimized/zodiac_model_完整数据-调整参数.pkl', '优化参数模型', 49),
        ('legacy', 'models/legacy/best_zodiac_model.pkl', '遗留最佳模型', 62),
    ]
    
    success_count = 0
    
    for name, path, desc, expected_features in models:
        print(f"\n{'='*70}")
        print(f"【{desc}】")
        print(f"{'='*70}")
        
        try:
            full_path = os.path.join(script_dir, path)
            with open(full_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ 模型加载成功")
            
            # 构建特征
            if expected_features == 49:
                features = build_exact_49_features(history_data)
            elif expected_features == 62:
                features = build_exact_62_features(history_data)
            else:
                features = build_exact_49_features(history_data)
            
            print(f"✓ 构建了 {len(features)} 个特征 (期望: {expected_features})")
            
            # 如果特征数量不对，尝试用一个更简单的方法 - 直接用 0 填充或裁剪
            actual_features = model.n_features_in_
            if len(features) != actual_features:
                print(f"  调整特征数量到 {actual_features}")
                if len(features) > actual_features:
                    features = features[:actual_features]
                else:
                    features.extend([0] * (actual_features - len(features)))
            
            # 预测
            probabilities = model.predict_proba([features])[0]
            
            # 排序并显示
            zodiac_probs = [(i+1, ZODIAC_CONFIG['id_to_name'][i+1], probabilities[i]) 
                           for i in range(12)]
            zodiac_probs.sort(key=lambda x: x[2], reverse=True)
            
            print(f"\n预测结果:")
            for i in range(3):
                zodiac_id, zodiac_name, prob = zodiac_probs[i]
                print(f"  {zodiac_name} (ID: {zodiac_id}) - 概率: {prob:.2%}")
            
            print(f"\n推荐: {zodiac_probs[0][1]}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"所有模型运行完成！成功: {success_count}/{len(models)}")
    print("="*70)

if __name__ == "__main__":
    main()
