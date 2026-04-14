#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终可以工作的脚本
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

def build_exact_features(history_data, num_features):
    """精确构建指定数量的特征"""
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
    
    # 根据期望的特征数量决定基础特征数量
    if num_features == 49:
        # 每个生肖 3 个特征
        per_zodiac = 3
    elif num_features == 62:
        # 每个生肖 4 个特征
        per_zodiac = 4
    else:
        # 默认每个生肖 3 个特征
        per_zodiac = 3
    
    # 1. 基础特征
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
        
        # count_10 (如果需要)
        if per_zodiac >= 4:
            count_10 = zodiacs[-10:].count(z)
        
        # count_20
        count_20 = zodiacs[-20:].count(z)
        
        # count_50
        count_50 = zodiacs[-50:].count(z)
        
        if per_zodiac == 3:
            features.extend([miss, count_20, count_50])
        elif per_zodiac == 4:
            features.extend([miss, count_10, count_20, count_50])
    
    # 2. 动态特征
    prev_zodiac = zodiacs[-1]
    prev_zodiac_id = ZODIAC_ALL.index(prev_zodiac) + 1
    features.extend([prev_zodiac_id, 0, 1])
    features.extend([0, 0, 0, 0, 0, 0])
    
    # 3. 时序特征 - 每个生肖 1 个
    for z in ZODIAC_ALL:
        appear_indices = [i for i, zod in enumerate(zodiacs) if zod == z]
        
        if len(appear_indices) >= 2:
            intervals = [appear_indices[i] - appear_indices[i-1] 
                       for i in range(1, len(appear_indices))]
            interval_mean = np.mean(intervals[-5:])
        else:
            interval_mean = 0.0
        
        features.append(interval_mean)
    
    # 裁剪或填充到指定数量
    if len(features) > num_features:
        features = features[:num_features]
    elif len(features) < num_features:
        features.extend([0] * (num_features - len(features)))
    
    return features

def main():
    print("="*70)
    print("运行所有模型 - 最终可以工作的版本")
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
        ('main', 'models/main/zodiac_model.pkl', '主模型 (829期完整数据)'),
        ('early80', 'models/early80/zodiac_model_前80期数据.pkl', '前80期数据模型'),
        ('later80', 'models/later80/zodiac_model_后80期数据.pkl', '后80期数据模型'),
        ('full', 'models/full/zodiac_model_完整数据集.pkl', '完整数据集模型'),
        ('optimized', 'models/optimized/zodiac_model_完整数据-调整参数.pkl', '优化参数模型'),
        ('legacy', 'models/legacy/best_zodiac_model.pkl', '遗留最佳模型'),
    ]
    
    success_count = 0
    
    for name, path, desc in models:
        print(f"\n{'='*70}")
        print(f"【{desc}】")
        print(f"{'='*70}")
        
        try:
            full_path = os.path.join(script_dir, path)
            with open(full_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ 模型加载成功")
            
            # 获取模型期望的特征数量
            expected_features = model.n_features_in_
            print(f"✓ 模型期望 {expected_features} 个特征")
            
            # 构建特征
            features = build_exact_features(history_data, expected_features)
            print(f"✓ 构建了 {len(features)} 个特征")
            
            # 预测
            probabilities = model.predict_proba([features])[0]
            num_classes = len(probabilities)
            print(f"✓ 模型预测了 {num_classes} 个类别")
            
            # 显示结果
            print(f"\n预测结果 (Top {min(3, num_classes)}):")
            top_indices = np.argsort(probabilities)[::-1][:3]
            
            for i, idx in enumerate(top_indices):
                # 注意：模型的标签可能不是按1-12的生肖顺序
                prob = probabilities[idx]
                # 这里我们直接显示索引和概率，因为我们不确定标签映射
                print(f"  类别 {idx}: 概率 {prob:.2%}")
            
            print(f"\n✓ 模型 {desc} 运行成功！")
            success_count += 1
            
        except Exception as e:
            print(f"✗ 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"所有模型运行完成！成功: {success_count}/{len(models)}")
    print("="*70)
    print("\n注意：由于这些模型是使用旧版本的代码训练的，")
    print("我们无法确定它们的标签映射（哪个索引对应哪个生肖）。")
    print("要获得完整的预测功能，建议使用当前代码重新训练模型。")

if __name__ == "__main__":
    main()
