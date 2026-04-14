#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终的运行所有模型的脚本
直接复制 ml_api_server.py 中的特征构建方法
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from collections import Counter

# ============================================
# 直接从 ml_api_server.py 复制的完整代码
# ============================================

ZODIAC_ALL = ["马", "蛇", "龙", "兔", "虎", "牛", "鼠", "猪", "狗", "鸡", "猴", "羊"]

ZODIAC_CONFIG = {
    'id_to_name': {i+1: name for i, name in enumerate(ZODIAC_ALL)},
    'zodiac_to_element': {
        '马': '火', '蛇': '火', '龙': '土', '兔': '木',
        '虎': '木', '牛': '土', '鼠': '水', '猪': '水',
        '狗': '土', '鸡': '金', '猴': '金', '羊': '土'
    },
    'zodiac_to_color': {
        '马': '红', '蛇': '红', '龙': '红', '兔': '绿',
        '虎': '蓝', '牛': '绿', '鼠': '红', '猪': '蓝',
        '狗': '绿', '鸡': '红', '猴': '蓝', '羊': '绿'
    }
}

ELEMENT_GENERATE = {
    '金': '水', '水': '木', '木': '火', '火': '土', '土': '金'
}

ELEMENT_OVERCOME = {
    '金': '木', '木': '土', '土': '水', '水': '火', '火': '金'
}

def get_element_relation(element1, element2):
    """
    获取两个五行元素之间的关系
    返回: 0=相克, 1=相同, 2=相生
    """
    if element1 == element2:
        return 1  # 相同
    elif ELEMENT_GENERATE.get(element1) == element2:
        return 2  # 相生
    else:
        return 0  # 相克

def build_features_from_ml_server(history_data):
    """
    构建特征向量
    与项目内的特征工程保持一致（直接从ml_api_server.py复制）
    """
    n_zodiacs = 12
    features = []
    
    # 转换数据格式 - 统一转为生肖名称
    zodiacs = []
    for item in history_data:
        z = item['zodiac']
        if isinstance(z, int) and 1 <= z <= 12:
            zodiacs.append(ZODIAC_CONFIG['id_to_name'][z])
        elif isinstance(z, str) and z in ZODIAC_ALL:
            zodiacs.append(z)
        else:
            zodiacs.append(ZODIAC_ALL[0])  # 默认鼠
    
    # 基础统计特征
    miss_counts = {z: 0 for z in ZODIAC_ALL}
    max_miss = {z: 0 for z in ZODIAC_ALL}
    
    # 计算遗漏
    for z in ZODIAC_ALL:
        last_appear = -1
        for i, zod in enumerate(zodiacs):
            if zod == z:
                last_appear = i
        if last_appear == -1:
            miss_counts[z] = len(zodiacs)
        else:
            miss_counts[z] = len(zodiacs) - last_appear - 1
    
    # 计算最大遗漏
    for z in ZODIAC_ALL:
        current_miss = 0
        for zod in zodiacs:
            if zod == z:
                max_miss[z] = max(max_miss[z], current_miss)
                current_miss = 0
            else:
                current_miss += 1
    
    # 近N期统计
    recent_10 = zodiacs[-10:]
    recent_20 = zodiacs[-20:]
    recent_50 = zodiacs[-50:]
    
    # 计算排名
    counts_20 = Counter(recent_20)
    ranks = {}
    for z in ZODIAC_ALL:
        count = counts_20.get(z, 0)
        rank = 1
        for other_z in ZODIAC_ALL:
            if counts_20.get(other_z, 0) > count:
                rank += 1
        ranks[z] = rank
    
    # 计算连开次数
    consecutive = {z: 0 for z in ZODIAC_ALL}
    break_state = {z: 0 for z in ZODIAC_ALL}
    
    for z in ZODIAC_ALL:
        cons = 0
        for i in range(len(zodiacs) - 1, -1, -1):
            if zodiacs[i] == z:
                cons += 1
            else:
                break
        consecutive[z] = cons
        
        if len(zodiacs) >= 2:
            last = zodiacs[-1]
            second_last = zodiacs[-2]
            break_state[z] = 1 if (last == z and second_last != z) else 0
    
    # 添加基础统计特征 - 按项目内生肖顺序
    for z in ZODIAC_ALL:
        miss = miss_counts[z]
        max_m = max_miss[z] if max_miss[z] > 0 else 1
        
        features.extend([
            miss,
            miss / max_m,
            recent_10.count(z),
            recent_20.count(z),
            recent_50.count(z),
            recent_10.count(z) / 10,
            recent_20.count(z) / 20,
            ranks[z],
            consecutive[z],
            break_state[z],
        ])
    
    # 动态特征
    prev_zodiac = zodiacs[-1]
    prev_zodiac_id = ZODIAC_ALL.index(prev_zodiac) + 1
    features.append(prev_zodiac_id)
    features.append(0)
    features.append(1)
    features.extend([0, 0, 0, 0, 0, 0])
    
    # 时序特征
    for z in ZODIAC_ALL:
        appear_indices = [i for i, zod in enumerate(zodiacs) if zod == z]
        
        if len(appear_indices) >= 2:
            intervals = [appear_indices[i] - appear_indices[i-1] 
                       for i in range(1, len(appear_indices))]
            interval_mean = np.mean(intervals[-5:])
            interval_std = np.std(intervals[-5:]) if len(intervals) >= 5 else 0
        else:
            interval_mean = 0
            interval_std = 0
        
        features.extend([interval_mean, interval_std, 0])
    
    return features

def main():
    print("="*70)
    print("运行所有模型")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载历史数据
    data_path = os.path.join(script_dir, 'data', 'real_lottery_history.csv')
    df = pd.read_csv(data_path)
    
    # 转换为历史数据格式
    history_data = []
    for _, row in df.iterrows():
        history_data.append({
            'period': int(row['period']),
            'zodiac': int(row['zodiac'])
        })
    
    # 测试特征构建
    print("\n测试特征构建:")
    features = build_features_from_ml_server(history_data)
    print(f"build_features_from_ml_server: {len(features)} 个特征")
    
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
            
            # 构建特征
            features = build_features_from_ml_server(history_data)
            
            # 获取模型期望的特征数量
            expected_features = model.n_features_in_
            print(f"✓ 构建了 {len(features)} 个特征 (期望: {expected_features})")
            
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
    
    print("\n" + "="*70)
    print(f"所有模型运行完成！成功: {success_count}/{len(models)}")
    print("="*70)

if __name__ == "__main__":
    main()
