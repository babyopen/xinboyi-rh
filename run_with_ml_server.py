#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 ml_api_server.py 中的 MLPredictor 来运行所有模型
"""

import os
import sys
import pickle
import pandas as pd

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'python'))

# 复制数据到 python 目录
data_src = os.path.join(script_dir, 'data', 'real_lottery_history.csv')
data_dst = os.path.join(script_dir, 'python', 'real_lottery_history.csv')
import shutil
shutil.copy(data_src, data_dst)

# 现在我们直接修改 ml_api_server.py 中的 MLPredictor，让它可以加载不同的模型
# 让我们创建一个自定义版本

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
    if element1 == element2:
        return 1
    elif ELEMENT_GENERATE.get(element1) == element2:
        return 2
    else:
        return 0

# 从 ml_api_server.py 复制的 _build_features 方法
def build_features_ml(history_data):
    from collections import Counter
    features = []
    
    zodiacs = []
    for item in history_data:
        z = item['zodiac']
        if isinstance(z, int) and 1 <= z <= 12:
            zodiacs.append(ZODIAC_CONFIG['id_to_name'][z])
        elif isinstance(z, str) and z in ZODIAC_ALL:
            zodiacs.append(z)
        else:
            zodiacs.append(ZODIAC_ALL[0])
    
    miss_counts = {z: 0 for z in ZODIAC_ALL}
    max_miss = {z: 0 for z in ZODIAC_ALL}
    
    for z in ZODIAC_ALL:
        last_appear = -1
        for i, zod in enumerate(zodiacs):
            if zod == z:
                last_appear = i
        if last_appear == -1:
            miss_counts[z] = len(zodiacs)
        else:
            miss_counts[z] = len(zodiacs) - last_appear - 1
    
    for z in ZODIAC_ALL:
        current_miss = 0
        for zod in zodiacs:
            if zod == z:
                max_miss[z] = max(max_miss[z], current_miss)
                current_miss = 0
            else:
                current_miss += 1
    
    recent_10 = zodiacs[-10:]
    recent_20 = zodiacs[-20:]
    recent_50 = zodiacs[-50:]
    
    counts_20 = Counter(recent_20)
    ranks = {}
    for z in ZODIAC_ALL:
        count = counts_20.get(z, 0)
        rank = 1
        for other_z in ZODIAC_ALL:
            if counts_20.get(other_z, 0) > count:
                rank += 1
        ranks[z] = rank
    
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
    
    prev_zodiac = zodiacs[-1]
    prev_zodiac_id = ZODIAC_ALL.index(prev_zodiac) + 1
    features.append(prev_zodiac_id)
    features.append(0)
    features.append(1)
    features.extend([0, 0, 0, 0, 0, 0])
    
    for z in ZODIAC_ALL:
        appear_indices = [i for i, zod in enumerate(zodiacs) if zod == z]
        
        if len(appear_indices) >= 2:
            intervals = [appear_indices[i] - appear_indices[i-1] 
                       for i in range(1, len(appear_indices))]
            interval_mean = sum(intervals[-5:]) / len(intervals[-5:])
            if len(intervals) >= 5:
                interval_std = sum((x - interval_mean) ** 2 for x in intervals[-5:]) / 5
                interval_std = interval_std ** 0.5
            else:
                interval_std = 0
        else:
            interval_mean = 0
            interval_std = 0
        
        features.extend([interval_mean, interval_std, 0])
    
    return features

def main():
    print("="*70)
    print("使用 ML Server 方法运行所有模型")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载历史数据
    df = pd.read_csv(data_dst)
    history_data = []
    for _, row in df.iterrows():
        history_data.append({
            'period': int(row['period']),
            'zodiac': int(row['zodiac'])
        })
    
    # 测试不同的模型，看看它们期望多少个特征
    print("\n检查模型期望的特征数量:")
    test_models = [
        ('main', 'models/main/zodiac_model.pkl'),
        ('early80', 'models/early80/zodiac_model_前80期数据.pkl'),
        ('later80', 'models/later80/zodiac_model_后80期数据.pkl'),
        ('full', 'models/full/zodiac_model_完整数据集.pkl'),
        ('optimized', 'models/optimized/zodiac_model_完整数据-调整参数.pkl'),
        ('legacy', 'models/legacy/best_zodiac_model.pkl'),
    ]
    
    for name, path in test_models:
        full_path = os.path.join(script_dir, path)
        with open(full_path, 'rb') as f:
            model = pickle.load(f)
        print(f"  {name}: 期望 {model.n_features_in_} 个特征")
    
    # 现在让我们尝试简化，直接使用 python 目录下的 zodiac_ml_predictor.py 来训练新的模型！
    # 或者，让我们看看是否有其他方式。
    
    print("\n" + "="*70)
    print("由于现有模型是使用旧版本的特征工程训练的，")
    print("让我们尝试一个更简单的方法：直接运行 python 目录下的训练脚本！")
    print("="*70)
    
    # 让我们直接在 python 目录下运行 zodiac_ml_predictor.py
    print("\n尝试运行 zodiac_ml_predictor.py ...")
    
    os.chdir(os.path.join(script_dir, 'python'))
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, 'zodiac_ml_predictor.py'],
            capture_output=True,
            text=True,
            timeout=120
        )
        print("\n" + "="*70)
        print("zodiac_ml_predictor.py 输出:")
        print("="*70)
        print(result.stdout)
        if result.stderr:
            print("\n警告/错误:")
            print(result.stderr)
    except Exception as e:
        print(f"运行失败: {e}")
    
    # 清理
    os.remove(data_dst)
    
    print("\n" + "="*70)
    print("完成！")
    print("="*70)

if __name__ == "__main__":
    main()
