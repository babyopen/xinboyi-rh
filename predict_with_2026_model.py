#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用2026年训练的模型进行预测（能够正确显示生肖名称）
"""

import pandas as pd
import numpy as np
import pickle
import os

# 生肖映射配置
ZODIAC_CONFIG = {
    'id_to_name': {
        1: '马', 2: '蛇', 3: '龙', 4: '兔', 5: '虎', 6: '牛',
        7: '鼠', 8: '猪', 9: '狗', 10: '鸡', 11: '猴', 12: '羊'
    }
}

def load_2026_data(file_path):
    """只加载2026年的数据"""
    try:
        df = pd.read_csv(file_path)
        df['period'] = df['period'].astype(int)
        df['zodiac'] = df['zodiac'].astype(int)
        df_2026 = df[df['period'] >= 2026001].copy()
        df_2026 = df_2026.sort_values('period').reset_index(drop=True)
        return df_2026
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def build_features_for_prediction(df):
    """构建预测用的特征"""
    n_zodiacs = 12
    idx = len(df)
    history = df
    features = []
    
    # 每个生肖3个特征（共12*3=36）
    for z in range(1, n_zodiacs + 1):
        miss = 0
        last_appear = -1
        for i, row in history.iterrows():
            if row['zodiac'] == z:
                last_appear = i
        if last_appear == -1:
            miss = len(history)
        else:
            miss = idx - last_appear - 1
        
        count_20 = (history.tail(20)['zodiac'] == z).sum()
        count_50 = (history.tail(50)['zodiac'] == z).sum() if len(history) >= 50 else (history['zodiac'] == z).sum()
        
        features.extend([miss, count_20, count_50])
    
    # 动态特征（3个）
    prev_zodiac = history.iloc[-1]['zodiac']
    features.extend([prev_zodiac, 0, 1])
    
    # 其他动态特征（6个）
    features.extend([0, 0, 0, 0, 0, 0])
    
    # 时序特征（每个生肖1个，共12个）
    for z in range(1, n_zodiacs + 1):
        appear_indices = []
        for i, row in history.iterrows():
            if row['zodiac'] == z:
                appear_indices.append(i)
        
        if len(appear_indices) >= 2:
            intervals = [appear_indices[i] - appear_indices[i-1] 
                       for i in range(1, len(appear_indices))]
            interval_mean = np.mean(intervals[-5:])
        else:
            interval_mean = 0.0
        
        features.append(interval_mean)
    
    # 确保正好49个特征
    features = features[:49] if len(features) > 49 else features + [0]*(49-len(features))
    
    return np.array([features])

def predict_single_model(model_path, model_name, df):
    """预测单个模型"""
    print(f"\n{'='*60}")
    print(f"【{model_name}】")
    print(f"{'='*60}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ 模型加载成功")
        print(f"✓ 模型期望 {model.n_features_in_} 个特征")
        
        # 构建特征
        features = build_features_for_prediction(df)
        print(f"✓ 构建了 {features.shape[1]} 个特征")
        
        # 预测
        probabilities = model.predict_proba(features)[0]
        num_classes = len(probabilities)
        print(f"✓ 模型预测了 {num_classes} 个类别")
        
        # 显示结果
        print(f"\n预测结果 (所有12个生肖):")
        print("-" * 50)
        
        # 获取模型的类别标签
        model_classes = model.classes_
        
        # 创建预测列表
        pred_list = []
        for i, class_idx in enumerate(model_classes):
            if class_idx in ZODIAC_CONFIG['id_to_name']:
                zodiac_name = ZODIAC_CONFIG['id_to_name'][class_idx]
                pred_list.append({
                    'zodiac_id': class_idx,
                    'zodiac_name': zodiac_name,
                    'probability': float(probabilities[i])
                })
        
        # 按概率排序
        pred_list_sorted = sorted(pred_list, key=lambda x: x['probability'], reverse=True)
        
        for i, item in enumerate(pred_list_sorted):
            prob = item['probability']
            bar = '█' * int(prob * 50)
            marker = " ★" if i < 3 else ""
            print(f"{i+1:2d}. {item['zodiac_id']:2d} {item['zodiac_name']:2s} - {prob:.2%} {bar}{marker}")
        
        print(f"\n✓ 模型 {model_name} 预测成功！")
        print(f"\n推荐:")
        print(f"  1. {pred_list_sorted[0]['zodiac_name']} (概率: {pred_list_sorted[0]['probability']:.2%})")
        print(f"  2. {pred_list_sorted[1]['zodiac_name']} (概率: {pred_list_sorted[1]['probability']:.2%})")
        print(f"  3. {pred_list_sorted[2]['zodiac_name']} (概率: {pred_list_sorted[2]['probability']:.2%})")
        
        return pred_list_sorted
        
    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主程序"""
    print("="*70)
    print("使用2026年训练的模型进行预测")
    print("="*70)
    
    # 加载2026年数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'real_lottery_history.csv')
    df = load_2026_data(data_path)
    
    if df is None:
        print("无法加载数据，退出")
        return
    
    print(f"\n数据概况:")
    print(f"  2026年数据: {len(df)} 条 (2026001 - {df['period'].max()})")
    print(f"  最后一期: {df['period'].iloc[-1]} - {ZODIAC_CONFIG['id_to_name'][df['zodiac'].iloc[-1]]}")
    
    # 定义所有模型
    models_config = [
        {
            'name': '主模型',
            'path': os.path.join(script_dir, 'models', 'main', 'zodiac_model.pkl')
        },
        {
            'name': '前80期数据模型',
            'path': os.path.join(script_dir, 'models', 'early80', 'zodiac_model_前80期数据.pkl')
        },
        {
            'name': '后80期数据模型',
            'path': os.path.join(script_dir, 'models', 'later80', 'zodiac_model_后80期数据.pkl')
        },
        {
            'name': '完整数据集模型',
            'path': os.path.join(script_dir, 'models', 'full', 'zodiac_model_完整数据集.pkl')
        },
        {
            'name': '优化参数模型',
            'path': os.path.join(script_dir, 'models', 'optimized', 'zodiac_model_完整数据-调整参数.pkl')
        },
    ]
    
    # 预测所有模型
    all_results = []
    success_count = 0
    
    for config in models_config:
        result = predict_single_model(config['path'], config['name'], df)
        if result is not None:
            all_results.append((config['name'], result))
            success_count += 1
    
    # 汇总所有模型的预测结果
    print("\n" + "="*70)
    print("所有模型预测汇总")
    print("="*70)
    
    # 统计每个生肖被推荐的次数
    zodiac_counts = {}
    for model_name, result in all_results:
        for i, item in enumerate(result[:3]):
            zodiac_id = item['zodiac_id']
            if zodiac_id not in zodiac_counts:
                zodiac_counts[zodiac_id] = {'count': 0, 'total_prob': 0.0, 'zodiac_name': item['zodiac_name']}
            zodiac_counts[zodiac_id]['count'] += 1
            zodiac_counts[zodiac_id]['total_prob'] += item['probability']
    
    # 计算综合评分
    final_scores = []
    for zodiac_id, data in zodiac_counts.items():
        avg_prob = data['total_prob'] / len(all_results)
        final_scores.append({
            'zodiac_id': zodiac_id,
            'zodiac_name': data['zodiac_name'],
            'count': data['count'],
            'avg_prob': avg_prob,
            'score': data['count'] * avg_prob  # 综合评分 = 出现次数 * 平均概率
        })
    
    final_scores.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n综合推荐（基于所有模型）:")
    print("-" * 60)
    print(f"{'排名':<6} {'生肖':<4} {'出现次数':<10} {'平均概率':<12} {'综合评分':<10}")
    print("-" * 60)
    
    for i, item in enumerate(final_scores[:5]):
        marker = " ★" if i < 3 else ""
        print(f"{i+1:<6} {item['zodiac_name']:<4} {item['count']:<10} {item['avg_prob']:.2%}{'':<6} {item['score']:.4f}{marker}")
    
    print(f"\n{'='*70}")
    print(f"预测完成！成功: {success_count}/{len(models_config)} 个模型")
    print(f"{'='*70}")
    print(f"\n最终推荐:")
    print(f"  🏆 首选: {final_scores[0]['zodiac_name']}")
    print(f"  🥈 次选: {final_scores[1]['zodiac_name']}")
    print(f"  🥉 备选: {final_scores[2]['zodiac_name']}")

if __name__ == '__main__':
    main()
