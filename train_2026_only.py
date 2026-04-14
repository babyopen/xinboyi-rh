#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只训练2026年数据的模型并进行预测
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report
import pickle
import warnings
import time
import os
warnings.filterwarnings('ignore')

# 生肖映射配置
ZODIAC_CONFIG = {
    'id_to_name': {
        1: '马', 2: '蛇', 3: '龙', 4: '兔', 5: '虎', 6: '牛',
        7: '鼠', 8: '猪', 9: '狗', 10: '鸡', 11: '猴', 12: '羊'
    },
    'zodiac_to_element': {
        1: '火', 2: '火', 3: '土', 4: '木', 5: '木', 6: '土',
        7: '水', 8: '水', 9: '土', 10: '金', 11: '金', 12: '土'
    },
    'zodiac_to_color': {
        1: '红', 2: '红', 3: '红', 4: '绿', 5: '蓝', 6: '绿',
        7: '红', 8: '蓝', 9: '绿', 10: '红', 11: '蓝', 12: '绿'
    }
}

def get_element_relation(element1, element2):
    element_generate = {'金': '水', '水': '木', '木': '火', '火': '土', '土': '金'}
    if element1 == element2:
        return 1
    elif element_generate.get(element1) == element2:
        return 2
    else:
        return 0

def get_zodiac_attributes(zodiac_id):
    odd_even = zodiac_id % 2
    big_small = 1 if zodiac_id >= 7 else 0
    if zodiac_id <= 4:
        zone = 0
    elif zodiac_id <= 8:
        zone = 1
    else:
        zone = 2
    head = 1 if zodiac_id >= 10 else 0
    tail = zodiac_id % 10
    return {
        'odd_even': odd_even,
        'big_small': big_small,
        'zone': zone,
        'head': head,
        'tail': tail
    }

def load_2026_data(file_path):
    """只加载2026年的数据"""
    try:
        df = pd.read_csv(file_path)
        df['period'] = df['period'].astype(int)
        df['zodiac'] = df['zodiac'].astype(int)
        
        # 只保留2026年的数据
        df_2026 = df[df['period'] >= 2026001].copy()
        df_2026 = df_2026.sort_values('period').reset_index(drop=True)
        
        print(f"="*60)
        print(f"数据加载成功")
        print(f"="*60)
        print(f"总数据量: {len(df)} 条")
        print(f"2026年数据: {len(df_2026)} 条")
        print(f"期号范围: {df_2026['period'].min()} - {df_2026['period'].max()}")
        print(f"\n生肖分布:")
        print(df_2026['zodiac'].value_counts().sort_index())
        
        return df_2026
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def build_features_for_training(df):
    """为训练构建特征"""
    n_samples = len(df)
    n_zodiacs = 12
    
    features_list = []
    labels = []
    
    # 从第51期开始构建特征（确保有足够的历史数据）
    start_idx = 50
    
    if n_samples <= start_idx:
        print(f"数据量不足，需要至少 {start_idx + 1} 条记录，当前只有 {n_samples} 条")
        return None, None, None
    
    print(f"\n开始构建特征...")
    
    for idx in range(start_idx, n_samples):
        current_zodiac = df.iloc[idx]['zodiac']
        history = df.iloc[:idx]
        
        features = []
        
        # 基础统计特征
        miss_counts = {i: 0 for i in range(1, n_zodiacs + 1)}
        max_miss = {i: 0 for i in range(1, n_zodiacs + 1)}
        
        for z in range(1, n_zodiacs + 1):
            last_appear = -1
            for i, row in history.iterrows():
                if row['zodiac'] == z:
                    last_appear = i
            if last_appear == -1:
                miss_counts[z] = len(history)
            else:
                miss_counts[z] = idx - last_appear - 1
        
        for z in range(1, n_zodiacs + 1):
            current_miss = 0
            for i, row in history.iterrows():
                if row['zodiac'] == z:
                    max_miss[z] = max(max_miss[z], current_miss)
                    current_miss = 0
                else:
                    current_miss += 1
        
        recent_10 = history.tail(10)
        recent_20 = history.tail(20)
        recent_50 = history.tail(50) if len(history) >= 50 else history
        
        counts_20 = recent_20['zodiac'].value_counts().to_dict()
        ranks = {}
        for z in range(1, n_zodiacs + 1):
            count = counts_20.get(z, 0)
            rank = 1
            for other_z in range(1, n_zodiacs + 1):
                if counts_20.get(other_z, 0) > count:
                    rank += 1
            ranks[z] = rank
        
        consecutive = {i: 0 for i in range(1, n_zodiacs + 1)}
        break_state = {i: 0 for i in range(1, n_zodiacs + 1)}
        
        for z in range(1, n_zodiacs + 1):
            cons = 0
            for i in range(len(history) - 1, -1, -1):
                if history.iloc[i]['zodiac'] == z:
                    cons += 1
                else:
                    break
            consecutive[z] = cons
            
            if len(history) >= 2:
                last = history.iloc[-1]['zodiac']
                second_last = history.iloc[-2]['zodiac']
                break_state[z] = 1 if (last == z and second_last != z) else 0
        
        for z in range(1, n_zodiacs + 1):
            miss = miss_counts[z]
            max_m = max_miss[z] if max_miss[z] > 0 else 1
            
            features.extend([
                miss,
                miss / max_m,
                (recent_10['zodiac'] == z).sum(),
                (recent_20['zodiac'] == z).sum(),
                (recent_50['zodiac'] == z).sum(),
                (recent_10['zodiac'] == z).sum() / 10 if len(recent_10) >= 10 else 0,
                (recent_20['zodiac'] == z).sum() / 20 if len(recent_20) >= 20 else 0,
                ranks[z],
                consecutive[z],
                break_state[z],
            ])
        
        # 动态特征
        prev_zodiac = history.iloc[-1]['zodiac']
        features.append(prev_zodiac)
        features.append(abs(current_zodiac - prev_zodiac))
        
        prev_element = ZODIAC_CONFIG['zodiac_to_element'][prev_zodiac]
        curr_element = ZODIAC_CONFIG['zodiac_to_element'][current_zodiac]
        features.append(get_element_relation(prev_element, curr_element))
        
        prev_attr = get_zodiac_attributes(prev_zodiac)
        curr_attr = get_zodiac_attributes(current_zodiac)
        
        features.extend([
            1 if prev_attr['odd_even'] == curr_attr['odd_even'] else 0,
            1 if prev_attr['big_small'] == curr_attr['big_small'] else 0,
            1 if prev_attr['zone'] == curr_attr['zone'] else 0,
            1 if prev_attr['head'] == curr_attr['head'] else 0,
            1 if prev_attr['tail'] == curr_attr['tail'] else 0,
        ])
        
        # 时序特征
        for z in range(1, n_zodiacs + 1):
            appear_indices = []
            for i, row in history.iterrows():
                if row['zodiac'] == z:
                    appear_indices.append(i)
            
            if len(appear_indices) >= 2:
                intervals = [appear_indices[i] - appear_indices[i-1] 
                           for i in range(1, len(appear_indices))]
                interval_mean = np.mean(intervals[-5:])
                interval_std = np.std(intervals[-5:]) if len(intervals) >= 5 else 0
            else:
                interval_mean = 0
                interval_std = 0
            
            features.extend([interval_mean, interval_std])
            
            if len(history) >= 20:
                recent_20_prev = history.iloc[-21:-1] if len(history) >= 21 else history.iloc[:-1]
                counts_20_prev = recent_20_prev['zodiac'].value_counts().to_dict()
                rank_prev = 1
                count_prev = counts_20_prev.get(z, 0)
                for other_z in range(1, n_zodiacs + 1):
                    if counts_20_prev.get(other_z, 0) > count_prev:
                        rank_prev += 1
                rank_change = rank_prev - ranks[z]
            else:
                rank_change = 0
            
            features.append(rank_change)
        
        features_list.append(features)
        labels.append(current_zodiac)
    
    X = np.array(features_list)
    y = np.array(labels)
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"标签形状: {y.shape}")
    
    return X, y

def train_model(X_train, y_train):
    """训练模型"""
    print("\n" + "="*60)
    print("开始训练模型...")
    print("="*60)
    
    params = {
        'n_estimators': 300,
        'max_depth': 12,
        'min_samples_split': 4,
        'min_samples_leaf': 2,
        'random_state': 42,
        'class_weight': 'balanced',
        'n_jobs': -1
    }
    
    print(f"模型参数: {params}")
    
    model = RandomForestClassifier(**params)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"模型训练完成，耗时: {train_time:.2f}秒")
    
    return model

def evaluate_model(model, X_test, y_test):
    """评估模型"""
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    
    y_proba = model.predict_proba(X_test)
    top3_correct = 0
    for i in range(len(y_test)):
        true_label = y_test[i]
        probs = y_proba[i]
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3_classes = [model.classes_[idx] for idx in top3_indices]
        if true_label in top3_classes:
            top3_correct += 1
    top3_accuracy = top3_correct / len(y_test)
    print(f"Top-3 准确率: {top3_accuracy:.4f}")
    
    try:
        # 只对模型已知的类别计算对数损失
        valid_indices = [i for i, label in enumerate(y_test) if label in model.classes_]
        if valid_indices:
            y_test_valid = [y_test[i] for i in valid_indices]
            y_proba_valid = y_proba[valid_indices]
            loss = log_loss(y_test_valid, y_proba_valid, labels=model.classes_)
            print(f"对数损失: {loss:.4f}")
        else:
            loss = float('inf')
            print(f"对数损失: 无法计算（测试集类别不在训练集中）")
    except Exception as e:
        loss = float('inf')
        print(f"对数损失: 无法计算 - {e}")
    
    return {
        'accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'log_loss': loss
    }

def predict_next(model, df):
    """预测下一期"""
    print("\n" + "="*60)
    print("下一期预测")
    print("="*60)
    
    n_zodiacs = 12
    idx = len(df)
    history = df
    
    features = []
    
    # 基础统计特征
    miss_counts = {i: 0 for i in range(1, n_zodiacs + 1)}
    max_miss = {i: 0 for i in range(1, n_zodiacs + 1)}
    
    for z in range(1, n_zodiacs + 1):
        last_appear = -1
        for i, row in history.iterrows():
            if row['zodiac'] == z:
                last_appear = i
        if last_appear == -1:
            miss_counts[z] = len(history)
        else:
            miss_counts[z] = idx - last_appear - 1
    
    for z in range(1, n_zodiacs + 1):
        current_miss = 0
        for i, row in history.iterrows():
            if row['zodiac'] == z:
                max_miss[z] = max(max_miss[z], current_miss)
                current_miss = 0
            else:
                current_miss += 1
    
    recent_10 = history.tail(10)
    recent_20 = history.tail(20)
    recent_50 = history.tail(50) if len(history) >= 50 else history
    
    counts_20 = recent_20['zodiac'].value_counts().to_dict()
    ranks = {}
    for z in range(1, n_zodiacs + 1):
        count = counts_20.get(z, 0)
        rank = 1
        for other_z in range(1, n_zodiacs + 1):
            if counts_20.get(other_z, 0) > count:
                rank += 1
        ranks[z] = rank
    
    consecutive = {i: 0 for i in range(1, n_zodiacs + 1)}
    break_state = {i: 0 for i in range(1, n_zodiacs + 1)}
    
    for z in range(1, n_zodiacs + 1):
        cons = 0
        for i in range(len(history) - 1, -1, -1):
            if history.iloc[i]['zodiac'] == z:
                cons += 1
            else:
                break
        consecutive[z] = cons
        
        if len(history) >= 2:
            last = history.iloc[-1]['zodiac']
            second_last = history.iloc[-2]['zodiac']
            break_state[z] = 1 if (last == z and second_last != z) else 0
    
    for z in range(1, n_zodiacs + 1):
        miss = miss_counts[z]
        max_m = max_miss[z] if max_miss[z] > 0 else 1
        
        features.extend([
            miss,
            miss / max_m,
            (recent_10['zodiac'] == z).sum(),
            (recent_20['zodiac'] == z).sum(),
            (recent_50['zodiac'] == z).sum(),
            (recent_10['zodiac'] == z).sum() / 10 if len(recent_10) >= 10 else 0,
            (recent_20['zodiac'] == z).sum() / 20 if len(recent_20) >= 20 else 0,
            ranks[z],
            consecutive[z],
            break_state[z],
        ])
    
    # 动态特征
    prev_zodiac = history.iloc[-1]['zodiac']
    features.append(prev_zodiac)
    features.append(0)
    features.append(1)
    
    features.extend([0, 0, 0, 0, 0])
    
    # 时序特征
    for z in range(1, n_zodiacs + 1):
        appear_indices = []
        for i, row in history.iterrows():
            if row['zodiac'] == z:
                appear_indices.append(i)
        
        if len(appear_indices) >= 2:
            intervals = [appear_indices[i] - appear_indices[i-1] 
                       for i in range(1, len(appear_indices))]
            interval_mean = np.mean(intervals[-5:])
            interval_std = np.std(intervals[-5:]) if len(intervals) >= 5 else 0
        else:
            interval_mean = 0
            interval_std = 0
        
        features.extend([interval_mean, interval_std, 0])
    
    X_next = np.array([features])
    
    probabilities = model.predict_proba(X_next)[0]
    
    pred_list = []
    for i, class_idx in enumerate(model.classes_):
        pred_list.append({
            'zodiac': class_idx,
            'name': ZODIAC_CONFIG['id_to_name'][class_idx],
            'probability': float(probabilities[i])
        })
    
    pred_list_sorted = sorted(pred_list, key=lambda x: x['probability'], reverse=True)
    
    print("\n生肖预测概率（按概率排序）:")
    print("-" * 40)
    for item in pred_list_sorted:
        prob = item['probability']
        bar = '█' * int(prob * 50)
        print(f"{item['zodiac']:2d}. {item['name']}: {prob:.4f} {bar}")
    
    print(f"\n推荐生肖: {pred_list_sorted[0]['name']} (概率: {pred_list_sorted[0]['probability']:.4f})")
    print(f"次选生肖: {pred_list_sorted[1]['name']} (概率: {pred_list_sorted[1]['probability']:.4f})")
    print(f"备选生肖: {pred_list_sorted[2]['name']} (概率: {pred_list_sorted[2]['probability']:.4f})")
    
    return pred_list_sorted

def save_model(model, file_path):
    """保存模型"""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n模型已保存: {file_path}")

def main():
    """主程序"""
    print("="*60)
    print("2026年数据模型训练与预测")
    print("="*60)
    
    # 加载2026年数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'real_lottery_history.csv')
    df = load_2026_data(data_path)
    
    if df is None:
        return
    
    # 准备训练数据
    X, y = build_features_for_training(df)
    if X is None:
        return
    
    # 划分训练集和测试集（80/20，按时间顺序）
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 训练模型
    model = train_model(X_train, y_train)
    
    # 评估模型
    metrics = evaluate_model(model, X_test, y_test)
    
    # 预测下一期
    pred_list = predict_next(model, df)
    
    # 保存模型
    model_path = os.path.join(script_dir, 'models', 'zodiac_model_2026_only.pkl')
    save_model(model, model_path)
    
    print("\n" + "="*60)
    print("训练与预测完成！")
    print("="*60)

if __name__ == '__main__':
    main()
