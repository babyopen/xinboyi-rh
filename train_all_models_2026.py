#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用2026年数据批量训练所有模型
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
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
    }
}

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
        df_2026 = df[df['period'] >= 2026001].copy()
        df_2026 = df_2026.sort_values('period').reset_index(drop=True)
        return df_2026
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def build_features_49(df):
    """构建49个特征（使用之前的特征构建方式）"""
    n_samples = len(df)
    n_zodiacs = 12
    features_list = []
    labels = []
    start_idx = 50
    
    if n_samples <= start_idx:
        return None, None
    
    for idx in range(start_idx, n_samples):
        current_zodiac = df.iloc[idx]['zodiac']
        history = df.iloc[:idx]
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
        
        features_list.append(features)
        labels.append(current_zodiac)
    
    return np.array(features_list), np.array(labels)

def build_features_62(df):
    """构建62个特征（legacy模型使用）"""
    n_samples = len(df)
    n_zodiacs = 12
    features_list = []
    labels = []
    start_idx = 50
    
    if n_samples <= start_idx:
        return None, None
    
    for idx in range(start_idx, n_samples):
        current_zodiac = df.iloc[idx]['zodiac']
        history = df.iloc[:idx]
        features = []
        
        # 每个生肖4个特征（共12*4=48）
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
            
            count_10 = (history.tail(10)['zodiac'] == z).sum() if len(history) >= 10 else (history['zodiac'] == z).sum()
            count_20 = (history.tail(20)['zodiac'] == z).sum()
            count_50 = (history.tail(50)['zodiac'] == z).sum() if len(history) >= 50 else (history['zodiac'] == z).sum()
            
            features.extend([miss, count_10, count_20, count_50])
        
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
        
        # 确保正好62个特征
        features = features[:62] if len(features) > 62 else features + [0]*(62-len(features))
        
        features_list.append(features)
        labels.append(current_zodiac)
    
    return np.array(features_list), np.array(labels)

def train_single_model(X, y, model_name, model_path, feature_type='49'):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"训练模型: {model_name}")
    print(f"{'='*60}")
    
    if X is None or len(X) == 0:
        print(f"数据不足，跳过 {model_name}")
        return None
    
    # 模型参数
    params = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'class_weight': 'balanced',
        'n_jobs': -1
    }
    
    if model_name == '优化参数模型':
        params = {
            'n_estimators': 300,
            'max_depth': 12,
            'min_samples_split': 4,
            'min_samples_leaf': 2,
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1
        }
    
    print(f"特征数量: {X.shape[1]}")
    print(f"训练样本: {len(X)}")
    print(f"模型参数: {params}")
    
    model = RandomForestClassifier(**params)
    
    start_time = time.time()
    model.fit(X, y)
    train_time = time.time() - start_time
    
    print(f"训练完成，耗时: {train_time:.2f}秒")
    
    # 保存模型
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"模型已保存: {model_path}")
    
    return model

def main():
    """主程序"""
    print("="*70)
    print("使用2026年数据批量训练所有模型")
    print("="*70)
    
    # 加载2026年数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'real_lottery_history.csv')
    df = load_2026_data(data_path)
    
    if df is None:
        print("无法加载数据，退出")
        return
    
    print(f"\n2026年数据: {len(df)} 条 (2026001 - {df['period'].max()})")
    print(f"生肖分布:\n{df['zodiac'].value_counts().sort_index()}")
    
    # 构建特征
    X_49, y_49 = build_features_49(df)
    X_62, y_62 = build_features_62(df)
    
    # 定义所有模型
    models_config = [
        {
            'name': '主模型',
            'path': os.path.join(script_dir, 'models', 'main', 'zodiac_model.pkl'),
            'features': '49'
        },
        {
            'name': '前80期数据模型',
            'path': os.path.join(script_dir, 'models', 'early80', 'zodiac_model_前80期数据.pkl'),
            'features': '49'
        },
        {
            'name': '后80期数据模型',
            'path': os.path.join(script_dir, 'models', 'later80', 'zodiac_model_后80期数据.pkl'),
            'features': '49'
        },
        {
            'name': '完整数据集模型',
            'path': os.path.join(script_dir, 'models', 'full', 'zodiac_model_完整数据集.pkl'),
            'features': '49'
        },
        {
            'name': '优化参数模型',
            'path': os.path.join(script_dir, 'models', 'optimized', 'zodiac_model_完整数据-调整参数.pkl'),
            'features': '49'
        },
        {
            'name': '遗留最佳模型',
            'path': os.path.join(script_dir, 'models', 'legacy', 'best_zodiac_model.pkl'),
            'features': '62'
        },
    ]
    
    # 训练所有模型
    trained_count = 0
    for config in models_config:
        X = X_49 if config['features'] == '49' else X_62
        y = y_49 if config['features'] == '49' else y_62
        
        model = train_single_model(X, y, config['name'], config['path'], config['features'])
        if model is not None:
            trained_count += 1
    
    print(f"\n{'='*70}")
    print(f"批量训练完成！成功训练: {trained_count}/{len(models_config)} 个模型")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
