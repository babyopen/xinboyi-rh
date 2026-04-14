#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用模型预测脚本
"""

import sys
import os
import pickle
import pandas as pd
import numpy as np

# 添加python目录到路径 - 从models目录的角度看
script_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.join(script_dir, '../python')
sys.path.insert(0, os.path.normpath(python_dir))

from zodiac_ml_predictor import (
    load_model, 
    predict_next, 
    ZODIAC_CONFIG
)

def load_history_data():
    """加载历史数据"""
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据文件的绝对路径
    history_path = os.path.join(script_dir, '../data/real_lottery_history.csv')
    history_path = os.path.normpath(history_path)
    try:
        df = pd.read_csv(history_path)
        print(f"✓ 历史数据加载成功: {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"✗ 历史数据加载失败: {e}")
        return None

def predict_with_model(model_path, model_name="模型"):
    """使用指定模型进行预测"""
    print(f"\n{'='*60}")
    print(f"【{model_name}】预测")
    print(f"{'='*60}")
    
    try:
        # 获取当前脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 如果路径是相对路径，构建绝对路径
        if not os.path.isabs(model_path):
            # 先尝试从调用脚本的目录找
            caller_dir = os.getcwd()
            full_path = os.path.join(caller_dir, model_path)
            if os.path.exists(full_path):
                model_path = full_path
            else:
                # 再尝试从models目录找
                model_path = os.path.join(script_dir, model_path)
        
        # 加载模型
        model = load_model(model_path)
        print(f"✓ 模型加载成功: {model_path}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None
    
    # 加载历史数据
    df = load_history_data()
    if df is None:
        return None
    
    try:
        # 进行预测
        probabilities = predict_next(model, df.iloc[-1], df)
        
        if probabilities is not None:
            # 排序并显示概率
            zodiac_probs = [(i+1, ZODIAC_CONFIG['id_to_name'][i+1], probabilities[i]) 
                           for i in range(12)]
            zodiac_probs.sort(key=lambda x: x[2], reverse=True)
            
            print(f"\n{model_name} - 预测结果:")
            for i in range(3):
                zodiac_id, zodiac_name, prob = zodiac_probs[i]
                print(f"  {zodiac_name} (ID: {zodiac_id}) - 概率: {prob:.2%}")
            
            print(f"\n推荐: {zodiac_probs[0][1]}")
            
            return {
                'predictions': [
                    {'id': zodiac_id, 'name': zodiac_name, 'probability': prob}
                    for zodiac_id, zodiac_name, prob in zodiac_probs
                ],
                'recommendation': {
                    'id': zodiac_probs[0][0],
                    'name': zodiac_probs[0][1]
                }
            }
    except Exception as e:
        print(f"预测失败: {e}")
    
    return None

if __name__ == "__main__":
    print("生肖预测器")
    print("请在各模型文件夹中运行对应的脚本")
