import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入 xboyi 模块
from xboyi import ZodiacRules, FeatureEngineer

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 加载模型
def load_model():
    models = []
    calibrators = []
    for i in range(3):
        # 加载模型
        model_path = os.path.join(script_dir, f'zodiac_ensemble_{i}.pkl')
        with open(model_path, 'rb') as f:
            models.append(pickle.load(f))
        # 加载校准器
        cal_path = os.path.join(script_dir, f'zodiac_ensemble_cal_{i}.pkl')
        with open(cal_path, 'rb') as f:
            calibrators.append(pickle.load(f))
    
    # 加载编码器
    encoder_path = os.path.join(script_dir, 'zodiac_ensemble_encoder.pkl')
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    
    # 加载missing_max
    missing_path = os.path.join(script_dir, 'zodiac_ensemble_missing_max.pkl')
    with open(missing_path, 'rb') as f:
        missing_max = pickle.load(f)
    
    return models, calibrators, encoder, missing_max

# 准备真实数据
def prepare_data():
    # 读取真实历史数据
    import os
    # 回到项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 再上一层到项目根目录
    project_root = os.path.dirname(project_root)
    data_path = os.path.join(project_root, 'real_lottery_history.csv')
    
    df = pd.read_csv(data_path)
    df['period'] = df['period'].astype(str)
    df['draw_time'] = pd.to_datetime('2026-01-01') + pd.to_timedelta(df.index, unit='D')
    
    # 添加 normal_zodiacs 列（模拟数据）
    df['normal_zodiacs'] = df['zodiac'].apply(lambda x: [(x+j) % 12 + 1 for j in range(6)])
    df = df.rename(columns={'zodiac': 'special_zodiac'})
    
    return df

# 构建特征
def build_features(df):
    fe = FeatureEngineer()
    X, y = fe.build_features(df)
    return X, y

# 集成预测
def predict_ensemble(models, calibrators, X):
    all_proba = []
    for cal in calibrators:
        proba = cal.predict_proba(X)
        all_proba.append(proba)
    return np.mean(all_proba, axis=0)

# 预测
def predict():
    try:
        # 加载模型
        models, calibrators, encoder, missing_max = load_model()
        
        # 准备数据
        df = prepare_data()
        
        # 构建特征
        X, y = build_features(df)
        
        # 确保特征列与模型一致
        if hasattr(models[0], 'feature_names_in_'):
            feature_names = models[0].feature_names_in_
            for col in feature_names:
                if col not in X.columns:
                    X[col] = 0
            X = X[feature_names]
        
        # 预测
        proba = predict_ensemble(models, calibrators, X)
        
        # 获取最后一行的预测结果
        last_proba = proba[-1]
        
        # 映射到生肖
        zodiac_map = {
            1: '马', 2: '蛇', 3: '龙', 4: '兔', 5: '虎', 6: '牛',
            7: '鼠', 8: '猪', 9: '狗', 10: '鸡', 11: '猴', 12: '羊'
        }
        
        # 排序预测结果
        sorted_indices = np.argsort(last_proba)[::-1]
        
        print("Xboyi模型预测结果:")
        print("=" * 50)
        
        for i, idx in enumerate(sorted_indices[:3]):
            zodiac_code = encoder.inverse_transform([idx])[0]
            zodiac_name = zodiac_map.get(zodiac_code, f'未知({zodiac_code})')
            probability = last_proba[idx] * 100
            print(f"{i+1}. {zodiac_name} - {probability:.2f}%")
        
        return sorted_indices, last_proba
        
    except Exception as e:
        print(f"预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    predict()
