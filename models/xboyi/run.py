import os
import sys
import pickle
import numpy as np

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 加载模型
def load_model():
    models = []
    for i in range(3):
        model_path = os.path.join(script_dir, f'zodiac_ensemble_{i}.pkl')
        with open(model_path, 'rb') as f:
            models.append(pickle.load(f))
    
    # 加载编码器
    encoder_path = os.path.join(script_dir, 'zodiac_ensemble_encoder.pkl')
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    
    # 加载missing_max
    missing_path = os.path.join(script_dir, 'zodiac_ensemble_missing_max.pkl')
    with open(missing_path, 'rb') as f:
        missing_max = pickle.load(f)
    
    return models, encoder, missing_max

# 准备数据
def prepare_data():
    # 这里使用简化的特征，实际使用时需要根据模型要求准备特征
    # 由于没有具体的特征工程代码，这里返回一个示例特征向量
    return np.random.rand(1, 10)  # 假设模型需要10维特征

# 预测
def predict():
    try:
        models, encoder, missing_max = load_model()
        
        # 准备数据
        X = prepare_data()
        
        # 集成预测
        predictions = []
        for model in models:
            pred = model.predict_proba(X)[0]
            predictions.append(pred)
        
        # 平均预测概率
        avg_pred = np.mean(predictions, axis=0)
        
        # 获取top3预测
        top_indices = np.argsort(avg_pred)[::-1][:3]
        
        # 生肖映射
        zodiac_map = {
            0: '鼠', 1: '牛', 2: '虎', 3: '兔', 4: '龙', 5: '蛇',
            6: '马', 7: '羊', 8: '猴', 9: '鸡', 10: '狗', 11: '猪'
        }
        
        print("Xboyi模型预测结果:")
        print("=" * 50)
        
        for i, idx in enumerate(top_indices):
            zodiac_name = zodiac_map.get(idx, f'未知({idx})')
            probability = avg_pred[idx] * 100
            print(f"{i+1}. {zodiac_name} - {probability:.2f}%")
        
        return top_indices, avg_pred
        
    except Exception as e:
        print(f"预测失败: {e}")
        return None, None

if __name__ == "__main__":
    predict()
