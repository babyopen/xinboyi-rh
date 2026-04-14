
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.model_service import model_service
from src.services.data_service import data_service
from src.config.zodiac_config import ZODIAC_MAP, ZODIAC_ELEMENT_MAP, ZODIAC_COLOR_MAP

def main():
    print("=" * 60)
    print("生肖预测系统 - 预测结果")
    print("=" * 60)
    
    # 加载模型
    print("\n正在加载模型...")
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'zodiac_model.pkl')
    model = model_service.load_model(model_path)
    if not model:
        print("模型加载失败")
        return
    
    print("模型加载成功")
    
    # 获取历史数据
    print("\n正在读取历史数据...")
    df = data_service.get_real_history_data()
    if df is None:
        print("无法读取历史数据")
        return
    
    print(f"历史数据读取成功，共 {len(df)} 期")
    
    # 确保数据量足够
    df = data_service.ensure_min_history_length(df)
    if df is None:
        print("历史数据不足")
        return
    
    # 获取最新数据
    last_row = data_service.get_latest_period_data(df)
    if last_row is None:
        print("无法获取最新数据")
        return
    
    # 进行预测
    print("\n正在进行预测...")
    predictions = model_service.predict_next(last_row, df)
    if predictions is None:
        print("预测失败")
        return
    
    # 格式化结果
    results = []
    for i, prob in enumerate(predictions):
        zodiac_num = i + 1
        results.append({
            'name': ZODIAC_MAP.get(zodiac_num, f'未知{zodiac_num}'),
            'number': zodiac_num,
            'probability': float(prob),
            'element': ZODIAC_ELEMENT_MAP.get(zodiac_num, ''),
            'color': ZODIAC_COLOR_MAP.get(zodiac_num, '')
        })
    
    # 按概率排序
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    # 显示结果
    print("\n" + "=" * 60)
    print("🏆 预测结果")
    print("=" * 60)
    
    print("\n🎯 Top 3 推荐:")
    zodiac_emoji = {1: '🐎', 2: '🐍', 3: '🐉', 4: '🐇', 5: '🐅', 6: '🐂', 7: '🐀', 8: '🐖', 9: '🐕', 10: '🐓', 11: '🐒', 12: '🐑'}
    
    for i, item in enumerate(results[:3], 1):
        rank = '🥇' if i == 1 else '🥈' if i == 2 else '🥉'
        print(f'  {rank} {i}. {zodiac_emoji[item["number"]]} {item["name"]} (ID: {item["number"]}) - {(item["probability"] * 100):.2f}%')
    
    print("\n📊 所有生肖概率:")
    for item in results:
        bar = '█' * int(item['probability'] * 30)
        print(f'  {zodiac_emoji[item["number"]]} {item["name"]:2s} (ID: {item["number"]:2d}) - {(item["probability"] * 100):6.2f}% {bar}')
    
    print("\n" + "=" * 60)
    print("⚠️  注意: 本预测仅供娱乐，非投注建议")
    print("=" * 60)

if __name__ == "__main__":
    main()
