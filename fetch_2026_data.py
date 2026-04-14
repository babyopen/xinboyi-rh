#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取2026年所有开奖历史数据，截止到2026104期
"""

import requests
import pandas as pd
import os
from datetime import datetime

# 前端API地址
API_URL = 'https://history.macaumarksix.com/history/macaujc2/y/{year}'

# 生肖映射（从前端代码中提取）
zodiac_map = {
    '鼠': 7,
    '牛': 6,
    '虎': 5,
    '兔': 4,
    '龙': 3,
    '蛇': 2,
    '马': 1,
    '羊': 12,
    '猴': 11,
    '鸡': 10,
    '狗': 9,
    '猪': 8
}

def fetch_year_history(year):
    """获取指定年份的历史数据"""
    url = API_URL.format(year=year)
    max_retries = 3
    
    for retry in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data
        except Exception as e:
            print(f"获取{year}年数据失败 (尝试 {retry+1}/{max_retries}): {e}")
            if retry < max_retries - 1:
                import time
                time.sleep(2)
            else:
                return None

def process_data(data, year):
    """处理API返回的数据"""
    if not data or 'data' not in data:
        return []
    
    history = []
    for item in data['data']:
        expect = item.get('expect')
        opencode = item.get('openCode')
        opentime = item.get('opentime')
        
        if not expect or not opencode:
            continue
        
        # 检查期号是否属于当前年份
        expect_str = str(expect)
        if len(expect_str) < 4:
            continue
        
        expect_year = int(expect_str[:4])
        if expect_year != year:
            continue
        
        # 检查期号是否大于2026104，如果是就跳过
        try:
            full_period = int(expect)
            if full_period > 2026104:
                continue
        except ValueError:
            continue
        
        # 解析开奖号码
        numbers = opencode.split(',')
        if len(numbers) != 7:
            continue
        
        special_num = int(numbers[-1])
        zodiac_num = (special_num - 1) % 12 + 1
        
        zodiac_name = None
        for name, num in zodiac_map.items():
            if num == zodiac_num:
                zodiac_name = name
                break
        
        if not zodiac_name:
            continue
        
        history.append({
            'period': full_period,
            'zodiac': zodiac_num,
            'zodiac_name': zodiac_name,
            'special_num': special_num,
            'opentime': opentime
        })
    
    return history

def main():
    """主函数"""
    print("="*60)
    print("获取2026年开奖历史数据（截止2026104期）")
    print("="*60)
    
    # 获取2026年数据
    year = 2026
    print(f"\n获取{year}年数据...")
    data = fetch_year_history(year)
    
    if not data:
        print(f"未获取到{year}年数据")
        return
    
    year_history = process_data(data, year)
    print(f"{year}年获取到{len(year_history)}条数据")
    
    if not year_history:
        print("未获取到任何有效数据")
        return
    
    # 按期号排序
    year_history.sort(key=lambda x: x['period'])
    
    df = pd.DataFrame(year_history)
    
    # 只保留模型训练需要的列
    df_train = df[['period', 'zodiac']]
    
    # 保存到主数据文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_data_path = os.path.join(script_dir, 'data', 'real_lottery_history.csv')
    
    # 检查是否存在旧数据
    if os.path.exists(main_data_path):
        old_df = pd.read_csv(main_data_path)
        print(f"\n现有数据有{len(old_df)}条数据")
        
        # 合并旧数据和新数据
        combined = pd.concat([old_df, df_train])
        # 去重（按期号）
        combined = combined.drop_duplicates(subset=['period'], keep='last')
        # 排序
        combined = combined.sort_values('period')
        df_final = combined
    else:
        df_final = df_train
    
    df_final.to_csv(main_data_path, index=False)
    
    # 同时保存到根目录的副本
    root_data_path = os.path.join(script_dir, 'real_lottery_history.csv')
    df_final.to_csv(root_data_path, index=False)
    
    print(f"\n数据处理完成!")
    print(f"总数据量: {len(df_final)}条")
    print(f"2026年数据: {len(df)}条")
    print(f"期号范围: {df_final['period'].min()} - {df_final['period'].max()}")
    print(f"数据保存到: {main_data_path}")
    print(f"同时保存到: {root_data_path}")
    
    # 显示2026年的数据
    print(f"\n2026年最新数据:")
    print(df.tail(10))

if __name__ == '__main__':
    main()
