#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看2026年的历史数据
"""

import pandas as pd
import os

# 生肖映射配置
ZODIAC_CONFIG = {
    'id_to_name': {
        1: '马', 2: '蛇', 3: '龙', 4: '兔', 5: '虎', 6: '牛',
        7: '鼠', 8: '猪', 9: '狗', 10: '鸡', 11: '猴', 12: '羊'
    }
}

def main():
    """主程序"""
    print("="*80)
    print("2026年历史数据查看")
    print("="*80)
    
    # 加载数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'real_lottery_history.csv')
    
    df = pd.read_csv(data_path)
    df['period'] = df['period'].astype(int)
    df['zodiac'] = df['zodiac'].astype(int)
    
    # 只显示2026年的数据
    df_2026 = df[df['period'] >= 2026001].copy()
    df_2026 = df_2026.sort_values('period').reset_index(drop=True)
    
    # 添加生肖名称列
    df_2026['zodiac_name'] = df_2026['zodiac'].map(ZODIAC_CONFIG['id_to_name'])
    
    print(f"\n数据概况:")
    print(f"  总数据量: {len(df)} 条")
    print(f"  2026年数据: {len(df_2026)} 条")
    print(f"  期号范围: {df_2026['period'].min()} - {df_2026['period'].max()}")
    
    print(f"\n生肖分布:")
    print(df_2026['zodiac'].value_counts().sort_index())
    print("\n生肖分布（带名称）:")
    zodiac_dist = df_2026.groupby(['zodiac', 'zodiac_name']).size().reset_index(name='count')
    zodiac_dist = zodiac_dist.sort_values('zodiac')
    for _, row in zodiac_dist.iterrows():
        print(f"  {row['zodiac']:2d} {row['zodiac_name']:2s}: {row['count']:2d} 次")
    
    print(f"\n{'='*80}")
    print(f"2026年完整数据（期号、生肖码、生肖名称）:")
    print(f"{'='*80}")
    
    # 按20期一组显示
    for i in range(0, len(df_2026), 20):
        end_idx = min(i + 20, len(df_2026))
        print(f"\n第 {i+1} - {end_idx} 期:")
        print("-" * 50)
        for j in range(i, end_idx):
            row = df_2026.iloc[j]
            print(f"  {row['period']:6d} | {row['zodiac']:2d} {row['zodiac_name']:2s}")
    
    print(f"\n{'='*80}")
    print(f"最后10期数据:")
    print(f"{'='*80}")
    print(df_2026[['period', 'zodiac', 'zodiac_name']].tail(10).to_string(index=False))

if __name__ == '__main__':
    main()
