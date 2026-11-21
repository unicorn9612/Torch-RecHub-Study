import pandas as pd

for split in ['train', 'val', 'test']:
    path = f"./data/ali-ccp/ali_ccp_{split}_sample.csv"
    try:
        df = pd.read_csv(path)
        print(f"\n=== {split.upper()} 数据集 ===")
        print(f"总行数: {len(df)}")
        
        # 统计 click 和 purchase 的频数
        click_counts = df['click'].value_counts().to_dict()
        purchase_counts = df['purchase'].value_counts().to_dict()
        
        print(f"Click 标签分布 (1点击/0未点): {click_counts}")
        print(f"Purchase 标签分布 (1购买/0未买): {purchase_counts}")
        
        # 验证 purchase <= click 的数学真理 (也就是不存在：没点击却购买了的样本)
        anomaly_count = len(df[(df['purchase'] == 1) & (df['click'] == 0)])
        print(f"异常样本数 (Purchase=1 但 Click=0): {anomaly_count}")
        
    except FileNotFoundError:
        print(f"找不到文件: {path}")