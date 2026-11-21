import sys
import pandas as pd
import torch

# 让 Python 能找到上一级目录的 torch_rechub 库，以及当前目录的 run_criteo
sys.path.append("../../")
sys.path.append(".")

from run_criteo import get_criteo_data_dict
from torch_rechub.utils.data import DataGenerator

DATA = "data/criteo/criteo_sample.csv"

# 1. 原表信息（你已经看过了，这里为了完整性再打印一次）
df = pd.read_csv(DATA)
print("=== 1. 原始数据 ===")
print(f"形状: {df.shape}")
print(f"标签分布:\n{df['label'].value_counts()}")
print(f"缺失率 Top 5:\n{df.isna().mean().sort_values(ascending=False).head()}")

# 2. 走一遍官方预处理
print("\n=== 2. 预处理后 ===")
# 这里下划线 _ 表示我们暂时不需要那两套 FFM 特征
dense_feas, sparse_feas, _, _, x, y = get_criteo_data_dict(DATA)
print(f"处理后的 x 形状: {x.shape}")
print(f"处理后的 y 形状: {y.shape}")
print(f"稠密特征(DenseFeature)数: {len(dense_feas)}")
print(f"稀疏特征(SparseFeature)数: {len(sparse_feas)}")

# 3. 构造 DataLoader
print("\n=== 3. 构造 DataLoader ===")
dg = DataGenerator(x, y)
# 按 70% / 10% 划分，batch_size 设为 16
train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=16)

# 4. 取一个 batch 看看
print("\n=== 4. 一个 Batch 的信息 ===")
# next(iter(...)) 表示从训练集中抓取第一批数据
x_dict, y_batch = next(iter(train_dl))

print(f"y_batch 形状: {y_batch.shape}, 类型: {y_batch.dtype}")
print(f"x_dict 包含的键数量: {len(x_dict.keys())}")

print("\n挑选几个代表性的键，查看其形状和类型:")
# I1 代表缩放后的数值特征，C1 代表编码后的类别特征，I1_cat 代表离散化后的数值特征
for key in ["I1", "C1", "I1_cat"]:
    if key in x_dict:
        print(f"  {key:6s} -> 形状: {x_dict[key].shape}, 类型: {x_dict[key].dtype}")