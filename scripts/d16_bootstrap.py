import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os

# 1. 读取 Day 15 生成的真实测试集预测明细
df = pd.read_csv("D:/CodexOutput/RecHub/d15_criteo/criteo_predictions.csv")
y_true = df['label'].values
y_pred_A = df['score'].values  # 模型A (DeepFM)

# 2. 模拟一个“稍微好一点点”的模型 B
# 我们给 A 加上一点点相关的噪声（相当于模型 B 的排序稍微强那么一丢丢）
np.random.seed(2022)
y_pred_B = y_pred_A + np.random.normal(0, 0.02, len(y_true))
y_pred_B = np.clip(y_pred_B, 0, 1)

# 观察原始的差异
auc_A = roc_auc_score(y_true, y_pred_A)
auc_B = roc_auc_score(y_true, y_pred_B)
delta_auc_original = auc_A - auc_B
print(f"模型 A (DeepFM) AUC: {auc_A:.4f}")
print(f"模型 B (模拟) AUC: {auc_B:.4f}")
print(f"原始 ΔAUC (A - B): {delta_auc_original:.4f}")

# 3. Bootstrap 过程
N_bootstraps = 1000  # 重采样次数
auc_diffs = []       # 存放 1000 次重采样的差值

for i in range(N_bootstraps):
    # 有放回重采样，得到长度为 N 的索引
    indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
    
    # 取出对应的样本（关键：必须是同一个 indices 去取 A 和 B，保持配对）
    y_t = y_true[indices]
    y_pA = y_pred_A[indices]
    y_pB = y_pred_B[indices]
    
    # 如果抽出来的样本全是正类或全是负类，AUC 没有定义，跳过
    if len(np.unique(y_t)) < 2:
        continue
        
    auc_A_boot = roc_auc_score(y_t, y_pA)
    auc_B_boot = roc_auc_score(y_t, y_pB)
    auc_diffs.append(auc_A_boot - auc_B_boot)

# 4. 计算置信区间
auc_diffs = np.array(auc_diffs)
lower_bound = np.percentile(auc_diffs, 2.5)
upper_bound = np.percentile(auc_diffs, 97.5)

print(f"\n有效重采样次数: {len(auc_diffs)} / {N_bootstraps}")
print(f"ΔAUC 的均值: {np.mean(auc_diffs):.4f}")
print(f"95% 置信区间: [{lower_bound:.4f}, {upper_bound:.4f}]")

if lower_bound <= 0 <= upper_bound:
    print("结论：区间包含 0，差异不显著（可能是随机波动）。")
else:
    print("结论：区间不包含 0，差异显著！")

# 5. 画出 ΔAUC 的分布直方图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(8, 5))
plt.hist(auc_diffs, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(lower_bound, color='red', linestyle='dashed', linewidth=2, label='2.5% 分位数')
plt.axvline(upper_bound, color='red', linestyle='dashed', linewidth=2, label='97.5% 分位数')
plt.axvline(0, color='green', linestyle='solid', linewidth=2, label='0 基准线')
plt.title('Bootstrap 重采样 ΔAUC 分布')
plt.xlabel('ΔAUC (模型A - 模型B)')
plt.ylabel('频数')
plt.legend()
plt.grid(True)

save_dir = "D:/CodexOutput/RecHub/d16_bootstrap"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "bootstrap_distribution.png"))
print(f"分布图已保存至: {save_dir}/bootstrap_distribution.png")