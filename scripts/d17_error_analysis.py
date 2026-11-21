import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score
import os

# 1. 读取测试集预测结果
df = pd.read_csv("D:/CodexOutput/RecHub/d15_criteo/criteo_predictions.csv")

# 2. 按预测概率进行切片（分 5 个箱）
# 切分点：0, 0.2, 0.4, 0.6, 0.8, 1.0
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
df['prob_bin'] = pd.cut(df['score'], bins=bins, include_lowest=True)

print(">>> 误差切片数据表 (按预测概率分箱)")
results = []

for bin_interval, group in df.groupby('prob_bin', observed=False):
    n = len(group)
    if n == 0:
        continue
    
    # 计算切片内的核心指标
    true_pos_rate = group['label'].mean() # 真实点击率
    pred_mean = group['score'].mean()     # 平均预测值
    logloss = log_loss(group['label'], group['score'], labels=[0, 1])
    
    # AUC 需要切片内既有正样本又有负样本才能算
    if group['label'].nunique() > 1:
        auc = roc_auc_score(group['label'], group['score'])
        auc_str = f"{auc:.4f}"
    else:
        auc_str = "N/A (单一类别)"
        
    results.append({
        "概率区间": str(bin_interval),
        "样本量 N": n,
        "真实点击率": f"{true_pos_rate:.4f}",
        "平均预测分": f"{pred_mean:.4f}",
        "LogLoss": f"{logloss:.4f}",
        "区间内 AUC": auc_str
    })

# 打印漂亮的表格
result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))

# 3. 画图分析
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax1 = plt.subplots(figsize=(10, 6))

# 柱状图：样本量 (左轴)
x = np.arange(len(result_df))
width = 0.35
bars = ax1.bar(x - width/2, result_df['样本量 N'], width, label='样本量 (N)', color='lightgray')
ax1.set_ylabel('样本数量 (N)', color='gray')
ax1.set_xticks(x)
ax1.set_xticklabels(result_df['概率区间'])

# 折线图：真实点击率 vs 预测分 (右轴)
ax2 = ax1.twinx()
line1 = ax2.plot(x, result_df['真实点击率'].astype(float), 'bo-', label='真实点击率', linewidth=2)
line2 = ax2.plot(x, result_df['平均预测分'].astype(float), 'rs--', label='平均预测分', linewidth=2)
ax2.set_ylabel('概率', color='black')

# 合并图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title('Criteo 测试集误差切片分析 (按预测概率)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

save_dir = "D:/CodexOutput/RecHub/d17_error_analysis"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "error_slices.png"))
print(f"\n可视化图表已保存至: {save_dir}/error_slices.png")