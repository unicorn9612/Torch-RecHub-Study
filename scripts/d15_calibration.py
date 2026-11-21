import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
import os

# ==========================================
# 1. 读取你的预测文件 (加上 .csv 后缀)
# ==========================================
file_path = "D:/CodexOutput/RecHub/d06_predictions.csv"
df = pd.read_csv(file_path)

print("数据前5行：")
print(df.head(5))

# 你的文件里列名就是 label 和 score
y_true = df['label'].values 
y_pred = df['score'].values 

# ==========================================
# 2. 计算指标
# ==========================================
auc = roc_auc_score(y_true, y_pred)
brier = brier_score_loss(y_true, y_pred)
print(f"\n全局 AUC: {auc:.4f}")
print(f"Brier Score: {brier:.4f} (越小越好)")

# ==========================================
# 3. 画校准曲线 (针对极小样本优化)
# ==========================================
# 数据太少，分 10 个箱子会报错，我们只分 5 个箱子
# strategy='quantile' 代表等频分箱（每个箱子里的数据量尽量一样）
prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=5, strategy='quantile')

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly calibrated')
plt.plot(prob_pred, prob_true, marker='o', color='blue', label='Model')
plt.xlabel('Mean predicted probability (模型预测概率)')
plt.ylabel('Fraction of positives (真实点击率)')
plt.title('Calibration Curve (Sample Size=32)')
plt.legend()
plt.grid(True)

save_dir = "D:/CodexOutput/RecHub/d15_calibration"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "calibration_curve.png"))
print(f"\n图表已保存至: {save_dir}/calibration_curve.png")