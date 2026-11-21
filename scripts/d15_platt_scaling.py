import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import os

# 读取多跑几个 epoch 后生成的预测文件
df = pd.read_csv("D:/CodexOutput/RecHub/d15_criteo/criteo_predictions.csv")
y_true = df['label'].values
y_raw = df['score'].values # 原始未校准的概率

# 我们模拟把数据切一半，一半用来拟合 Platt 参数，一半用来画图验证
# 现实中，你会用独立的验证集来拟合，用测试集来评估
split_idx = len(y_true) // 2
y_true_fit, y_raw_fit = y_true[:split_idx], y_raw[:split_idx]
y_true_test, y_raw_test = y_true[split_idx:], y_raw[split_idx:]

# ==========================================
# 1. 原始模型的表现
# ==========================================
auc_raw = roc_auc_score(y_true_test, y_raw_test)
brier_raw = brier_score_loss(y_true_test, y_raw_test)
print(f"原始模型: AUC={auc_raw:.4f}, Brier={brier_raw:.4f}")

# ==========================================
# 2. 拟合 Platt Scaling (逻辑回归)
# ==========================================
# 在验证集上，拟合真实标签 y 与原始输出 y_raw 之间的关系
# 注意：y_raw 不能是一维列向量，需要用 reshape 变成 (N, 1)
platt_model = LogisticRegression()
platt_model.fit(y_raw_fit.reshape(-1, 1), y_true_fit)

# 提取参数 A 和 B
A = platt_model.coef_[0][0]
B = platt_model.intercept_[0]
print(f"拟合出的参数: A={A:.4f}, B={B:.4f}")

# 应用到测试集上
y_calibrated = platt_model.predict_proba(y_raw_test.reshape(-1, 1))[:, 1]
auc_cal = roc_auc_score(y_true_test, y_calibrated)
brier_cal = brier_score_loss(y_true_test, y_calibrated)
print(f"校准后模型: AUC={auc_cal:.4f}, Brier={brier_cal:.4f}")

# ==========================================
# 3. 画图对比（原始 vs 校准后）
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：原始曲线
prob_true_raw, prob_pred_raw = calibration_curve(y_true_test, y_raw_test, n_bins=10)
axes[0].plot([0, 1], [0, 1], 'k--', label='完美校准')
axes[0].plot(prob_pred_raw, prob_true_raw, 'bo-', label='原始输出')
axes[0].set_title(f'原始模型 (Brier={brier_raw:.4f})')
axes[0].set_xlabel('预测概率')
axes[0].set_ylabel('真实点击率')
axes[0].legend()
axes[0].grid(True)

# 右图：Platt Scaling 后
prob_true_cal, prob_pred_cal = calibration_curve(y_true_test, y_calibrated, n_bins=10)
axes[1].plot([0, 1], [0, 1], 'k--', label='完美校准')
axes[1].plot(prob_pred_cal, prob_true_cal, 'ro-', label='Platt 校准后')
axes[1].set_title(f'Platt 校准后 (Brier={brier_cal:.4f})')
axes[1].set_xlabel('预测概率')
axes[1].set_ylabel('真实点击率')
axes[1].legend()
axes[1].grid(True)

save_dir = "D:/CodexOutput/RecHub/d15_criteo"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "platt_scaling_compare.png"))
print(f"对比图已保存至: {save_dir}/platt_scaling_compare.png")