import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. 读数据与预处理 (与之前一模一样)
DATA_PATH = "data/criteo/criteo_100k.tsv.txt"
cols = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
data = pd.read_csv(DATA_PATH, sep='\t', header=None, names=cols)
data = data.drop_duplicates().head(50000).copy()

dense_features = [f'I{i}' for i in range(1, 14)]
sparse_features = [f'C{i}' for i in range(1, 27)]

data[sparse_features] = data[sparse_features].fillna('0')
data[dense_features] = data[dense_features].fillna(0)

# 按同样的 seed 和比例划分
train_val, test = train_test_split(data, test_size=0.2, random_state=2022, stratify=data['label'])
train, val = train_test_split(train_val, test_size=0.2, random_state=2022, stratify=train_val['label'])

# 把 x 和 y 分开
train_x, train_y = train.drop('label', axis=1), train['label']
test_x, test_y = test.drop('label', axis=1), test['label']

print("=== 开始训练 LR 基线 ===")

# 2. 核心：构建 ColumnTransformer（统计防泄漏版）
# 数值特征做 MinMax 缩放；类别特征做 One-Hot 编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), dense_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), sparse_features)
    ]
)

# 3. 构建 Pipeline：先预处理，再扔进 LR
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, C=1.0, random_state=2022))
])

# 4. 训练
lr_pipeline.fit(train_x, train_y)

# 5. 预测与评估
y_pred_proba = lr_pipeline.predict_proba(test_x)[:, 1]

auc = roc_auc_score(test_y, y_pred_proba)
ll = log_loss(test_y, y_pred_proba)
print(f"【Logistic Regression 最终结果】 Test AUC: {auc:.4f} | LogLoss: {ll:.4f}")