import sys
import os
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

sys.path.append("../../")
sys.path.append(".")

from run_criteo import get_criteo_data_dict
from torch_rechub.utils.data import DataGenerator
from torch_rechub.models.ranking import DeepFM, WideDeep
from torch_rechub.trainers import CTRTrainer

# 设置固定种子
torch.manual_seed(2022)
np.random.seed(2022)

DATA = "data/criteo/criteo_sample.csv"
OUT_DIR = "D:/Codex输出/Torch-RecHub学习计划/实验/d03"

os.makedirs(OUT_DIR, exist_ok=True)

# 1. 准备数据
print("=== 1. 准备数据 ===")
dense_feas, sparse_feas, _, _, x, y = get_criteo_data_dict(DATA)
dg = DataGenerator(x, y)
# 划出 70% train, 10% val, 20% test
train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=16)

# 2. 实例化模型
#model = DeepFM(
#    deep_features=dense_feas, 
#    fm_features=sparse_feas, 
#    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"}
#)
model = WideDeep(
    wide_features=dense_feas, 
    deep_features=sparse_feas, 
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"}
)

# 3. 实例化 Trainer
trainer = CTRTrainer(
    model, 
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3}, 
    n_epoch=2,  # 小样本只跑 2 轮
    earlystop_patience=10, 
    device="cpu", 
    model_path=OUT_DIR
)

# 4. 开始训练
print("\n=== 2. 开始训练 ===")
trainer.fit(train_dl, val_dl)

# 5. 在测试集上预测
print("\n=== 3. 预测测试集 ===")
predicts = trainer.predict(trainer.model, test_dl)

# 6. 提取测试集的真实标签
# 注意：因为 test_dl 的 shuffle=False，所以顺序是固定的
test_targets = []
for _, y_batch in test_dl:
    test_targets.extend(y_batch.tolist())

# 7. 计算指标
y_true = np.array(test_targets).flatten()
y_pred = np.array(predicts).flatten()

auc = roc_auc_score(y_true, y_pred)
loss = log_loss(y_true, y_pred)
print(f"\nTest AUC: {auc:.4f}")
print(f"Test LogLoss: {loss:.4f}")

# 8. 保存逐行预测结果
result_df = pd.DataFrame({
    "label": y_true,
    "pred": y_pred
})
save_path = os.path.join(OUT_DIR, "test_predictions.csv")
result_df.to_csv(save_path, index=False)
print(f"\n逐行预测已保存至: {save_path}")