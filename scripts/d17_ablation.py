import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# 导入你 Day 15 写好的数据处理函数
from d15_criteo_standalone import process_data
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

# ==========================================
# 1. 定义消融模型 (去掉 FM 部分)
# ==========================================
# ==========================================
# 1. 定义消融模型 (物理切断 FM)
# ==========================================
# ==========================================
# 1. 定义消融模型 (物理切断 FM)
# ==========================================
class Ablated_DeepFM(DeepFM):
    def __init__(self, deep_features, fm_features, mlp_params):
        # 继承父类，不需要改任何维度
        super().__init__(deep_features, fm_features, mlp_params)

    def forward(self, x):
        # 完全复刻 deepfm.py 的源码逻辑
        input_deep = self.embedding(x, self.deep_features, squeeze_dim=True)
        input_fm = self.embedding(x, self.fm_features, squeeze_dim=False)

        y_linear = self.linear(input_fm.flatten(start_dim=1))
        y_fm = self.fm(input_fm)
        y_deep = self.mlp(input_deep)
        
        # 核心消融：不加 y_fm（或者写成 y_linear + y_fm * 0.0 + y_deep）
        y = y_linear + y_deep
        
        return torch.sigmoid(y.squeeze(1))

def run_experiment(model_class, is_ablation=False):
    data_path = "D:/RecStudy/torch-rechub/examples/ranking/data/criteo/criteo_100k.tsv.txt"
    
    # 复用刚才的数据处理 (复用同一个划分，保证对比公平)
    dense_feas, sparse_feas, x, y = process_data(data_path)
    dg = DataGenerator(x, y)
    # 注意：这里固定随机种子，保证每次划分完全一样
    torch.manual_seed(2022) 
    train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.8, 0.1], batch_size=2048, num_workers=0)
    
    if is_ablation:
        model = model_class(dense_feas, sparse_feas, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
    else:
        # 原始模型
        model = model_class(deep_features=dense_feas, fm_features=sparse_feas, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
        
    trainer = CTRTrainer(model, optimizer_params={"lr": 1e-3, "weight_decay": 1e-5}, n_epoch=2, earlystop_patience=2, device='cpu', model_path='./')
    trainer.fit(train_dl, val_dl)
    
    # 预测测试集
    model.eval()
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for x_dict, y_batch in tqdm(test_dl, desc="预测中"):
            x_dict = {k: v.to('cpu') for k, v in x_dict.items()}
            logits = model(x_dict)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            y_pred_list.extend(probs)
            y_true_list.extend(y_batch.cpu().numpy().flatten())
            
    auc = roc_auc_score(y_true_list, y_pred_list)
    return auc

if __name__ == "__main__":
    print(">>> 1. 跑完整的 DeepFM")
    auc_full = run_experiment(DeepFM, is_ablation=False)
    
    print("\n>>> 2. 跑砍掉 FM 的 DeepFM (消融版)")
    auc_ablated = run_experiment(Ablated_DeepFM, is_ablation=True)
    
    print("\n========== 消融实验结果 ==========")
    print(f"完整 DeepFM AUC: {auc_full:.4f}")
    print(f"消融版 (去FM) AUC: {auc_ablated:.4f}")
    print(f"性能下降幅度 (Delta): {auc_full - auc_ablated:.4f}")