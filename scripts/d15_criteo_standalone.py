import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from tqdm import tqdm

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

def convert_numeric_feature(val):
    v = int(val)
    if v > 2:
        return int(np.log(v)**2)
    else:
        return v - 2

def process_data(data_path):
    # 强制定义 Criteo 的标准列名，防止文件没有表头
    col_names = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
    
    print("尝试读取数据...")
    # 先尝试用 Tab 分隔读取
    data = pd.read_csv(data_path, sep='\t', header=None, names=col_names)
    
    # 如果读取出来不是 40 列，说明可能不是 Tab 分隔（或者是空格分隔）
    if data.shape[1] != 40:
        print("Tab 分隔读取列数不对，尝试使用正则空白符分隔...")
        data = pd.read_csv(data_path, sep=r'\s+', header=None, names=col_names)
        
    print(f"数据加载完成, 形状: {data.shape}")
    
    dense_features = [f'I{i}' for i in range(1, 14)]
    sparse_features = [f'C{i}' for i in range(1, 27)]
    
    # 处理缺失值
    data[sparse_features] = data[sparse_features].fillna('0').astype(str)
    data[dense_features] = data[dense_features].fillna(0).astype(float)
    
    # 连续特征分桶转稀疏特征
    for feat in tqdm(dense_features, desc="离散化数值特征"):
        sparse_features.append(feat + "_cat")
        data[feat + "_cat"] = data[feat].apply(lambda x: convert_numeric_feature(x))
        
    # 归一化连续特征
    sca = MinMaxScaler()
    data[dense_features] = sca.fit_transform(data[dense_features])
    
    # LabelEncoder 编码
    for feat in tqdm(sparse_features, desc="编码类别特征"):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        
    dense_feas = [DenseFeature(name) for name in dense_features]
    sparse_feas = [SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16) for name in sparse_features]
    
    y = data['label'].values
    x = data.drop(columns=['label'])
    
    return dense_feas, sparse_feas, x, y

def main():
    # ⚠️ 确认你的文件路径
    data_path = "D:/RecStudy/torch-rechub/examples/ranking/data/criteo/criteo_100k.tsv.txt"
    
    if not os.path.exists(data_path):
        print(f"找不到文件: {data_path}")
        return

    print(">>> 1. 处理数据")
    dense_feas, sparse_feas, x, y = process_data(data_path)
    
    print(">>> 2. 生成 DataLoader")
    dg = DataGenerator(x, y)
    # 80% 训练，10% 验证，10% 测试 (10万数据测试集就有 1 万条)
    train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.8, 0.1], batch_size=2048, num_workers=0)
    
    print(">>> 3. 初始化并训练 DeepFM (1 epoch 演示)")
    model = DeepFM(deep_features=dense_feas, fm_features=sparse_feas, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
    trainer = CTRTrainer(model, optimizer_params={"lr": 1e-3, "weight_decay": 1e-5}, n_epoch=5, earlystop_patience=2, device='cpu', model_path='./')
    
    trainer.fit(train_dl, val_dl)
    
    print(">>> 4. 在测试集上预测并导出明细")
    model.eval()
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for x_dict, y_batch in tqdm(test_dl, desc="预测中"):
            x_dict = {k: v.to('cpu') for k, v in x_dict.items()}
            logits = model(x_dict)
            probs = torch.sigmoid(logits).cpu().numpy().flatten() # 转成 0~1 的概率
            y_pred_list.extend(probs)
            y_true_list.extend(y_batch.cpu().numpy().flatten())
            
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    if len(np.unique(y_true)) < 2:
        print("⚠️ 警告: 测试集只有一种类别，无法计算 AUC！")
        return

    save_dir = "D:/CodexOutput/RecHub/d15_criteo"
    os.makedirs(save_dir, exist_ok=True)
    df_pred = pd.DataFrame({"label": y_true, "score": y_pred})
    df_pred.to_csv(os.path.join(save_dir, "criteo_predictions.csv"), index=False)
    print(f"预测明细已保存至: {save_dir}/criteo_predictions.csv")
    
    print(">>> 5. 计算指标并绘制校准曲线")
    auc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    print(f"\n测试集 AUC: {auc:.4f}")
    print(f"测试集 Brier Score: {brier:.4f}")
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='完美校准')
    plt.plot(prob_pred, prob_true, marker='o', color='blue', label='DeepFM 模型')
    plt.xlabel('模型预测概率均值')
    plt.ylabel('真实点击率')
    plt.title(f'Criteo 真实数据校准曲线 (N={len(y_true)})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "criteo_calibration.png"))
    print(f"校准曲线已保存至: {save_dir}/criteo_calibration.png")

if __name__ == "__main__":
    main()