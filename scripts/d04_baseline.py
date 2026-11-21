import sys
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

# 确保能导入项目源码
sys.path.append("../../")
sys.path.append(".")

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator
from torch_rechub.models.ranking import DeepFM, WideDeep
from torch_rechub.trainers import CTRTrainer

# 固定随机种子
torch.manual_seed(2022)
np.random.seed(2022)

DATA_PATH = "data/criteo/criteo_100k.tsv.txt"
OUT_DIR = "D:/Codex输出/Torch-RecHub学习计划/实验/d04"
os.makedirs(OUT_DIR, exist_ok=True)

print("=== 1. 读取与清洗数据 ===")
cols = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
# 读取数据，注意 sep='\t' 和 header=None
data = pd.read_csv(DATA_PATH, sep='\t', header=None, names=cols)
print(f"原始行数: {len(data)}")

# 去重
data = data.drop_duplicates()
print(f"去重后行数: {len(data)}")

# 截取前 5 万条，节省 CPU 时间
data = data.head(50000).copy()

dense_features = [f'I{i}' for i in range(1, 14)]
sparse_features = [f'C{i}' for i in range(1, 27)]

# 填补缺失值
data[sparse_features] = data[sparse_features].fillna('0')
data[dense_features] = data[dense_features].fillna(0)

print("\n=== 2. 严谨的防泄漏划分 ===")
# 80% 训练+验证，20% 测试
train_val, test = train_test_split(data, test_size=0.2, random_state=2022, stratify=data['label'])
# 从剩下的里面再分 20% 作为验证集（即总量的 16%）
train, val = train_test_split(train_val, test_size=0.2, random_state=2022, stratify=train_val['label'])
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

print("\n=== 3. 防泄漏的特征工程 (向量化加速版) ===")
# 3.1 稠密特征离散化（取整转字符串作为新的类别特征）
for feat in dense_features:
    train[feat + "_cat"] = train[feat].fillna(0).astype(int).astype(str)
    val[feat + "_cat"] = val[feat].fillna(0).astype(int).astype(str)
    test[feat + "_cat"] = test[feat].fillna(0).astype(int).astype(str)
    sparse_features.append(feat + "_cat")

# 3.2 连续特征缩放（只在 Train 上 fit）
scaler = MinMaxScaler()
train[dense_features] = scaler.fit_transform(train[dense_features])
val[dense_features] = scaler.transform(val[dense_features])
test[dense_features] = scaler.transform(test[dense_features])

# 3.3 类别特征编码（防泄漏 + O(1) 字典映射）
for feat in sparse_features:
    lbe = LabelEncoder()
    # 只在 train 上 fit_transform
    train[feat] = lbe.fit_transform(train[feat])
    
    # 建立 {类别: ID} 的字典，速度极快
    le_dict = dict(zip(lbe.classes_, range(len(lbe.classes_))))
    
    # 验证集和测试集直接用字典 map，没见过的填 0 (OOV处理)
    val[feat] = val[feat].map(le_dict).fillna(0).astype(int)
    test[feat] = test[feat].map(le_dict).fillna(0).astype(int)

# 构建特征对象
dense_feas = [DenseFeature(name) for name in dense_features]
# 词表大小必须严格等于训练集中的类别数
sparse_feas = [SparseFeature(name, vocab_size=len(train[name].unique()), embed_dim=16) for name in sparse_features]

# 拆出 x 和 y
# 拆出 x 和 y，并重置索引 (极为重要，否则 DataLoader 会 KeyError)
train_x, train_y = train.drop('label', axis=1).reset_index(drop=True), train['label'].reset_index(drop=True)
val_x, val_y = val.drop('label', axis=1).reset_index(drop=True), val['label'].reset_index(drop=True)
test_x, test_y = test.drop('label', axis=1).reset_index(drop=True), test['label'].reset_index(drop=True)

# 使用 DataGenerator 直接传入 val 和 test
dg = DataGenerator(train_x, train_y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=val_x, y_val=val_y, 
    x_test=test_x, y_test=test_y, 
    batch_size=1024, num_workers=0
)

# 封装训练与评估函数
def run_experiment(model_name, model, lr=1e-3):
    print(f"\n================ 运行模型: {model_name}, 学习率: {lr} ================")
    
    trainer = CTRTrainer(
        model, 
        # 这里也一定要把原来的 1e-3 改成 lr 变量！
        optimizer_params={"lr": lr, "weight_decay": 1e-4}, 
        n_epoch=3, 
        device="cpu",
        model_path=OUT_DIR
    )

    # 训练
    trainer.fit(train_dl, val_dl)
    # 新增：在验证集上评估
    val_auc = trainer.evaluate(trainer.model, val_dl)
    print(f"【{model_name} (lr={lr})】 验证集 AUC: {val_auc:.4f}")
    # 测试集预测
    y_pred = trainer.predict(trainer.model, test_dl)
    y_pred = np.array(y_pred).flatten()
    
    # 评估指标
    auc = roc_auc_score(test_y, y_pred)
    ll = log_loss(test_y, y_pred)
    print(f"\n【{model_name} 最终结果】 Test AUC: {auc:.4f} | LogLoss: {ll:.4f}")
    return auc, ll

# ---------------- 1. 跑 WideDeep 基线 ----------------
#model_wd = WideDeep(wide_features=dense_feas, deep_features=sparse_feas, mlp_params={"dims": [256, 128], "dropout": 0.2})
#run_experiment("WideDeep (Baseline)", model_wd)

# ---------------- 2. 跑 DeepFM 进阶模型 ----------------
#model_dfm = DeepFM(deep_features=dense_feas, fm_features=sparse_feas, mlp_params={"dims": [256, 128], "dropout": 0.2})
#run_experiment("DeepFM", model_dfm)

#print("\n全部实验完成！")
print("\n>>> 实验 1：学习率 1e-3 <<<")
model_dfm_1 = DeepFM(deep_features=dense_feas, fm_features=sparse_feas, mlp_params={"dims": [256, 128], "dropout": 0.2})
run_experiment("DeepFM", model_dfm_1, lr=1e-3)

print("\n>>> 实验 2：学习率 5e-4 <<<")
model_dfm_2 = DeepFM(deep_features=dense_feas, fm_features=sparse_feas, mlp_params={"dims": [256, 128], "dropout": 0.2})
run_experiment("DeepFM", model_dfm_2, lr=5e-4)