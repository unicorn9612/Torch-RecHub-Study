import os
import time
import numpy as np
import torch
import faiss

# 从你原来的脚本里复用数据和模型结构
from run_ml_dssm import get_movielens_data
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator

def calculate_recall(I_true, I_pred):
    """计算精确 top-k 和近似 top-k 的交集比例"""
    hits = 0
    for i in range(I_true.shape[0]):
        hits += len(set(I_true[i]) & set(I_pred[i]))
    return hits / (I_true.shape[0] * I_true.shape[1])

def main():
    save_dir = "D:/CodexOutput/RecHub/d13_dssm_full"
    device = "cpu"
    torch.manual_seed(2022)

    print(">>> 1. 准备数据与模型...")
    user_features, item_features, x_train, y_train, all_item, test_user = get_movielens_data(
        data_path="./data/ml-1m/ml-1m.csv", load_cache=True
    )
    dg = MatchDataGenerator(x=x_train, y=y_train)
    
    model = DSSM(
        user_features, item_features, temperature=0.02,
        user_params={"dims": [256, 128, 64], "activation": 'prelu'},
        item_params={"dims": [256, 128, 64], "activation": 'prelu'}
    )
    
    # 直接指定 model_path，它会自动去加载 save_dir/dssm_model.pth
    trainer = MatchTrainer(model, mode=0, in_batch_neg=False, n_epoch=1, device=device, model_path=save_dir)
    train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=4096, num_workers=0)

    print(">>> 2. 推理 embedding...")
    # 走一遍前向传播，抽出向量
    user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=test_dl, model_path=save_dir)
    item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=save_dir)

    # 转成 faiss 只认的 float32 numpy 格式
    U = user_embedding.cpu().numpy().astype(np.float32)
    V = item_embedding.cpu().numpy().astype(np.float32)
    print(f"User 向量形状: {U.shape}, Item 向量形状: {V.shape}")
     # --- 新增：检查向量的范数分布 ---
    import numpy.linalg as LA
    print(f"User 向量范数 | 均值: {LA.norm(U, axis=1).mean():.4f} | 标准差: {LA.norm(U, axis=1).std():.4f}")
    print(f"Item 向量范数 | 均值: {LA.norm(V, axis=1).mean():.4f} | 标准差: {LA.norm(V, axis=1).std():.4f}")

    print("\n>>> 3. 构建索引与对比实验 (K=100)")
    K = 100
    
    # 实验 A：Flat 暴力 + 内积 IP (不归一化，与 DSSM 训练目标一致)
    index_ip = faiss.IndexFlatIP(V.shape[1])
    index_ip.add(V)
    t0 = time.time()
    D_ip, I_ip = index_ip.search(U, K)
    time_ip = (time.time() - t0) * 1000
    print(f"[A] Flat + IP(不归一化)  耗时: {time_ip:.2f} ms")

    # 实验 B：Flat 暴力 + L2 (不归一化)
    index_l2 = faiss.IndexFlatL2(V.shape[1])
    index_l2.add(V)
    D_l2, I_l2 = index_l2.search(U, K)
    
    # 实验 C：归一化后的 Flat + IP (等价于 Cosine 相似度)
    # faiss.normalize_L2 原地操作，把向量变成单位长度
    U_norm = U.copy()
    V_norm = V.copy()
    faiss.normalize_L2(U_norm)
    faiss.normalize_L2(V_norm)
    index_ip_norm = faiss.IndexFlatIP(V.shape[1])
    index_ip_norm.add(V_norm)
    D_norm, I_norm = index_ip_norm.search(U_norm, K)

    # 实验 D：IVF 近似索引 (用训练好的 IP 索引作 ground truth)
    # nlist=100 把电影聚成 100 类, nprobe=8 查最近的 8 个类
    quantizer = faiss.IndexFlatIP(V.shape[1])
    index_ivf = faiss.IndexIVFFlat(quantizer, V.shape[1], 100, faiss.METRIC_INNER_PRODUCT)
    index_ivf.train(V)
    index_ivf.add(V)
    index_ivf.nprobe = 8
    t0 = time.time()
    D_ivf, I_ivf = index_ivf.search(U, K)
    time_ivf = (time.time() - t0) * 1000
    print(f"[D] IVF + IP(nprobe=8)   耗时: {time_ivf:.2f} ms")

    print("\n>>> 4. 计算 Recall@10 对比")
    # 以 A(Flat + IP 不归一化) 作为绝对基准
    recall_B = calculate_recall(I_ip, I_l2)
    recall_C = calculate_recall(I_ip, I_norm)
    recall_D = calculate_recall(I_ip, I_ivf)

    print(f"基准 [A] Flat+IP(不归一化) 的 Recall: 1.0000")
    print(f"对比 [B] Flat+L2(不归一化) 的 Recall: {recall_B:.4f}  <-- 探究训练/服务不一致的代价")
    print(f"对比 [C] Flat+IP(归一化)   的 Recall: {recall_C:.4f}  <-- 探究 Normalize 带来的变化")
    print(f"对比 [D] IVF+IP(近似搜索)  的 Recall: {recall_D:.4f}  <-- 探究近似索引的召回损失")

if __name__ == "__main__":
    main()