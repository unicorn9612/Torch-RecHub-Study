import os
import numpy as np
import torch
import faiss
import torch.nn as nn

from run_ml_dssm import get_movielens_data
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator

# ==========================================
# 1. 极简排序模型 (Ranking Model)
# 模拟 DIN/DeepFM 的输入：用户ID + 目标电影ID + 召回分数 -> 打分
# ==========================================
class SimpleRankModel(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=16):
        super().__init__()
        self.user_emb = nn.Embedding(num_users + 1, embed_dim)
        self.item_emb = nn.Embedding(num_items + 1, embed_dim)
        # 输入维度：user_emb(16) + item_emb(16) + recall_score(1)
        self.mlp = nn.Sequential(
            nn.Linear(16 + 16 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids, recall_scores):
        u_vec = self.user_emb(user_ids)
        i_vec = self.item_emb(item_ids)
        # 拼接特征: [batch, 33]
        x = torch.cat([u_vec, i_vec, recall_scores.unsqueeze(1)], dim=1)
        return self.mlp(x)

def main():
    save_dir = "D:/CodexOutput/RecHub/d13_dssm_full"
    device = "cpu"
    torch.manual_seed(2022)

    print(">>> 1. 准备数据与双塔模型...")
    # 注意：x_test 是一个 dict，包含用户和电影特征，这里我们只需要 ID 来做演示
    user_features, item_features, x_train, y_train, all_item, test_user = get_movielens_data(
        data_path="./data/ml-1m/ml-1m.csv", load_cache=True
    )
    
    # 提取测试集用户 ID 和真实历史行为
    # x_test 里的 user_id 和 hist_movie_id 已经是编码后的 id (1~N)
    test_user_ids = test_user['user_id']
    test_histories = test_user['hist_movie_id']# 形状 [num_samples, 50]

    dg = MatchDataGenerator(x=x_train, y=y_train)
    model = DSSM(user_features, item_features, temperature=0.02,
                 user_params={"dims": [256, 128, 64], "activation": 'prelu'},
                 item_params={"dims": [256, 128, 64], "activation": 'prelu'})
    trainer = MatchTrainer(model, mode=0, in_batch_neg=False, n_epoch=1, device=device, model_path=save_dir)
    train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=4096, num_workers=0)

    print(">>> 2. 提取双塔向量并建 Faiss 索引...")
    U = trainer.inference_embedding(model=model, mode="user", data_loader=test_dl, model_path=save_dir).cpu().numpy().astype(np.float32)
    V = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=save_dir).cpu().numpy().astype(np.float32)
    
    index = faiss.IndexFlatIP(V.shape[1])
    index.add(V)

    print(">>> 3. 初始化排序模型...")
    num_users = max(test_user_ids) + 10
    num_items = V.shape[0]
    rank_model = SimpleRankModel(num_users, num_items)
    
    # 【重点】这里我们直接拿第一个用户做 Demo，跑通完整链路
    demo_idx = 0
    user_id = test_user_ids[demo_idx]
    history = test_histories[demo_idx]
    
    print(f"\n=== 开始模拟在线推荐链路: User {user_id} ===")
    print(f"该用户历史看过的电影 ID: {history[history > 0]}")

    # --- 阶段 A: 召回 ---
    print("\n[阶段 A] Faiss 召回 Top-50 候选电影...")
    query_vec = U[demo_idx].reshape(1, -1)
    D, I = index.search(query_vec, 50) # 召回 50 个
    recall_items = I[0] + 1 # Faiss 返回的是 0~N-1 的索引，我们的 movie_id 编码是从 1 开始的
    recall_scores = D[0]
    print(f"召回候选集 (未过滤): {recall_items[:10]} ... (共 50 个)")

    # --- 阶段 B: 过滤 ---
    print("\n[阶段 B] 过滤掉用户已经看过的历史物品...")
    filtered_items = []
    filtered_scores = []
    for item, score in zip(recall_items, recall_scores):
        if item not in history:
            filtered_items.append(item)
            filtered_scores.append(score)
    
    filtered_items = np.array(filtered_items)
    filtered_scores = np.array(filtered_scores)
    print(f"过滤后剩余候选数: {len(filtered_items)}")
    print(f"过滤后候选集: {filtered_items[:10]} ...")

    # --- 阶段 C: 排序 ---
    print("\n[阶段 C] 拼接特征，输入排序模型重排...")
    # 构造 Batch 数据
    batch_user = torch.tensor([user_id] * len(filtered_items), dtype=torch.long)
    batch_item = torch.tensor(filtered_items, dtype=torch.long)
    batch_score = torch.tensor(filtered_scores, dtype=torch.float32)

    rank_model.eval()
    with torch.no_grad():
        # 打分！
        rank_preds = rank_model(batch_user, batch_item, batch_score).squeeze(-1).numpy()

    # 排序输出
    sorted_indices = np.argsort(-rank_preds) # 降序
    final_top_10 = filtered_items[sorted_indices][:10]
    final_scores = rank_preds[sorted_indices][:10]

    print("\n=== 最终推荐给用户的 Top-10 电影 ===")
    for i, (item, score) in enumerate(zip(final_top_10, final_scores)):
        print(f"Rank {i+1}: Movie ID {item}, 排序分 {score:.4f}")

if __name__ == "__main__":
    main()