import random
import numpy as np
import torch
import pandas as pd
import os

# 1. 强制加满所有的随机种子
seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 2. 引入需要的模块
from run_amazon_electronics import get_amazon_data_dict
from torch_rechub.models.ranking import DIN
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

if __name__ == '__main__':
    dataset_path = "./data/amazon-electronics/amazon_electronics_sample.csv"
    
    # 3. 获取数据并检查词表大小
    print("=== 开始检查词表大小 ===")
    features, target_features, history_features, (train_x, train_y), (val_x, val_y), (test_x, test_y) = get_amazon_data_dict(dataset_path)
    
    for f in features + history_features:
        max_val = train_x[f.name].max()
        print(f"特征: {f.name:15s} | 词表大小 (vocab_size): {f.vocab_size:3d} | 实际最大 ID: {max_val}")
        assert f.vocab_size > max_val, f"报错！{f.name} 的 vocab_size 不足以覆盖最大 ID！"
    print("词表检查通过！\n")
    
    # 4. 自己构建 DataLoader
    dg = DataGenerator(train_x, train_y)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(
        x_val=val_x, y_val=val_y, x_test=test_x, y_test=test_y, batch_size=16
    )
    
    # 5. 构建模型和 Trainer
    model = DIN(features=features, history_features=history_features, target_features=target_features, 
                mlp_params={"dims": [256, 128]}, attention_mlp_params={"dims": [256, 128]})
    save_dir = "D:/CodexOutput/RecHub/d06_din"
    os.makedirs(save_dir, exist_ok=True)
    
    ctr_trainer = CTRTrainer(
        model,
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
        n_epoch=2,
        earlystop_patience=4,
        device='cpu',
        model_path=save_dir, # 必须用英文路径
    )
    
    # 6. 开始训练
    print("=== 开始训练 DIN ===")
    ctr_trainer.fit(train_dataloader, val_dataloader)
    
    # 7. 在测试集上预测，并保存明细 CSV
    print("=== 开始预测并保存明细 ===")
    ctr_trainer.model.eval()
    all_scores = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_dataloader:
            y_pred = ctr_trainer.model(x_batch)
            all_scores.extend(y_pred.cpu().numpy())
            
    df_res = pd.DataFrame({
        'user_id': test_x['user_id'],
        'target_item_id': test_x['target_item_id'],
        'label': test_y,
        'score': all_scores
    })
    
    df_res.to_csv("D:/CodexOutput/RecHub/d06_predictions.csv", index=False)
    print(f"预测明细已保存！共 {len(df_res)} 条")
    
    
    # ==========================================
    # 8. 终极挑战：Mask 鲁棒性测试（错位测试）
    # ==========================================
    print("\n=== 开始 Mask 错位鲁棒性测试 ===")
    
    # 构造样本 A：真实历史 [2, 3, 4] 紧挨着，后面全补 0
    # (注意：这里得防着模型抛错，所以不要用太小的假 ID，我们借用模型能认的)
    x_A = {
        "hist_item_id": torch.tensor([[10, 20, 30] + [0]*47]),
        "hist_cate_id": torch.tensor([[10, 20, 30] + [0]*47]),
        "target_item_id": torch.tensor([40]),
        "target_cate_id": torch.tensor([40]),
        "user_id": torch.tensor([1])
    }
    
    # 构造样本 B：真实历史 [10, 20, 30]，但是中间被补 0 隔开了
    x_B = {
        "hist_item_id": torch.tensor([[10, 0, 20, 0, 30] + [0]*45]),
        "hist_cate_id": torch.tensor([[10, 0, 20, 0, 30] + [0]*45]),
        "target_item_id": torch.tensor([40]),
        "target_cate_id": torch.tensor([40]),
        "user_id": torch.tensor([1])
    }
    
    with torch.no_grad():
        pred_A = ctr_trainer.model(x_A)
        pred_B = ctr_trainer.model(x_B)
        
    print(f"样本 A (历史 [10, 20, 30, 0, 0...]) 预测概率: {pred_A.item():.8f}")
    print(f"样本 B (历史 [10, 0, 20, 0, 30...]) 预测概率: {pred_B.item():.8f}")
    print(f"两者差值 (绝对值): {abs(pred_A.item() - pred_B.item()):.8f}")
    
    if abs(pred_A.item() - pred_B.item()) < 1e-6:
        print("✅ 测试通过！Mask 完美生效。补 0 无论在哪里，都被模型当成了空气，预测概率完全一致。")
    else:
        print("❌ 测试失败！Mask 没起作用，补 0 干扰了模型。")