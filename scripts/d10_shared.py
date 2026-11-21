import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

# 引入官方脚本的数据加载和模型
sys.path.append("../../")
from run_census import get_census_data_dict
from torch_rechub.models.multi_task import SharedBottom
from torch_rechub.trainers import MTLTrainer
from torch_rechub.utils.data import DataGenerator

# 1. 固定种子
seed = 2022
torch.manual_seed(seed)
np.random.seed(seed)

if __name__ == '__main__':
    # 2. 加载数据 (注意：run_census 里的 model_name 决定了标签列顺序)
    features, x_train, y_train, x_val, y_val, x_test, y_test = get_census_data_dict("SharedBottom")
    
    dg = DataGenerator(x_train, y_train)
    train_dl, val_dl, test_dl = dg.generate_dataloader(x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, batch_size=16)

    # 3. 初始化模型 (完全复制 run_census.py 里的参数)
    task_types = ["classification", "classification"]
    model = SharedBottom(
        features, 
        task_types, 
        bottom_params={"dims": [117]}, 
        tower_params_list=[{"dims": [8]}, {"dims": [8]}]
    )

    # 4. 训练
    save_dir = "D:/CodexOutput/RecHub/d10_shared"
    os.makedirs(save_dir, exist_ok=True)
    
    trainer = MTLTrainer(
        model, 
        task_types=task_types, 
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-4}, 
        n_epoch=2, 
        earlystop_patience=30, 
        device='cpu', 
        model_path=save_dir
    )
    print("=== 开始训练 SharedBottom ===")
    trainer.fit(train_dl, val_dl)

       # 5. 预测与评估 (手动写循环，绕过库的 Bug)
    print("=== 开始测试集评估 ===")
    trainer.model.eval()
    all_preds = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_dl:  # 正确解包
            x_batch = {k: v.to('cpu') for k, v in x_batch.items()}
            y_preds = trainer.model(x_batch)
            all_preds.extend(y_preds.cpu().numpy())
            
    preds = np.array(all_preds)
    y_test_arr = np.array(y_test)

    # 计算两个任务的 AUC 和 LogLoss
    from sklearn.metrics import roc_auc_score, log_loss
    auc_0 = roc_auc_score(y_test_arr[:, 0], preds[:, 0])
    auc_1 = roc_auc_score(y_test_arr[:, 1], preds[:, 1])
    
    ll_0 = log_loss(y_test_arr[:, 0], preds[:, 0])
    ll_1 = log_loss(y_test_arr[:, 1], preds[:, 1])

    print(f"\n【SharedBottom 测试集结果】")
    print(f"Task 0 (收入)    : AUC = {auc_0:.4f} | LogLoss = {ll_0:.4f}")
    print(f"Task 1 (婚姻状态): AUC = {auc_1:.4f} | LogLoss = {ll_1:.4f}")

    # 6. 保存预测明细 CSV
    import pandas as pd
    df_res = pd.DataFrame({
        'task0_label': y_test_arr[:, 0],
        'task0_pred': preds[:, 0],
        'task1_label': y_test_arr[:, 1],
        'task1_pred': preds[:, 1]
    })
    csv_path = "D:/CodexOutput/RecHub/d10_shared_predictions.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n预测明细已保存至: {csv_path}")

    # 7. 打印参数规模
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SharedBottom 总参数量: {total_params}")