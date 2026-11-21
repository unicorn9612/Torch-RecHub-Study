import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

# 导入 torch-rechub 的相关模块
from run_criteo import get_criteo_data_dict
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

def main():
    # ⚠️【注意修改这里】：换成你真实 Criteo 数据的路径
    # 如果你只有 sample，就用 "./data/criteo/criteo_sample.csv"
    data_path = "D:/RecStudy/torch-rechub/examples/ranking/data/criteo/criteo_100k.tsv.txt"  
    
    print(f">>> 1. 加载数据: {data_path}")
    # get_criteo_data_dict 内部会自动处理特征，返回 dnn_feas, linear_feas, train, val, test
    dnn_feas, linear_feas, train, val, test = get_criteo_data_dict(data_path)
    
    print(">>> 2. 构建 DataLoader...")
    dg = DataGenerator(train, val, test)
    # batch_size 根据你的数据量调整，10万行可以设 2048 或 4096
    train_dl, val_dl, test_dl = dg.generate_dataloader(batch_size=256, num_workers=0)
    
    print(">>> 3. 初始化 DeepFM 并快速训练 1 个 Epoch...")
    model = DeepFM(deep_features=dnn_feas, fm_features=linear_feas, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
    
    # 为了演示校准，我们只训 1 个 epoch。如果数据量大，可以调大 epoch
    trainer = CTRTrainer(model, optimizer_params={"lr": 1e-3, "weight_decay": 1e-5}, n_epoch=1, earlystop_patience=2, device='cpu', model_path='./')
    trainer.fit(train_dl, val_dl)
    
    print(">>> 4. 在测试集上生成预测概率...")
    # 注意：trainer.predict 通常返回预测概率。我们还需要获取对应的 label
    # CTRTrainer.predict 返回的是 (y_pred_prob, y_true) 或类似结构，具体看源码
    # 这里我们用最稳妥的方法，遍历 test_dl 手动获取
    model.eval()
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for x_dict, y in test_dl:
            # 将 x_dict 里的数据搬到 cpu
            x_dict = {k: v.to('cpu') for k, v in x_dict.items()}
            logits = model(x_dict)
            probs = torch.sigmoid(logits).cpu().numpy().flatten() # 确保转成概率
            y_pred_list.extend(probs)
            y_true_list.extend(y.cpu().numpy().flatten())
            
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    print(f"测试集样本数: {len(y_true)}")
    
    if len(np.unique(y_true)) < 2:
        print("⚠️ 警告: 测试集只有一种类别 (全 0 或全 1)，无法计算 AUC 和校准曲线！")
        return

    # 保存预测结果到 D 盘
    save_dir = "D:/CodexOutput/RecHub/d15_criteo"
    os.makedirs(save_dir, exist_ok=True)
    df_pred = pd.DataFrame({"label": y_true, "score": y_pred})
    df_pred.to_csv(os.path.join(save_dir, "criteo_predictions.csv"), index=False)
    print(f"预测明细已保存至: {save_dir}/criteo_predictions.csv")

    print(">>> 5. 计算指标并绘制校准曲线...")
    auc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    print(f"真实数据 AUC: {auc:.4f}")
    print(f"真实数据 Brier Score: {brier:.4f}")

    # 画图
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