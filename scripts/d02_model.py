import sys
import torch

sys.path.append("../../")
sys.path.append(".")

from run_criteo import get_criteo_data_dict
from torch_rechub.utils.data import DataGenerator
from torch_rechub.models.ranking import DeepFM

DATA = "data/criteo/criteo_sample.csv"

print("=== 1. 准备数据 ===")
dense_feas, sparse_feas, _, _, x, y = get_criteo_data_dict(DATA)
dg = DataGenerator(x, y)
# 为了打印方便，我们把 batch_size 设为 4
train_dl, _, _ = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=4)
x_dict, y_batch = next(iter(train_dl))
print(f"y_batch 形状: {y_batch.shape}")

print("\n=== 2. 实例化 DeepFM ===")
model = DeepFM(
    deep_features=dense_feas, 
    fm_features=sparse_feas, 
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"}
)

print("\n=== 3. 手动执行 forward 并打印中间形状 ===")
# 对应 forward 的前两行：获取输入
input_deep = model.embedding(x_dict, model.deep_features, squeeze_dim=True)
input_fm = model.embedding(x_dict, model.fm_features, squeeze_dim=False)

print(f"input_deep 形状 (DNN输入): {input_deep.shape}")
print(f"input_fm 形状 (FM/Linear输入): {input_fm.shape}")

# 对应 forward 的中间三行：三个分支
y_linear = model.linear(input_fm.flatten(start_dim=1))
y_fm = model.fm(input_fm)
y_deep = model.mlp(input_deep)

print(f"y_linear 形状: {y_linear.shape}")
print(f"y_fm 形状: {y_fm.shape}")
print(f"y_deep 形状: {y_deep.shape}")

# 对应 forward 的最后两行：汇总和激活
y_sum = y_linear + y_fm + y_deep
final_pred = torch.sigmoid(y_sum.squeeze(1))

print(f"相加后的 y_sum 形状: {y_sum.shape}")
print(f"最终输出 (Sigmoid后) 形状: {final_pred.shape}")
print(f"\n最终输出的 4 个预测概率:\n{final_pred}")

# 顺便对比一下直接调用模型的结果
with torch.no_grad():
    direct_pred = model(x_dict)
print(f"\n直接调用 model(x_dict) 的形状: {direct_pred.shape}")
print(f"直接调用模型输出的概率:\n{direct_pred}")