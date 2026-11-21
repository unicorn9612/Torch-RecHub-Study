import pandas as pd
path = "data/amazon-electronics/amazon_electronics_sample.csv"
d = pd.read_csv(path)
print("shape:", d.shape)
print("columns:", d.columns.tolist())
print(d.head(8))
print("\n每用户交互数分布:")
print(d.groupby("user_id").size().describe())
print("\n用户数:", d["user_id"].nunique())
print("物品数:", d["item_id"].nunique())
print("类别数:", d["cate_id"].nunique() if "cate_id" in d.columns else "无 cate_id")