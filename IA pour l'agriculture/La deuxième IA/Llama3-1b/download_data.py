from datasets import load_dataset

# 指定数据集名称
dataset_name = "ShuklaShreyansh/Agriculture-QA"

# 加载数据集
dataset = load_dataset(dataset_name)

# 创建保存目录
import os
save_path = "./data"
os.makedirs(save_path, exist_ok=True)

# 保存数据集为 JSON 格式
dataset["train"].to_json(os.path.join(save_path, "agriculture_dataset2.json"))

print("数据集已成功下载并保存到 ./data/agriculture_dataset2.json")