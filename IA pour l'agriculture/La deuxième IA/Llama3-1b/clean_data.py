import json
import os

# 读取 JSON 文件
input_file = "data/agriculture_dataset1.json"
output_file = "./data/agriculture_dataset_cleaned.json"

# 加载 JSON 数据
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# 删除 "instruction" 键
for entry in data:
    if "instruction" in entry:
        del entry["instruction"]

# 重新保存精简后的 JSON 数据
with open(output_file, "w", encoding="utf-8") as f:
    for entry in data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"数据集已处理并保存至 {output_file}")