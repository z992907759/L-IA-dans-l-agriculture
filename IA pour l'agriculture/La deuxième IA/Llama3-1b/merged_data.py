import json
import pandas as pd

# 文件路径
json_file_path = "./data/agriculture_dataset_merged.json"
csv_file_path = "./data/agriculture_qa.csv"
output_csv_path = "./data/agriculture_qa_final.csv"

# 读取 JSON 数据
json_data = []
with open(json_file_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        json_data.append({
            "question": entry["input"].strip(),  # 仅去除首尾空格
            "answers": entry["response"].strip()
        })

# 读取 CSV 数据
csv_data = pd.read_csv(csv_file_path)

# 统一列名
csv_data.columns = ["question", "answers"]

# 仅去除前后空格（不转换大小写）
csv_data["question"] = csv_data["question"].astype(str).str.strip()
csv_data["answers"] = csv_data["answers"].astype(str).str.strip()

# 转换 JSON 数据为 DataFrame
json_df = pd.DataFrame(json_data)

# 打印数据集大小（检查是否正确读取）
print(f"CSV 数据集大小: {csv_data.shape[0]}")
print(f"JSON 数据集大小: {json_df.shape[0]}")

# 合并数据集
merged_df = pd.concat([csv_data, json_df], ignore_index=True)

# 合并后大小（去重前）
print(f"合并后数据集大小（去重前）: {merged_df.shape[0]}")

# 解决去重问题：group by question，把 answers 合并成列表
merged_df = merged_df.groupby("question", as_index=False).agg({"answers": lambda x: "; ".join(set(x))})

# 合并后大小（去重后）
print(f"去重后数据集大小: {merged_df.shape[0]}")

# 保存为 CSV
merged_df.to_csv(output_csv_path, index=False, encoding="utf-8")

print(f"数据集已成功合并并保存至 {output_csv_path}")