import os
import torch
import pandas as pd
import faiss
import numpy as np
from model_evaluator import ModelEvaluator
from langchain_community.embeddings import HuggingFaceEmbeddings
import math
from bert_score import score
from tqdm import tqdm
import time

def main():
    MODEL_PATH = "./model/llama3_lora_colab"
    DATASET_PATH = "./data/agriculture_qa.csv"
    df = pd.read_csv(DATASET_PATH)

    if 'question' not in df.columns or 'answers' not in df.columns:
        raise ValueError(f"⚠️ 数据集必须包含 'question' 和 'answers' 列！")

    df.rename(columns={'question': 'text'}, inplace=True)
    test_size = min(len(df), 100)
    test_samples = df.sample(n=test_size, random_state=42)

    index_path = "./database/faiss_db/index.faiss"
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"⚠️ FAISS 索引文件未找到！")

    index = faiss.read_index(index_path)
    faiss.omp_set_num_threads(1)

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def retrieve_faiss(query_text, top_k=5):
        # print(f"🔍 Querying FAISS: {query_text}")
        query_vector = embedder.embed_query(query_text)
        query_vector = np.array(query_vector).astype("float32").reshape(1, -1)
        retrieved_texts = []

        try:
            _, I = index.search(query_vector, k=top_k)
            for idx in I[0]:
                if 0 <= idx < len(df):
                    text = df.iloc[idx]["text"]
                    if query_text.lower() in text.lower():
                        retrieved_texts.append(text.split("\nAnswer: ")[-1].strip())

        except Exception as e:
            print(f"⚠️ FAISS 查询失败: {e}")
            return "No relevant information found."

        return " ".join(retrieved_texts[:1]) if retrieved_texts else "No relevant information found."

    test_dataset = []
    for _, row in test_samples.iterrows():
        # if len(test_dataset) >= 50: break
        retrieved_text = retrieve_faiss(row["text"])
        if retrieved_text:
            test_dataset.append({"text": retrieved_text, "answers": row["answers"]})

    print(f"✅ 已从 {DATASET_PATH} 通过 FAISS 召回 {len(test_dataset)} 条测试数据")

    evaluator = ModelEvaluator(MODEL_PATH, test_dataset=test_dataset)

    def generate_response(prompt):
        """ ✅ 修复 `attention_mask` 错误 """
        inputs = evaluator.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="longest"
        ).to(evaluator.device)

        output = evaluator.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            pad_token_id=evaluator.tokenizer.eos_token_id
        )

        return evaluator.tokenizer.decode(output[0], skip_special_tokens=True)

    def calculate_perplexity(text):
        """ ✅ 计算 Perplexity（PPL）"""
        inputs = evaluator.tokenizer(text, return_tensors="pt").to(evaluator.device)

        with torch.no_grad():
            outputs = evaluator.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        return math.exp(loss)  # PPL = e^(loss)

    references = [sample["answers"] for sample in test_dataset]
    predictions = [generate_response(sample["text"]) for sample in tqdm(test_dataset, desc="Generating predictions")]

    import time
    start = time.time()
    print("📊 正在计算 BERTScore，请稍等（首次运行可能较慢）...")
    # ✅ 计算 BERTScore
    bert_score = score(predictions, references, lang="en", model_type="bert-base-uncased")[2].mean().item()
    print(f"✅ BERTScore 计算完成，用时 {time.time() - start:.2f} 秒")

    # ✅ 计算 Perplexity（PPL）
    perplexities = [calculate_perplexity(pred) for pred in tqdm(predictions, desc="Calculating Perplexity")]
    avg_ppl = sum(perplexities) / len(perplexities)

    print(f"\n🚀 Final evaluation results:")
    print(f"✅ BERTScore: {bert_score:.4f}")
    print(f"✅ Perplexity (PPL): {avg_ppl:.4f}")

if __name__ == "__main__":
    main()