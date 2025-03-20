import os
import torch
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from evaluate import load
import evaluate

# ✅ 彻底禁用 MPS，防止 PyTorch 误用
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class ModelEvaluator:
    def __init__(self, model_path, test_dataset=None):
        """
        ✅ 初始化评估器
        :param model_path: 已微调的模型路径
        :param test_dataset: 用于测试的测试集（可选）
        """
        self.device = torch.device("cpu")  # ✅ 强制使用 CPU
        print(f"✅ 设备选择: {self.device}")

        # ✅ 加载模型（确保 float32，不使用 float16）
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # ✅ 防止 MPS 设备崩溃
            device_map={"": "cpu"}  # ✅ 确保在 CPU 运行
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)

        # ✅ 替换 `load_metric()` 为 `evaluate.load()`
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")

        self.test_dataset = test_dataset  # ✅ 测试数据集

    def calculate_perplexity(self, text):
        """ ✅ 计算 Perplexity（PPL），确保数据在 CPU """
        encodings = self.tokenizer(text, return_tensors="pt").to("cpu")  # ✅ 强制到 CPU

        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings["input_ids"])
        loss = outputs.loss.item()
        return math.exp(loss)

    def evaluate_generation(self, reference, generated):
        """ ✅ 计算 BLEU 和 ROUGE 分数，修正输入格式 """
        bleu_score = self.bleu.compute(
            predictions=[generated],  # ✅ 直接传入字符串列表
            references=[[reference]]  # ✅ 参考答案是字符串列表的列表
        )
        rouge_score = self.rouge.compute(
            predictions=[generated],
            references=[reference]
        )

        return {
            "BLEU": bleu_score["bleu"],
            "ROUGE": rouge_score
        }

    def test_on_dataset(self):
        """ ✅ 在测试集上评估模型 """
        if not self.test_dataset:
            print("⚠️ 没有提供测试集，跳过测试集评估")
            return None

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for sample in tqdm(self.test_dataset, desc="Evaluating Test Set"):
                text = sample["text"]
                encodings = self.tokenizer(text, return_tensors="pt").to("cpu")  # ✅ 确保数据在 CPU
                outputs = self.model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss.item()

                total_loss += loss
                num_batches += 1

        avg_loss = total_loss / num_batches
        ppl = math.exp(avg_loss)  # 计算平均 PPL

        return {
            "Test Loss": avg_loss,
            "Test Perplexity (PPL)": ppl
        }

    def run_evaluation(self, sample_text, reference_text=None, generated_text=None):
        """
        ✅ 运行所有评估方法
        :param sample_text: 用于计算 PPL 的示例文本
        :param reference_text: 参考答案（如果评估文本生成）
        :param generated_text: 模型生成的文本
        :return: 评估结果字典
        """
        results = {}

        # 1️⃣ 计算 Perplexity（PPL）
        ppl = self.calculate_perplexity(sample_text)
        results["Perplexity"] = ppl
        print(f"✅ Perplexity（困惑度）: {ppl:.4f}")

        # 2️⃣ 计算 BLEU / ROUGE（如果提供了生成文本）
        if reference_text and generated_text:
            gen_scores = self.evaluate_generation(reference_text, generated_text)
            results["BLEU"] = gen_scores["BLEU"]
            results["ROUGE"] = gen_scores["ROUGE"]
            print(f"✅ BLEU Score: {gen_scores['BLEU']:.4f}")
            print(f"✅ ROUGE Score: {gen_scores['ROUGE']}")

        # 3️⃣ 计算测试集上的损失和 PPL
        test_results = self.test_on_dataset()
        if test_results:
            results.update(test_results)
            print(f"✅ 测试集平均损失: {test_results['Test Loss']:.4f}")
            print(f"✅ 测试集 Perplexity（PPL）: {test_results['Test Perplexity (PPL)']:.4f}")

        return results