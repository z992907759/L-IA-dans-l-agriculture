import re  # ✅ 需要手动导入正则表达式库
import os
import torch
import glob
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time  # ✅ 需要导入 `time` 以计算生成时间


class LoRAModel:
    def __init__(self, base_model_name="meta-llama/Llama-3.2-1B-Instruct",
                 lora_model_path="./model/llama3_lora_colab",
                 faiss_index_path="./database/faiss_db"):
        """
        ✅ Encapsulation class for LoRA fine-tuning, supporting FAISS database queries| LoRA 微调模型的封装类，支持 FAISS 先查询数据库
        ✅ Generating answers by combining FAISS database with LLaMA-3 | 结合 FAISS 数据库 + LLaMA-3 生成答案
        ✅ Text-to-speech synthesis (gTTS) |语音合成 (gTTS)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.faiss_index_path = faiss_index_path

        # ✅  Ensure that the audio storage folder exists. 确保音频存储文件夹存在
        self.audio_folder = "audio"
        os.makedirs(self.audio_folder, exist_ok=True)
        self.clear_audio_folder()

        # ✅  Load the model and FAISS.| 加载模型和 FAISS
        self.load_model()
        self.load_faiss_index()

    def clear_audio_folder(self):
        """✅ 启动时清空 `./audio/` 目录"""
        audio_files = glob.glob(os.path.join(self.audio_folder, "*.mp3"))
        for file in audio_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"⚠️ 无法删除 {file}: {e}")  # 删除失败，打印错误信息
        print("🗑️The `./audio/` directory has been cleared! | `./audio/` 目录已清空！")  # ✅ 提示清理完成

    def load_model(self):
        """✅“Load LLaMA-3 + LoRA.| 加载 LLaMA-3 + LoRA"""
        print("🔄Loading the LLaMA-3 base model…  | 正在加载 LLaMA-3 基础模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        ).to(self.device)

        print("🔄 Loading the LoRA adapter layers… | 加载 LoRA 适配层...")
        self.lora_model = PeftModel.from_pretrained(self.base_model, self.lora_model_path)
        self.model = self.lora_model.merge_and_unload().to(self.device)

        # ✅ 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✅LLaMA-3 + LoRA loaded successfully! | LLaMA-3 + LoRA 加载完成！")

    def load_faiss_index(self):
        """✅Load the FAISS vector database.| 加载 FAISS 向量数据库"""
        print("🔍 Loading the FAISS index… | 正在加载 FAISS 索引...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": self.device}
        )

        self.faiss_store = FAISS.load_local(
            self.faiss_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        print("✅FAISS index loaded successfully! | FAISS 索引加载完成！")

    def search_faiss(self, query, k=3):
        """✅Retrieve the top k most relevant documents from the FAISS database. | 在 FAISS 数据库中查找最相关的 k 个文档"""
        print(f"🔍 Serach FAISS: {query}")
        results = self.faiss_store.similarity_search(query, k=k)

        if results:
            return [doc.page_content for doc in results]
        return []

    def generate_response(self, prompt):
        """✅ 结合 FAISS + LLaMA-3 生成完整答案，并计算 Token 速率"""

        # 1️⃣ **查询 FAISS**
        faiss_results = self.search_faiss(prompt, k=3)

        # 2️⃣ **清理 FAISS 结果**
        if faiss_results:
            faiss_cleaned = []
            for doc in faiss_results:
                cleaned_text = re.sub(r"\(.*?et al\..*?\)", "", doc)  # ✅ 删除引用
                cleaned_text = re.sub(r"(\d+\.\s[A-Za-z ]+)", "", cleaned_text)  # ✅ 删除章节标题
                cleaned_text = cleaned_text.strip()
                if cleaned_text:
                    faiss_cleaned.append(cleaned_text)

            faiss_summary = " ".join(faiss_cleaned)[:500]  # ✅ 限制 FAISS 片段长度
        else:
            faiss_summary = "No relevant knowledge found in FAISS."

        # 3️⃣ **优化 Prompt**
        full_prompt = f"""You are an expert in agriculture. Provide a **complete, structured answer** to the user's question.
        Use FAISS knowledge and your expertise to answer in a **detailed yet concise way**.
        Do NOT repeat FAISS text verbatim. Instead, **extract useful insights and explain them naturally**.
        Avoid unnecessary information and ensure the answer flows well.

        ### FAISS Knowledge:
        {faiss_summary}

        ### User Question:
        {prompt}

        ### Answer (Provide all useful details in a well-structured manner, at least 200 words):"""

        print(f"📚 Length of the generated prompt: {len(full_prompt)}")
        print(f"📚 Generated prompt preview:\n{full_prompt[:500]}...")

        # 4️⃣ **让 LLaMA-3 生成答案**
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True).to(self.device)

        # ✅ **记录开始时间**
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1024,
                min_length=200,  # ✅ 确保至少生成 200 个 Token，避免提前结束
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # ✅ **记录结束时间**
        end_time = time.time()

        # ✅ **计算生成时间**
        generation_time = end_time - start_time  # 计算总时间（秒）

        # ✅ **计算生成 Token 数**
        num_tokens = outputs.shape[1]  # 获取生成的 Token 数量

        # ✅ **计算每秒 Token 速率**
        tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0

        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # 5️⃣ **删除模型可能复述的 Prompt**
        response_text = response_text.replace(full_prompt, "").strip()
        response_text = response_text.split("### Answer:")[-1].strip()

        # ✅ **打印生成 Token 速率**
        print(f"🚀 Generation completed!")
        print(f"✅ Number of tokens generated: {num_tokens}")
        print(f"✅ Generated at: {generation_time:.2f} 秒")
        print(f"✅ Tokens generated per second: {tokens_per_second:.2f} tokens/sec")

        # 6️⃣ **确保 `./audio/` 目录存在**
        os.makedirs("./audio", exist_ok=True)

        # 7️⃣ **生成音频文件名**
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_file = f"./audio/{timestamp}.mp3"

        # 8️⃣ **生成 gTTS 语音**
        tts = gTTS(response_text, lang="en")
        tts.save(audio_file)
        print(f"🔊 The speech has been generated and saved in: {audio_file}")

        return response_text, audio_file  # ✅ 返回回答和音频路径

    def chat(self):

        # 1️⃣ **确保 `./audio/` 目录存在**
        os.makedirs("./audio", exist_ok=True)

        # 3️⃣ **进入对话模式**
        print("\n🤖 Interactive mode started! Type ‘exit’ or ‘quit' to quit.\n") # ✅ 提示进入对话模式

        while True:
            user_input = input("🟢 You: ").strip()
            if user_input.lower() == "exit" or user_input.lower() == "quit":
                print("👋 Good Bye！")
                break

            response, audio_file = self.generate_response(user_input)
            print(f"\n🤖 AI: {response}\n")


if __name__ == "__main__":
    llama = LoRAModel()
    llama.chat()
