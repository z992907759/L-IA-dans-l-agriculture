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


class LoRAModel:
    def __init__(self, base_model_name="meta-llama/Llama-3.2-1B-Instruct",
                 lora_model_path="./model/llama3_lora_colab",
                 faiss_index_path="./database/faiss_db"):
        """
        ✅ LoRA 微调模型的封装类，支持 FAISS 先查询数据库
        ✅ 结合 FAISS 数据库 + LLaMA-3 生成答案
        ✅ 语音合成 (gTTS)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.faiss_index_path = faiss_index_path

        # ✅ 确保音频存储文件夹存在
        self.audio_folder = "audio"
        os.makedirs(self.audio_folder, exist_ok=True)
        self.clear_audio_folder()

        # ✅ 加载模型和 FAISS
        self.load_model()
        self.load_faiss_index()

    def clear_audio_folder(self):
        """✅ 清空 audio 文件夹"""
        for file in os.listdir(self.audio_folder):
            file_path = os.path.join(self.audio_folder, file)
            if file.endswith(".mp3"):
                os.remove(file_path)

    def load_model(self):
        """✅ 加载 LLaMA-3 + LoRA"""
        print("🔄 正在加载 LLaMA-3 基础模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        ).to(self.device)

        print("🔄 加载 LoRA 适配层...")
        self.lora_model = PeftModel.from_pretrained(self.base_model, self.lora_model_path)
        self.model = self.lora_model.merge_and_unload().to(self.device)

        # ✅ 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✅ LLaMA-3 + LoRA 加载完成！")

    def load_faiss_index(self):
        """✅ 加载 FAISS 向量数据库"""
        print("🔍 正在加载 FAISS 索引...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": self.device}
        )

        self.faiss_store = FAISS.load_local(
            self.faiss_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        print("✅ FAISS 索引加载完成！")

    def search_faiss(self, query, k=3):
        """✅ 在 FAISS 数据库中查找最相关的 k 个文档"""
        print(f"🔍 搜索 FAISS: {query}")
        results = self.faiss_store.similarity_search(query, k=k)

        if results:
            return [doc.page_content for doc in results]
        return []

    import os
    from gtts import gTTS  # ✅ 确保导入 gTTS

    import os
    import re
    import time
    from datetime import datetime
    from gtts import gTTS  # ✅ 确保导入 gTTS

    def generate_response(self, prompt):
        """✅ 结合 FAISS 结果 + LLaMA-3 生成完整答案，并用 gTTS 生成音频"""

        # 1️⃣ **查询 FAISS**
        faiss_results = self.search_faiss(prompt, k=3)

        # 2️⃣ **清理 FAISS 数据**
        if faiss_results:
            faiss_cleaned = []
            for doc in faiss_results:
                cleaned_text = re.sub(r"\(.*?et al\..*?\)", "", doc)  # ✅ 删除 "Lu et al. 2001" 这种引用
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

        print(f"📚 生成的 Prompt 长度: {len(full_prompt)}")
        print(f"📚 生成的 Prompt:\n{full_prompt[:500]}...")  # ✅ 限制输出，避免超长打印

        # 4️⃣ **让 LLaMA-3 生成答案**
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True).to(self.device)

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

        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # 5️⃣ **删除模型可能复述的 Prompt**
        response_text = response_text.replace(full_prompt, "").strip()
        response_text = response_text.split("### Answer:")[-1].strip()

        # 6️⃣ **确保 `./audio/` 目录存在**
        os.makedirs("./audio", exist_ok=True)

        # 7️⃣ **使用当前时间作为文件名**
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成 `YYYYMMDD_HHMMSS`
        audio_file = f"./audio/{timestamp}.mp3"  # ✅ 音频文件名格式示例：`./audio/20240318_153045.mp3`

        # 8️⃣ **生成 gTTS 语音并保存**
        tts = gTTS(response_text, lang="en")
        tts.save(audio_file)
        print(f"🔊 语音已生成并保存在: {audio_file}")

        return response_text, audio_file  # ✅ 返回回答和音频路径

    def chat(self):
        """✅ 交互模式启动时，清空 ./audio/ 目录，并启动对话"""

        # 1️⃣ **确保 `./audio/` 目录存在**
        os.makedirs("./audio", exist_ok=True)

        # 2️⃣ **清空 `./audio/` 目录**
        audio_files = glob.glob("./audio/*.mp3")  # 找到所有 `.mp3` 文件
        for file in audio_files:
            try:
                os.remove(file)  # ✅ 删除音频文件
            except Exception as e:
                print(f"⚠️ 无法删除 {file}: {e}")

        print("🗑️ 已清空 `./audio/` 文件夹！")

        # 3️⃣ **进入对话模式**
        print("\n🤖 交互模式启动！输入 'exit' 退出。\n")

        while True:
            user_input = input("🟢 You: ").strip()
            if user_input.lower() == "exit":
                print("👋 退出交互模式！")
                break

            response, audio_file = self.generate_response(user_input)
            print(f"\n🤖 AI: {response}\n")


if __name__ == "__main__":
    llama = LoRAModel()
    llama.chat()
