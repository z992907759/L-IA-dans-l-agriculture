import re
import os
import torch
import glob
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


class LoRAModel:
    def __init__(self, base_model_name="meta-llama/Llama-3.2-1B-Instruct",
                 lora_model_path="./model/llama3_lora_colab",
                 faiss_index_path="./database/faiss_db"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.faiss_index_path = faiss_index_path

        self.audio_folder = "audio"
        os.makedirs(self.audio_folder, exist_ok=True)
        self.clear_audio_folder()

        self.load_model()
        self.load_faiss_index()

    def clear_audio_folder(self):
        audio_files = glob.glob(os.path.join(self.audio_folder, "*.mp3"))
        for file in audio_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"⚠️ 无法删除 {file}: {e}")
        print("🗑️The `./audio/` directory has been cleared!")

    def load_model(self):
        print("🔄Loading the LLaMA-3 base model…")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        ).to(self.device)

        print("🔄 Loading the LoRA adapter layers…")
        self.lora_model = PeftModel.from_pretrained(self.base_model, self.lora_model_path)
        self.model = self.lora_model.merge_and_unload().to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✅LLaMA-3 + LoRA loaded successfully!")

    def load_faiss_index(self):
        print("🔍 Loading the FAISS index…")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": self.device}
        )

        self.faiss_store = FAISS.load_local(
            self.faiss_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        print("✅FAISS index loaded successfully!")

    def search_faiss(self, query, k=3):
        print(f"🔍 Serach FAISS: {query}")
        results = self.faiss_store.similarity_search(query, k=k)
        if results:
            return [doc.page_content for doc in results]
        return []

    def generate_response(self, prompt):
        faiss_results = self.search_faiss(prompt, k=3)

        if faiss_results:
            faiss_cleaned = []
            for doc in faiss_results:
                cleaned_text = re.sub(r"\(.*?et al\..*?\)", "", doc)
                cleaned_text = re.sub(r"(\d+\.\s[A-Za-z ]+)", "", cleaned_text)
                cleaned_text = cleaned_text.strip()
                if cleaned_text:
                    faiss_cleaned.append(cleaned_text)

            faiss_summary = " ".join(faiss_cleaned)[:500]
        else:
            faiss_summary = "No relevant knowledge found in FAISS."

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

        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True).to(self.device)
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1024,
                min_length=200,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        end_time = time.time()
        generation_time = end_time - start_time
        num_tokens = outputs.shape[1]
        tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0

        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        response_text = response_text.replace(full_prompt, "").strip()
        response_text = response_text.split("### Answer:")[-1].strip()

        print(f"🚀 Generation completed!")
        print(f"✅ Number of tokens generated: {num_tokens}")
        print(f"✅ Generated at: {generation_time:.2f} 秒")
        print(f"✅ Tokens generated per second: {tokens_per_second:.2f} tokens/sec")

        os.makedirs("./audio", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_file = f"./audio/{timestamp}.mp3"
        tts = gTTS(response_text, lang="en")
        tts.save(audio_file)
        print(f"🔊 The speech has been generated and saved in: {audio_file}")

        return response_text, audio_file

    def chat(self):
        os.makedirs("./audio", exist_ok=True)
        print("\n🤖 Interactive mode started! Type ‘exit’ or ‘quit' to quit.\n")

        while True:
            user_input = input("🟢 You: ").strip()
            if user_input.lower() == "exit" or user_input.lower() == "quit":
                print("👋 Good Bye！")
                break

            response, audio_file = self.generate_response(user_input)
            print(f"\n🤖 AI: {response}\n")


# ----------------- FastAPI 接口 --------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserPrompt(BaseModel):
    prompt: str

llama_instance = LoRAModel()

@app.post("/ask")
async def ask_question(item: UserPrompt):
    response_text, _ = llama_instance.generate_response(item.prompt)
    return {"answer": response_text}

# ----------------- 启动方式 ------------------------

if __name__ == "__main__":
    import threading

    # 启动 API 模式
    def run_api():
        print("🚀 Starting API server at http://localhost:8000 ...")
        uvicorn.run("Run:app", host="0.0.0.0", port=8000)

    # 后台线程启动 API 服务
    threading.Thread(target=run_api, daemon=True).start()

    # 同时启动终端交互模式
    llama = LoRAModel()
    llama.chat()