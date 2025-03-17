import torch
import os
import shutil
import pyttsx3
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from gtts import gTTS
import torch


class LoRAModel:
    def __init__(self, base_model_name="meta-llama/Llama-3.2-1B-Instruct",
                 lora_model_path="./Model/llama3_lora_colab",
                 audio_dir="./audio"):
        """
        ✅ LoRA 微调模型的封装类
        ✅ 支持加载 LLaMA-3 + LoRA 适配层
        ✅ 支持 GPU 加速
        ✅ 支持用户交互式对话
        ✅ 支持生成语音音频
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path
        self.audio_dir = audio_dir

        # ✅ 初始化TTS引擎
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # 语速
        self.tts_engine.setProperty('volume', 1.0)  # 音量

        # ✅ 清空音频目录
        self.clear_audio_directory()

        # ✅ 加载模型
        self.load_model()

        # ✅ 打印当前设备运行类型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            print(f"🔍 当前设备: GPU ({torch.cuda.get_device_name(0)})")
            print(f"🛠️ CUDA 版本: {torch.version.cuda}")
            print(f"💾 可用显存: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        else:
            print("🔍 当前设备: CPU")

    def clear_audio_directory(self):
        """✅ 清空 `audio/` 目录"""
        if os.path.exists(self.audio_dir):
            shutil.rmtree(self.audio_dir)
        os.makedirs(self.audio_dir, exist_ok=True)

    def load_model(self):
        """✅ 加载 LLaMA-3 基础模型，并合并 LoRA 适配层"""
        print("🔄 正在加载 LLaMA-3 基础模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        ).to(self.device)

        print("🔄 Loading LoRA adaptation layer...")
        self.lora_model = PeftModel.from_pretrained(self.base_model, self.lora_model_path)

        print("🔄 Merging LoRA adaptation layer...")
        self.model = self.lora_model.merge_and_unload().to(self.device)

        # ✅ 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✅ Model loading completed!")

    def generate_response(self, prompt):
        """✅ 优化 Prompt，减少 AI 复述问题的概率"""
        optimized_prompt = f"Answer directly without repeating the question:\n{prompt}"  # 关键点
        inputs = self.tokenizer(optimized_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 1024,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,  # ✅ 增强惩罚，减少重复
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # ✅ 进一步去除 AI 复述的部分
        if response_text.lower().startswith(prompt.lower()):  # 如果回答前半部分是问题
            response_text = response_text[len(prompt):].strip()  # 去掉问题部分

        print(f"🤖 AI: {response_text}")
        self.generate_audio(response_text)
        return response_text

    def generate_audio(self, text):
        """✅ 使用 gTTS 生成语音，并保存为 MP3 文件"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = os.path.join(self.audio_dir, f"audio_{timestamp}.mp3")

        try:
            # 生成音频（语言默认是英语，你可以改成 'zh-cn' ）
            tts = gTTS(text=text, lang="en")  # 如果要用中文，改成 lang="zh-cn"
            tts.save(audio_path)
            print(f"🔊 语音已生成: {audio_path}")

        except Exception as e:
            print(f"❌ 语音生成失败: {e}")

    def chat(self):
        """✅ 支持用户交互式聊天"""
        print("\n🤖 LLaMA-3 LoRA model launched! Type 'exit' to exit.\n")
        while True:
            user_input = input("🟢 You: ")
            if user_input.lower() == "exit":
                print("👋 Good Bye！")
                break
            response = self.generate_response(user_input)


# ✅ 运行交互式聊天
if __name__ == "__main__":
    llama = LoRAModel()
    llama.chat()
