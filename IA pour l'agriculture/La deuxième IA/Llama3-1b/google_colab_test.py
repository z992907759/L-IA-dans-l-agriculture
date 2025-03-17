import os
import torch
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datetime import datetime


class LoRAModel:
    def __init__(self, base_model_name="meta-llama/Llama-3.2-1B-Instruct",
                 lora_model_path="./Model/llama3_lora_colab"):
        """
        ✅ LoRA 微调模型的封装类
        ✅ 支持加载 LLaMA-3 + LoRA 适配层
        ✅ 支持 GPU 加速
        ✅ 支持用户交互式对话
        ✅ 生成语音回复 (gTTS)
        ✅ 自动清空 audio 文件夹
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_name = base_model_name
        self.lora_model_path = lora_model_path

        # ✅ 确保音频存储文件夹存在
        self.audio_folder = "audio"
        os.makedirs(self.audio_folder, exist_ok=True)

        # ✅ 清空 audio 文件夹
        self.clear_audio_folder()

        # ✅ 加载模型
        self.load_model()

    def clear_audio_folder(self):
        """✅ 清空 audio 文件夹"""
        print("🗑️ 正在清空 audio 文件夹...")
        for file in os.listdir(self.audio_folder):
            file_path = os.path.join(self.audio_folder, file)
            if file.endswith(".mp3"):  # 仅删除 MP3 文件
                os.remove(file_path)
        print("✅ audio 文件夹已清空！")

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
        """✅ 生成文本，并调用 gTTS 生成音频"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 1024,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # ✅ 生成音频文件
        audio_path = self.generate_audio(response_text)

        return response_text, audio_path

    def generate_audio(self, text):
        """✅ 使用 gTTS 生成音频"""
        print("🔊 正在生成语音...")

        # ✅ 生成唯一的音频文件名
        filename = f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        audio_path = os.path.join(self.audio_folder, filename)

        # ✅ 生成音频
        tts = gTTS(text=text, lang="en")  # 语言可改为 "zh-CN" (中文) 或其他支持的语言
        tts.save(audio_path)

        print(f"✅ 语音已生成: {audio_path}")
        return audio_path

    def chat(self):
        """✅ 交互式聊天，支持文本+语音"""
        print("\n🤖 LLaMA-3 LoRA model launched! Type 'exit' to exit.\n")
        while True:
            user_input = input("🟢 You: ")
            if user_input.lower() == "exit":
                print("👋 Good Bye！")
                break
            response, audio_file = self.generate_response(user_input)
            print(f"🤖 AI: {response}\n")

            # ✅ 选择是否播放音频
            # play_audio = input("🎵 播放语音？(y/n): ").strip().lower()
            # if play_audio == "y":
            #     os.system(f"open {audio_file}")  # Mac
                # os.system(f"start {audio_file}")  # Windows
                # os.system(f"xdg-open {audio_file}")  # Linux


# ✅ 运行交互式聊天
if __name__ == "__main__":
    llama = LoRAModel()
    llama.chat()