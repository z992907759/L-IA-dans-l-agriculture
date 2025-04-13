from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from your_ai_code import LoRAModel  # 导入您的 AI 模型

app = FastAPI()
model = LoRAModel()  # 初始化模型

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    content: str
    audio_path: Optional[str] = None

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response_text, audio_file = model.generate_response(request.message)
        return ChatResponse(content=response_text, audio_path=audio_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000) 