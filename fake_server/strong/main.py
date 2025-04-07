from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

#  모델 및 토크나이저 로딩 (Qwen2.5 등등
model_name = "/home/da02/models/Llama-3.1-8B-Instruct/"  # or your local path
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    # 메시지 합치기
    history = []
    for msg in req.messages:
        history.append((msg.role, msg.content))
    
    user_input = [msg.content for msg in req.messages if msg.role == "user"][-1]
    messages = req.messages
    
    def build_prompt(messages):
        prompt = ""
        for m in messages:
            if m.role == "user":
                prompt += f"<|user|>\n{m.content}\n"
            elif m.role == "assistant":
                prompt += f"<|assistant|>\n{m.content}\n"
        prompt += "<|assistant|>\n"
        return prompt

    input_text = build_prompt(messages)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(response)

    return {
        "id": "chatcmpl-fakeid",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response
            },
            "finish_reason": "stop"
        }],
    }