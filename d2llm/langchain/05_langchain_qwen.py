import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
# from langchain.agents import create_json_agent

load_dotenv()

model = init_chat_model(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model = "qwen3.6-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.1,
    max_tokens = 512
)

response = model.invoke("你好,你是谁")
print(response.content)