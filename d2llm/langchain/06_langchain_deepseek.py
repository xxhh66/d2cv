import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_deepseek import ChatDeepSeek
# from langchain.agents import create_json_agent

load_dotenv()
# print(os.getenv("DEEPSEEK_API_KEY"))
# 版本2：
model = ChatDeepSeek(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model="deepseek-v4-pro",
    base_url="https://api.deepseek.com"
)
response = model.invoke("你好,你是谁")
print(response.content)

# 版本1：
# model = init_chat_model(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     model = "deepseek-v4-pro",
#     model_provider="deepseek",
#     # base_url="https://api.deepseek.com",
#     temperature=0.1,
#     max_tokens = 512
# )

# response = model.invoke("你好,你是谁")
# print(response.content)