# 版本2：
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="qwen2.5:7b",           # 与 ollama pull 的模型名一致
    model_provider="ollama",
    base_url="http://localhost:5005",  # Ollama 默认地址
    temperature=0.7,
)

response = llm.invoke("你好，请介绍一下自己")
print(response.content)

# 版本1：
# from langchain_ollama import ChatOllama
# from langchain_core.messages import HumanMessage

# # 1. 初始化本地 Qwen 模型
# # 请确保 model 参数与你刚才 pull 的模型名完全一致
# llm = ChatOllama(
#     model="qwen2.5:7b",
#     temperature=0.7,  # 控制回复的创造性，0-1之间
# )

# # 2. 创建一条用户消息
# messages = [HumanMessage(content="你好，请用一句话介绍你自己。")]

# # 3. 调用模型并获取回复
# response = llm.invoke(messages)

# # 4. 打印结果
# print(response.content)