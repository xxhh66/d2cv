from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage,AIMessage,HumanMessage

llm = init_chat_model(
    model="qwen2.5:7b",           # 与 ollama pull 的模型名一致
    model_provider="ollama",
    base_url="http://localhost:5005",  # Ollama 默认地址
    temperature=0.7,
)

messages = [SystemMessage(content="你是个乐于助人的AI助手。")]

while 1:
    user_input = input("请提问：")
    messages.append(HumanMessage(content=user_input))
    res = llm.invoke(messages)
    messages.append(res)
    print("机器人回答:",res.content)