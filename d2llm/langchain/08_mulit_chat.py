from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage,AIMessage,HumanMessage

llm = init_chat_model(
    model="qwen2.5:7b",           # 与 ollama pull 的模型名一致
    model_provider="ollama",
    base_url="http://localhost:5005",  # Ollama 默认地址
    temperature=0.7,
)

message = [
    SystemMessage(content="你是一个乐于助人的Ai助手！"),
    HumanMessage(content="你好，我是H王"),
]

res = llm.invoke(message)
print(res.content)
message.append(res)
print("=" *60)

message.append(HumanMessage(content="我是谁？"))
res = llm.invoke(message)
print(res.content)
print("=" *60)