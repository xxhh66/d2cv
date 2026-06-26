from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

llm = init_chat_model(
    model="qwen2.5:7b",           # 与 ollama pull 的模型名一致
    model_provider="ollama",
    base_url="http://localhost:5005",  # Ollama 默认地址
    temperature=0.7,
) 
# llm = ChatOpenAI(
#     api_key="ollama",  
#     model="qwen2.5:7b",
#     base_url="http://localhost:5005/v1",
#     temperature=0.1,
#     max_tokens = 512,
#     timeout=30
# )
def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="你是一个乐于助人的助手",
)

# 运行代理
res = agent.invoke(
    {"messages": [{"role": "user", "content": "旧金山的天气怎么样"}]}
)
print("=" *60)

for i, msg in enumerate(res["messages"]):
    msg_type = msg.__class__.__name__
    content = msg.content
    
    # 截断过长的内容，方便阅读
    if len(content) > 100:
        content = content[:100] + "..."
    
    print(f"[{i:2d}] {msg_type}: {content}")
    
    # 如果有工具调用，额外打印
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tool in msg.tool_calls:
            print(f"      └─ 调用工具: {tool['name']}({tool['args']})")
    
    print("-" * 50)

