from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage,AIMessage,SystemMessage
load_dotenv()

# 步骤 1：用 @tool 装饰器定义一个工具
# 函数的文档字符串就是工具的描述
# 模型会根据描述来判断何时调用这个工具
@tool
def get_weather(city:str)->str:
    """查询指定城市的天气情况。

    Args:
        city: 城市名称，如 "杭州"、"北京"
    """
    # 这里用模拟数据演示
    # 实际项目中可以替换为真实的天气 API 调用    
    weather_data={
        "杭州": "晴，25°C，湿度 60%",
        "北京": "多云，18°C，湿度 45%",
        "上海": "小雨，22°C，湿度 80%",        
    }
    return weather_data.get(city,f'未找到{city}的天气数据')

@tool
def get_calculate(expression:str)->str:
    """执行数学计算。支持加减乘除等基本运算。

    Args:
        expression: 数学表达式，如 "3 * 7 + 2"
    """
    try:
        # 安全地计算数学表达式
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"
    

if __name__ =="__main__":
    model = init_chat_model(    
        model="qwen2.5:7b",           # 与 ollama pull 的模型名一致
        model_provider="ollama",
        base_url="http://localhost:5005",  # Ollama 默认地址
        temperature=0.7,
    )

    agent = create_agent(
        model=model,
        tools = [get_weather,get_calculate],
        system_prompt="你是一个乐于助人的助手，会使用工具来回答问题",
    )

    inputs = {"messages":[HumanMessage(content="杭州今天的天气怎么样？")]}
    results = agent.invoke(input=inputs)

    print("="*60)
    for msg in results["messages"]:
        print(f"[{msg.type}] {msg.content[:100]}")
    print("\n=== 最终回复 ===")
    # 最后一条 AI 消息就是最终答案
    print(results["messages"][-1].content)

    print("="*60)
    
    input2 = {
        "messages": [HumanMessage(content="杭州和北京今天温差多少度？")]
    }
    result2 = agent.invoke(input2)

    for msg in result2["messages"]:
        if msg.type == "tool":
            print(f"[tool {msg.name}] {msg.content}")
        else:
            print(f"[{msg.type}] {msg.content[:120]}")

    print("\n=== 最终回复 ===")
    print(result2["messages"][-1].content)