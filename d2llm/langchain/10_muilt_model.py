from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
model = init_chat_model(
    model="qwen3-asr-flash",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

messages = [
    HumanMessage(content=[
        {
            "type": "input_audio",
            "input_audio": {
                "data": "https://dashscope.oss-cn-beijing.aliyuncs.com/audios/welcome.mp3"
            }
        }
    ])
]

res = model.invoke(messages)
print(res.content)