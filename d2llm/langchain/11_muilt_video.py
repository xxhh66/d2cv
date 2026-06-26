from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
model = init_chat_model(
    model="qwen3.5-omni-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

messages = [
    HumanMessage(content=[
        {
            "type": "video_url",
            "video_url": {
                "url": "https://aweme.snssdk.com/aweme/v1/play/?video_id=v0200fg10000cuhogevog65hkurqon00&ratio=720p&line=0"
            },
        },
        {
            "type": "text",
            "text": "请只提取这段视频中的语音内容，转写成文字。如果有多个人说话，请尽量区分说话人。"
        },


    ])
]

# res = model.invoke(messages)
# print(res.content)

res =model.stream(messages)
for chunk in res:
    print(chunk.content,end="",flush=True)