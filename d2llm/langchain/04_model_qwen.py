import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

print(os.getenv("DASHSCOPE_API_KEY"))

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen3.6-plus",
    messages=[
        # system 消息：用于设定模型的角色、风格或规则
        {"role": "system", "content": "你是一名 AI 助教。"},

        # user 消息：表示用户真正提出的问题
        {"role": "user", "content": "你好，你是谁？来自哪一个厂商？"},
    ]
)
print(completion.choices[0].message.content)