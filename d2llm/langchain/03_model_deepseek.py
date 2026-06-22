# 从 openai 包中导入 OpenAI 客户端类
# 注意：这里虽然导入的是 OpenAI，但因为 DeepSeek 兼容 OpenAI 接口格式，
# 所以也可以用 OpenAI SDK 调用 DeepSeek
from openai import OpenAI

# 导入 load_dotenv，用来读取 .env 文件中的环境变量
from dotenv import load_dotenv

# 导入 os，用来读取系统环境变量
import os

# 加载当前项目目录下的 .env 文件
# 例如 .env 中可以写：DEEPSEEK_API_KEY=你的 API Key
load_dotenv()
print(f"api_key:{os.getenv("DEEPSEEK_API_KEY")}")
# 创建模型服务客户端
client = OpenAI(
    # 从环境变量中读取 DeepSeek 的 API Key
    api_key=os.getenv("DEEPSEEK_API_KEY"),

    # 指定 DeepSeek 的 API 地址
    # 这样请求会发往 DeepSeek，而不是 OpenAI 官方服务
    base_url="https://api.deepseek.com",
)

# 调用对话模型接口，发送一次聊天请求
response = client.chat.completions.create(
    # 指定要调用的 DeepSeek 模型
    model="deepseek-v4-pro",

    # messages 是对话模型的输入，格式是一个消息列表
    messages=[
        # system 消息：用于设定模型的角色、风格或规则
        {"role": "system", "content": "你是一名 AI 助教。"},

        # user 消息：表示用户真正提出的问题
        {"role": "user", "content": "你好，你是谁？来自哪一个厂商？"},
    ],

    # stream=False 表示不使用流式输出
    # 模型生成完整回答后，一次性返回结果
    stream=False,

    # temperature 控制回答的随机性
    # 值越高，回答越发散；值越低，回答越稳定
    temperature=0.1,
)

# 从响应对象中取出模型生成的文本内容
# choices[0] 表示取第一个候选回答
# message.content 表示取这个回答的正文
print(response.choices[0].message.content)