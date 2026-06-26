# # 文件路径：config.py
# # 在程序开头加载 .env 文件
# import os
# from dotenv import load_dotenv

# # 加载 .env 文件中的环境变量
# load_dotenv()

# # 验证 API Key 是否加载成功
# api_key = os.getenv("OPENAI_API_KEY")
# if api_key:
#     # 只显示前8位和后4位，避免泄露完整 Key
#     print(f"API Key 已加载: {api_key[:8]}...{api_key[-4:]}")
# else:
#     print("警告：未找到 OPENAI_API_KEY，请检查 .env 文件")


# 文件路径：verify_install.py
# 验证 LangChain 安装和 API Key 配置
from dotenv import load_dotenv
load_dotenv()

# 测试 1：验证 langchain 导入
try:
    import langchain
    print(f"langchain 版本: {langchain.__version__}")
except ImportError:
    print("错误：langchain 未安装，请运行 pip install langchain")

# 测试 2：验证 langchain-openai 导入
try:
    import langchain_openai
    print("langchain-openai 已安装")
except ImportError:
    print("错误：langchain-openai 未安装，请运行 pip install langchain-openai")

# 测试 3：验证 API Key 配置
import os
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("错误：OPENAI_API_KEY 未配置，请在 .env 文件中设置")
else:
    print(f"API Key 配置成功: {api_key[:8]}...{api_key[-4:]}")

# 测试 4：发送一条测试请求
from langchain.chat_models import init_chat_model

model = init_chat_model("ollama:qwen2.5:7b")
# model = init_chat_model("openai:gpt-4o-mini")
response = model.invoke("用一句话介绍菜鸟教程（RUNOOB）")
print(f"\n模型回复: {response.content}")