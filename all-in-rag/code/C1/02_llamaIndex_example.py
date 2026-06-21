import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptHelper

load_dotenv()
# 手动配置 PromptHelper，指定上下文大小
prompt_helper = PromptHelper(
    context_window=4096,  # 模型的最大上下文长度
    num_output=1024,      # 预期输出的最大 token 数
    chunk_overlap_ratio=0.1
)
Settings.prompt_helper = prompt_helper

# 使用 AIHubmix
# Settings.llm = OpenAILike(
#     model="glm-4.7-flash-free",
#     api_key=os.getenv("AIHUBMIX_API_KEY"),
#     api_base="https://aihubmix.com/v1",
#     is_chat_model=True
# )

# 使用 modelscope
Settings.llm = OpenAILike(
    model="deepseek-ai/DeepSeek-V3.2", # 确认 ModelScope 上准确的模型名
    api_key=os.getenv("MODELSCOPE_API_KEY"), # 以 "ms-" 开头的 Key
    api_base="https://api-inference.modelscope.cn/v1", # 关键：指定 ModelScope 的地址
    is_chat_model=True,
    context_window=4096
)


# Settings.llm = OpenAI(
#     model="deepseek-chat",
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     api_base="https://api.deepseek.com"
# )
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine()

print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些例子?"))