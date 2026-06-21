import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.ollama import Ollama  # 改用 Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptHelper

load_dotenv()

# 手动配置 PromptHelper
prompt_helper = PromptHelper(
    context_window=4096,
    num_output=1024,
    chunk_overlap_ratio=0.1
)
Settings.prompt_helper = prompt_helper

# ========== 修改部分：使用本地 Ollama ==========
Settings.llm = Ollama(
    model="qwen2:0.5b",  # 本地模型名称
    temperature=0.7,
    request_timeout=120.0,
    base_url="http://localhost:11434",
    context_window=4096
)

Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine()

print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些例子?"))