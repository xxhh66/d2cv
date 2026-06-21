import os
import json
import numpy as np
from pymilvus import CollectionSchema,connections,MilvusClient,FieldSchema, DataType, Collection, AnnSearchRequest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# 1. 初始化设置
COLLECTION_NAME = "dragon_hybrid_demo"
MILVUS_URI="http://localhost:19530"
DATA_PATH = "../../data/C4/metadata/dragon.json"
BATCH_SIZE=50

# 2. 连接 Milvus 并初始化嵌入模型
print(f"--> 正在连接 milvus :{MILVUS_URI}")
connections.connect(uri=MILVUS_URI)

print("--> 正在初始化 BGE-M3 嵌入模型...")
ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
print(f"--> 嵌入模型初始化完成。密集向量维度: {ef.dim['dense']}")
