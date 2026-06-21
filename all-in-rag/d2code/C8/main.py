"""RAG系统主系统"""

import os
import sys
import logging
from pathlib import Path
from typing import List

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig


class RecipeRAGSystem:
    """食谱RAG系统主类"""
    def __init__(self,config:RAGConfig=None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """        
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        print(f"RAGConfig配置LLM:{self.config.llm_model}\n embedding model: {self.config.embedding_model}")
        pass    
# 添加模块路径

if __name__ == "__main__":
    rag_system = RecipeRAGSystem()
    print("RAG系统主系统")