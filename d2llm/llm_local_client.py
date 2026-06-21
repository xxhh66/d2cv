import os
import time
from typing import List, Dict, Any

# ============ 1. LangChain Community ============
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ============ 2. LangChain OpenAI Compatible ============
from langchain_openai import ChatOpenAI

# ============ 3. LlamaIndex ============
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage, MessageRole

# ============ 4. Official OpenAI Client ============
from openai import OpenAI


class LocalLLMTest:
    """本地LLM测试类"""
    
    def __init__(self, 
                 model_name: str = "qwen2.5:7b", 
                 base_url: str = "http://localhost:5005",
                 temperature=0.1,
                 max_tokens=4096):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.test_prompt = "请用一句话介绍什么是人工智能"
        self.test_messages = [
            {"role": "system", "content": "你是一个专业的AI助手"},
            {"role": "user", "content": "请用一句话介绍什么是人工智能"}
        ]
    
    def test_langchain_chatollama(self):
        """方法1: LangChain Community - ChatOllama"""
        print("\n" + "="*60)
        print("1. LangChain Community - ChatOllama")
        print("="*60)
        
        try:
            llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens,
            )
            
            # 测试1: 简单调用
            start = time.time()
            response = llm.invoke(self.test_prompt)
            elapsed = time.time() - start
            
            print(f"✅ 响应时间: {elapsed:.2f}s")
            print(f"📝 响应内容: {response.content}")
            
            # 测试2: 消息列表调用
            messages = [
                SystemMessage(content="你是一个专业的AI助手"),
                HumanMessage(content=self.test_prompt)
            ]
            response2 = llm.invoke(messages)
            print(f"\n📝 消息列表调用: {response2.content[:100]}...")
            
            # 测试3: 流式输出
            print("\n🔄 流式输出:")
            for chunk in llm.stream(self.test_prompt):
                print(chunk.content, end="", flush=True)
            print("\n")
            
            return True
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            return False
    
    def test_langchain_openai_like(self):
        print("\n" + "="*60)
        """方法2: LangChain - ChatOpenAI (OpenAI兼容模式)"""
        print("\n" + "="*60)
        print("2. LangChain - ChatOpenAI (兼容Ollama)")
        print("="*60)
        
        try:
            llm = ChatOpenAI(
                model=self.model_name,
                base_url=f"{self.base_url}/v1",
                api_key="ollama",  # Ollama不需要真实的API key
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            start = time.time()
            response = llm.invoke(self.test_prompt)
            elapsed = time.time() - start
            
            print(f"✅ 响应时间: {elapsed:.2f}s")
            print(f"📝 响应内容: {response.content}")
            
            return True
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            return False
    
    def test_llamaindex_ollama(self):
        print("\n" + "="*60)
        """方法3: LlamaIndex - Ollama"""
        print("\n" + "="*60)
        print("3. LlamaIndex - Ollama")
        print("="*60)
        
        try:
            llm = Ollama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                request_timeout=120.0,
            )
            
            # 测试: 普通调用
            start = time.time()
            response = llm.complete(self.test_prompt)
            elapsed = time.time() - start
            
            print(f"✅ 响应时间: {elapsed:.2f}s")
            print(f"📝 响应内容: {response.text}")
            
            # 测试: 聊天消息
            messages = [
                ChatMessage(role=MessageRole.SYSTEM, content="你是一个专业的AI助手"),
                ChatMessage(role=MessageRole.USER, content=self.test_prompt)
            ]
            response2 = llm.chat(messages)
            print(f"\n📝 聊天调用: {response2.message.content[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            return False
    
    def test_llamaindex_openailike(self):
        print("\n" + "="*60)
        """方法4: LlamaIndex - OpenAILike"""
        print("\n" + "="*60)
        print("4. LlamaIndex - OpenAILike")
        print("="*60)
        
        try:
            llm = OpenAILike(
                model=self.model_name,
                api_base=f"{self.base_url}/v1",
                api_key="ollama",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            start = time.time()
            response = llm.complete(self.test_prompt)
            elapsed = time.time() - start
            
            print(f"✅ 响应时间: {elapsed:.2f}s")
            print(f"📝 响应内容: {response.text}")
            
            return True
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            return False
    
    
    def test_openai_client(self):
        print("\n" + "="*60)
        """方法5: OpenAI官方客户端 (指向Ollama)"""
        print("\n" + "="*60)
        print("5. OpenAI Client (指向Ollama)")
        print("="*60)
        
        try:
            client = OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key="ollama",
            )
            
            start = time.time()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=self.test_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            elapsed = time.time() - start
            
            print(f"✅ 响应时间: {elapsed:.2f}s")
            print(f"📝 响应内容: {response.choices[0].message.content}")
            
            # 流式输出
            print("\n🔄 流式输出:")
            stream = client.chat.completions.create(
                model=self.model_name,
                messages=self.test_messages,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print("\n")
            
            return True
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("\n" + "🚀 开始测试本地LLM (Qwen2.5:7b)".center(60, "="))
        print(f"Model: {self.model_name}")
        print(f"Base URL: {self.base_url}")
        print(f"Temperature: {self.temperature}")
        print(f"Max Tokens: {self.max_tokens}")
        print("\n" + "="*60)
        
        results = []
        
        # 依次执行所有测试
        tests = [
            ("LangChain ChatOllama", self.test_langchain_chatollama),
            ("LangChain ChatOpenAI", self.test_langchain_openai_like),
            ("LlamaIndex Ollama", self.test_llamaindex_ollama),
            ("LlamaIndex OpenAILike", self.test_llamaindex_openailike),
            ("OpenAI Client", self.test_openai_client),
        ]
        
        for name, test_func in tests:
            try:
                success = test_func()
                results.append((name, "✅ 成功" if success else "❌ 失败"))
            except Exception as e:
                results.append((name, f"❌ 异常: {str(e)[:50]}..."))
        
        # 打印测试总结
        print("\n" + "="*60)
        print("📊 测试总结".center(60))
        print("="*60)
        for name, status in results:
            print(f"{name:<30} {status}")



# ============ 依赖安装说明 ============

INSTALL_COMMANDS = """
# 安装必要的库
pip install langchain langchain-community langchain-openai
pip install llama-index llama-index-llms-ollama llama-index-llms-openai llama-index-llms-openai-like
pip install openai

# 确保Ollama已安装并运行
# 下载模型
ollama pull qwen2.5:7b

# 启动Ollama服务 (通常默认已启动)
ollama serve
"""


if __name__ == "__main__":
    print("\n📦 依赖安装命令:")
    print(INSTALL_COMMANDS)
    
    # 运行完整测试
    tester = LocalLLMTest()
    tester.run_all_tests()
    
    # 或者单独测试某个方法
    # tester.test_langchain_chatollama()
    # tester.test_langchain_openai_like()
    # tester.test_llamaindex_ollama()
    # tester.test_llamaindex_openailike()
    # tester.test_openai_client()
