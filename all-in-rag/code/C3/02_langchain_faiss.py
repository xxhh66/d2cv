from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库",
    "FAISS是一个向量数据库",
    "LangChain是一个用于开发由语言模型驱动的应用程序框架"
]
docs = [Document(page_content=t) for t in texts]
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 创建向量储存，并保存在本地
vectorstore = FAISS.from_documents(docs,embedding=embeddings)
local_faiss_path = "./faiss_index_store"
vectorstore.save_local(local_faiss_path)

print("FAISS index has been saved to {local_faiss_path}")

# 加载向量数据库，并计算向量相似性
loaded_vectorstore=FAISS.load_local(
    local_faiss_path,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
# 相似性搜索，返回的是欧式距离[0, ∞)
query = "FAISS是做什么的？"
results = loaded_vectorstore.similarity_search(query, k=4)

print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for doc in results:
    print(f"- {doc.page_content}")

# 相似性搜索，返回的是欧式距离[0, ∞)
query = "FAISS是做什么的？"
results = loaded_vectorstore.similarity_search_with_score(query, k=4)

print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for doc,score in results:
    print(f"- {doc.page_content}") 
    print(f"- 相似度得分: {score}")    