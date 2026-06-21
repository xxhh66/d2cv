from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
docs = loader.load()

separators=[
    "\n\n", "\n", " ",
    ".", ",", "\u200b",      # 零宽空格(泰文、日文)
    "\uff0c", "\u3001",      # 全角逗号、表意逗号
    "\uff0e", "\u3002",      # 全角句号、表意句号
    ""
]

# spiltter = RecursiveCharacterTextSplitter.from_language(
#     language=Language.PYTHON
# )

text_splitter = RecursiveCharacterTextSplitter(
    # 针对中英文混合文本，定义一个更全面的分隔符列表
    separators=separators, # 按顺序尝试分割
    chunk_size=200,
    chunk_overlap=10
)

chunks = text_splitter.split_documents(docs)

print(f"文本被切分为 {len(chunks)} 个块。\n")
print("--- 前5个块内容示例 ---")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')
