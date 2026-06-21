from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker #基于语义相似度来分割文本

loader = TextLoader("../../data/C2/txt/蜂医.txt",encoding="utf-8")
docs = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap=10
)

chunks = text_splitter.split_documents(docs)
print(f"文本被分割为{len(chunks)}个块 \n")

for i,chunk in enumerate(chunks[:1]):
    print("="*60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')

separators=[
    "\n\n", "\n", " ",
    ".", ",", "\u200b",      # 零宽空格(泰文、日文)
    "\uff0c", "\u3001",      # 全角逗号、表意逗号
    "\uff0e", "\u3002",      # 全角句号、表意句号
    ""
]
text_splitter_recursive = RecursiveCharacterTextSplitter(
    separators=separators,
    chunk_size = 100,
    chunk_overlap=10
)

chunks = text_splitter_recursive.split_documents(docs)
print(f"文本被分割为{len(chunks)}个块 \n")

for i,chunk in enumerate(chunks[:1]):
    print("="*60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')



embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile" # 也可以是 "standard_deviation", "interquartile", "gradient"
)


chunks = text_splitter.split_documents(docs)

print(f"文本被切分为 {len(chunks)} 个块。\n")
print("--- 前2个块内容示例 ---")
for i, chunk in enumerate(chunks[:2]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}):\n"{chunk.page_content}"')
