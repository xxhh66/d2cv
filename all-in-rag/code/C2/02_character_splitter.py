from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
docs = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=200,     # 每个目标块大小为200字符
    chunk_overlap=10    # 每个块之间重叠10个字符
)

chunks = text_splitter.split_documents(docs)

print(f"文本被分割为{len(chunks)}个块\n")

for i,chunk in enumerate(chunks):
    print("=" *60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')