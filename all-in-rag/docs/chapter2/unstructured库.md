# unstructured库

## 1. `unstructured` 库的核心功能

`unstructured` 是一个专门用于**解析非结构化文档**的Python库，可以将PDF、Word、PPT、图片、HTML等格式的文件，解析成**可处理的文档元素**（标题、段落、表格、图片等）。

> PDF/Word/HTML 文件 → partition() → 元素列表(List[Element])
>                                       ├─ Title (标题)
>                                       ├─ NarrativeText (正文段落)
>                                       ├─ ListItem (列表项)
>                                       ├─ Table (表格)
>                                       ├─ Image (图片)
>                                       └─ ...

```python
from unstructured.partition.auto import partition

pdf_path = "../../data/C2/pdf/rag.pdf"

elements = partition(
    filename=pdf_path,
    content_type="application/pdf"
)
# 打印解析结果
print(f"解析完成：{len(elements)}个元素,{sum(len(str(e)) for e in elements)} 字符")

from collections import Counter
# 统计类型
types = Counter(e.category for e in elements)
print(f'元素类型:{dict(types)}')

# 显示所有元素
print("\n 元素")
for i, element in enumerate(elements,1):
    print(f'元素{i} ({element.category}):')
    print(element)
    print("=" *60)
```





| `category`      | 含义           | 示例                      |
| :-------------- | :------------- | :------------------------ |
| `Title`         | 文档标题       | "# 第一章" 或大号加粗文字 |
| `NarrativeText` | 普通叙述段落   | "这是一个正文段落..."     |
| `ListItem`      | 列表项         | "- 第一点" 或 "1. 第一步" |
| `Table`         | 表格           | 包含表格数据              |
| `FigureCaption` | 图片/表格标题  | "图1：架构图"             |
| `Image`         | 图片（元数据） | 图片位置和描述            |
| `Header`        | 页眉           | 文档每页顶部的重复文字    |
| `Footer`        | 页脚           | 页码、版权信息等          |
| `PageBreak`     | 分页符         | 标识分页位置              |

------

+ 控制解析颗粒度

```python
elements = partition(
    filename="document.pdf",
    strategy="hi_res",      # hi_res(高清OCR) / fast(快速) / auto(自动)
    infer_table_structure=True,  # 自动识别表格结构
    extract_images_in_pdf=True,   # 提取图片
    max_characters=4000,   # 单个元素最大字符数
    new_after_n_chars=2000 # 超过此长度强制切分
)
```

+ 获取每个元素元数据

```python
for element in elements:
    print(f"内容: {element}")
    print(f"类型: {element.category}")
    print(f"页码: {element.metadata.page_number}")
    print(f"文件来源: {element.metadata.filename}")
    print("---")
```

+ 按元素类型过滤

```python
# 只保留标题和正文
filtered = [e for e in elements if e.category in ['Title', 'NarrativeText']]

# 获取所有表格
tables = [e for e in elements if e.category == 'Table']
```

## 2. 常用接口

+ 文档加载器

| 函数/类                      | 说明                     | 示例/来源                                                    |
| :--------------------------- | :----------------------- | :----------------------------------------------------------- |
| `TextLoader`                 | 加载 `.txt` 文本文件     | `from langchain_community.document_loaders import TextLoader` |
| `UnstructuredMarkdownLoader` | 加载 Markdown 文件       | `from langchain_community.document_loaders import UnstructuredMarkdownLoader` |
| `PyPDFLoader`                | 加载 PDF 文件            | 需安装 `pypdf`                                               |
| `CSVLoader`                  | 加载 CSV 文件            | `from langchain_community.document_loaders import CSVLoader` |
| `DirectoryLoader`            | 批量加载目录下的所有文件 | `from langchain_community.document_loaders import DirectoryLoader` |



+ 文本分割器

| 函数/类                          | 说明                                                         | 示例/来源                                                    |
| :------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| `RecursiveCharacterTextSplitter` | **最常用**，按字符递归分割，尽量保持语义完整性               | `from langchain.text_splitter import RecursiveCharacterTextSplitter` |
| `CharacterTextSplitter`          | 按指定字符分隔符切分                                         | `from langchain.text_splitter import CharacterTextSplitter`  |
| `SemanticChunker`                | **你已使用过**，基于语义相似度智能分割，支持多种阈值方法 (`percentile`, `standard_deviation`, `interquartile`)，需注意其所在包目前仍为实验性 | `from langchain_experimental.text_splitter import SemanticChunker` |
