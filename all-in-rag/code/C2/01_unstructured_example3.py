from unstructured.partition.docx import partition_docx
from collections import Counter

docx_path = "../../data/C2/doc/lidar.docx"

# 解析docx文档，获取元素列表
elements = partition_docx(filename=docx_path)

# 打印基础信息
print(f"解析完成，共提取 {len(elements)} 个元素")
print(f"总字符数: {sum(len(str(e)) for e in elements)}\n")

from collections import Counter
types = Counter(e.category for e in elements)
print(f'元素类型:{dict(types)}')

# 显示所有元素
print("\n 元素")
for i, element in enumerate(elements,1):
    print(f'元素{i} ({element.category}):')
    print(element)
    print("=" *60)
