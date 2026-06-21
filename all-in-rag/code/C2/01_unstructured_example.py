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
