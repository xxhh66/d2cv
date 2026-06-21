from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition

pdf_path = "../../data/C2/pdf/rag.pdf"

# unstructured库使用hi_res时依赖poppler和Tesseract
elements = partition_pdf(
    filename=pdf_path,
    languages=["chi_sim", "eng"], # 关键：指定中文
    strategy="hi_res",# 使用布局检测模型
    infer_table_structure=True # 识别表格结构
)

# 打印解析结果
print(f"解析完成：{len(elements)}个元素,{sum(len(str(e)) for e in elements)} 字符")

