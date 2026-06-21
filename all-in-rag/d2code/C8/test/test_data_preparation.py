"""
简单测试 load_documents 功能
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from rag_modules.data_preparation import DataPreparationModule


def main():
    """主函数：测试加载文档"""
    
    print("=" * 60)
    print("测试 DataPreparationModule.load_documents()")
    print("=" * 60)
    
    # 1. 设置数据路径
    # 根据你的实际路径调整
    data_path = "../../data/C8/cook"
    
    # 转换为绝对路径
    data_path_abs = Path(data_path).resolve()
    print(f"\n 数据路径: {data_path_abs}")
    print(f"   是否存在: {data_path_abs.exists()}")
    
    if not data_path_abs.exists():
        print(f"\n 错误: 数据路径不存在!")
        print(f"   请检查路径: {data_path}")
        print(f"   当前工作目录: {Path.cwd()}")
        return
    
    # 2. 统计文件数量
    md_files = list(data_path_abs.rglob("*.md"))
    print(f"   找到 {len(md_files)} 个 .md 文件")
    
    if len(md_files) == 0:
        print("\n 没有找到任何 .md 文件!")
        return
    
    # 显示前3个文件
    print(f"\n 示例文件:")
    for i, f in enumerate(md_files[:3], 1):
        rel_path = f.relative_to(data_path_abs)
        print(f"   {i}. {rel_path}")
    if len(md_files) > 3:
        print(f"   ... 还有 {len(md_files) - 3} 个文件")
    
    # 3. 创建模块实例并加载文档
    print("\n" + "-" * 60)
    print(" 正在加载文档...")
    print("-" * 60)
    
    try:
        # 创建数据准备模块
        module = DataPreparationModule(str(data_path_abs))
        
        # 加载文档
        documents = module.load_documents()
        
        # 4. 显示结果
        print(f"\n 加载完成!")
        print(f"   加载文档数: {len(documents)}")
        
        if len(documents) > 0:
            # 显示第一个文档信息
            first_doc = documents[0]
            print(f"\n 第一个文档信息:")
            print(f"   来源: {first_doc.metadata.get('source', 'N/A')}")
            print(f"   Parent ID: {first_doc.metadata.get('parent_id', 'N/A')[:16]}...")
            print(f"   文档类型: {first_doc.metadata.get('doc_type', 'N/A')}")
            print(f"   内容长度: {len(first_doc.page_content)} 字符")
            print(f"   内容预览:\n   {first_doc.page_content[:200]}...")
            
            # 显示所有文档的元数据统计
            print(f"\n 元数据统计:")
            doc_types = {}
            for doc in documents:
                doc_type = doc.metadata.get('doc_type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            for doc_type, count in doc_types.items():
                print(f"   - {doc_type}: {count} 个")
            
            # 显示前5个文档的 parent_id
            print(f"\n 文档 ID (前5个):")
            for i, doc in enumerate(documents[:5], 1):
                parent_id = doc.metadata.get('parent_id', 'N/A')
                source = Path(doc.metadata.get('source', '')).name
                print(f"   {i}. {source[:30]:<30} → {parent_id[:8]}...")
            
        else:
            print("\n 没有加载到任何文档")
            
    except Exception as e:
        print(f"\n 加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()