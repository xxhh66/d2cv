import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient,FieldSchema,CollectionSchema,DataType
import numpy as np
import cv2
from PIL import Image

# 初始化参数
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH="../../models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR="../../data/C3"
COLLECTION_NAME="multimodal_demo"
MILVUS_URL="http://localhost:19530"

# 定义工具
class Encoder:
    """编辑器类，用于图像和文本编码为向量"""
    def __init__(self,model_name:str,model_path:str):
        self.model = Visualized_BGE(
            model_name_bge=model_name,
            model_weight=model_path
        )
        self.model.eval()
    def encode_query(self,image_path:str,text:str)->list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path,text=text)
        return query_emb.tolist()[0]
    def encode_image(self,image_path:str)->list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]
    
def visualize_results(query_image_path: str, 
                      retrieved_images: list, 
                      img_height: int = 300, 
                      img_width: int = 300, 
                      row_count: int = 3) -> np.ndarray:
    """
    从检索到的图像列表创建一个全景图用于可视化。
    从检索到的图像列表创建一个全景图，用于【图像检索结果可视化】。
    左边显示查询图片，右边显示检索返回的图片网格。
    
    参数:
        query_image_path: 查询图片的路径
        retrieved_images: 检索返回的图片路径列表
        img_height: 每张图片显示高度
        img_width: 每张图片显示宽度
        row_count: 右边检索结果一行显示几张图（默认3列）
    
    返回:
        拼接好的大图（numpy数组，OpenCV格式）
    """
    # 全景图大小
    panoramic_width = img_width * row_count
    panoramic_height = img_height * row_count
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # 处理查询图像
    query_pil = Image.open(query_image_path).convert("RGB")
    query_cv = np.array(query_pil)[:, :, ::-1]
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 处理检索到的图像
    for i, img_path in enumerate(retrieved_images):
        row, col = i // row_count, i % row_count
        start_row, start_col = row * img_height, col * img_width
        
        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved
        
        # 添加索引号
        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return np.hstack([query_display_area, panoramic_image])


# 初始化客户端
print("=" *60)
print("-->正在初始化编码器和Milvus客户端")
encoder = Encoder(MODEL_NAME,MODEL_PATH)
milvus_client = MilvusClient(uri=MILVUS_URL)

# 创建 Milvus Collection
print(f"\n --> 正在创建Collection '{COLLECTION_NAME}'")
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"已删除已存在的Collection:'{COLLECTION_NAME}'")

image_list = glob(os.path.join(DATA_DIR,"dragon","*.png"))
if not image_list:
    raise FileNotFoundError(f"在{DATA_DIR}/dragon/ 中未找到任何 .png 图像。")
dim = len(encoder.encode_image(image_list[0]))

fields =[
    FieldSchema(name="id",dtype=DataType.INT64,is_primary=True,auto_id=True),
    FieldSchema(name="vector",dtype=DataType.FLOAT_VECTOR,dim=dim),
    FieldSchema(name="image_path",dtype=DataType.VARCHAR,max_length=512),
]

#创建集合schema
schema = CollectionSchema(fields,description="多模态图文检索")
print("Schema结构:",schema)

# 创建集合
milvus_client.create_collection(collection_name=COLLECTION_NAME,schema=schema)
print(f"成功创建 Collection:'{COLLECTION_NAME}'")
print("Collection 结构：")
print(milvus_client.describe_collection(collection_name=COLLECTION_NAME))
# 5. 准备并插入数据
print(f"\n--> 正在向 '{COLLECTION_NAME}' 插入数据")
data_to_insert = []

for image_path in tqdm(image_list,desc="生成图像嵌入"):
    vector = encoder.encode_image(image_path=image_path)
    data_to_insert.append({"vector":vector,"image_path":image_path})

if data_to_insert:
    result = milvus_client.insert(collection_name=COLLECTION_NAME,data=data_to_insert)
    print(f"成功插入{result['insert_count']}条数据。")

# 6. 创建索引
print("\n-->正在为'{COLLECTION_NAME}' 创建索引")
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M":16,"efConstruction": 256}
)

milvus_client.create_index(COLLECTION_NAME,index_params=index_params)
print("成功为向量字段创建HNSW索引")
print("索引详情：")
print(milvus_client.describe_index(collection_name=COLLECTION_NAME,
                                   index_name="vector"))
milvus_client.load_collection(collection_name=COLLECTION_NAME)
print("已加载Collection到内存")

# 7. 执行多模态检索
print("\n  正在 '{COLLECTION_NAME}' 中执行检索")
query_image_path = os.path.join(DATA_DIR,"dragon","query.png")
query_text="一条龙"
query_verctor = encoder.encode_query(image_path=query_image_path,text=query_text)

search_results=milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_verctor],
    output_fields=["image_path"],
    limit=5,
    search_params={"metric_type": "COSINE", "params": {"ef": 128}}
)[0]
retrieved_images = []
print("检索结果：")
for i,hit in enumerate(search_results):
    print(f"  Top {i+1}: ID={hit['id']}, 距离={hit['distance']:.4f}, 路径='{hit['entity']['image_path']}'")
    retrieved_images.append(hit['entity']['image_path'])


import matplotlib.pyplot as plt

result_image = visualize_results(
    query_image_path=query_image_path,
    retrieved_images=retrieved_images,
    img_height=300,
    img_width=300,
    row_count=3
)

# 使用 Matplotlib 显示（支持缩放、保存等交互操作）
plt.figure(figsize=(15, 10))
# OpenCV 是 BGR 格式，需要转换为 RGB 才能正确显示颜色
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title("Image Retrieval Results", fontsize=16)
plt.axis('off')  # 隐藏坐标轴
plt.tight_layout()
plt.show()

# 也可保存
plt.savefig("retrieval_result.png", dpi=150, bbox_inches='tight')