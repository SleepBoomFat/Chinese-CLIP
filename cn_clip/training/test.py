import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

# 假设已经提取了图像特征和文本特征，并计算了每个区域与文本的相似度
image_path = '/mnt/data/e97fca6d-f99a-4472-a0d4-e98c574264f0.png'
image = Image.open(image_path)

# 假设图像特征（区域）和文本特征已经通过模型提取
image_features = torch.randn(196, 768)  # 196 个区域，每个区域 768 维特征
text_features = torch.randn(1, 768)  # 假设是通过 BERT 得到的文本特征

# 计算图像区域和文本之间的相似性（余弦相似度）
similarities = torch.matmul(image_features, text_features.T)  # (196, 1)
similarities = similarities.squeeze()

# 假设图像区域的坐标已经知道（例如，使用ViT提取的区域块）
# 这里我们模拟一个 14x14 的网格，假设每个区域的坐标是 [x1, y1, x2, y2]
image_size = image.size
num_grid = 14  # 14x14 网格
region_coords = np.array([[i // num_grid * image_size[0] // num_grid,
                           i % num_grid * image_size[1] // num_grid,
                           (i // num_grid + 1) * image_size[0] // num_grid,
                           (i % num_grid + 1) * image_size[1] // num_grid]
                          for i in range(num_grid ** 2)])

# 选择与文本最相关的 top-k 区域（假设 k = 5）
top_k = 5
top_k_indices = torch.topk(similarities, top_k).indices

# 绘制图像并标记相关区域
plt.imshow(image)
for idx in top_k_indices:
    x1, y1, x2, y2 = region_coords[idx]
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='r', facecolor='none'))

plt.title("Highlighted Image Regions Related to Text")
plt.show()
