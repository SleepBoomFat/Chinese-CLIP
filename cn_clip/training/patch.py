import base64
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv

# 设置patch的大小
patch_size = (64, 64)  # 每个patch的大小为64x64


# 从TSV文件中读取base64编码的图片
def load_base64_images_from_tsv(file_path, num_images=5):
    images_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        next(reader)  # 跳过表头
        for i, row in enumerate(reader):
            if i >= num_images:
                break
            images_data.append(row[1])  # 假设第二列是base64编码图像
    return images_data


# 解码base64并将图像进行patch划分
def load_image_from_base64(base64_str):
    # 解码base64图像数据
    img_data = base64.b64decode(base64_str)
    # 打开图像
    img = Image.open(BytesIO(img_data)).convert("RGB")
    return img


# 切分图像为5x5的patch
def split_image_into_patches(image, patch_size, rows=5, cols=5):
    # 获取图像尺寸
    width, height = image.size

    # 计算每个patch的实际大小
    patch_width = width // cols
    patch_height = height // rows

    patches = []
    for i in range(rows):
        for j in range(cols):
            # 计算每个patch的坐标
            left = j * patch_width
            upper = i * patch_height
            right = left + patch_width
            lower = upper + patch_height

            # 裁剪出每个patch
            patch = image.crop((left, upper, right, lower))
            patches.append((patch, (left, upper, right, lower)))  # 保存patch和坐标
    return patches


# 展示前5张图片及其patches
def process_and_display_images(file_path, num_images=5):
    images_data = load_base64_images_from_tsv(file_path, num_images)

    for i, base64_image in enumerate(images_data):
        # 解码并加载图像
        image = load_image_from_base64(base64_image)

        # 切分图像为5x5的patch
        patches = split_image_into_patches(image, patch_size)

        # 绘制并显示patches
        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        axes = axes.flatten()  # 将二维数组展平，便于访问每个子图

        for idx, (patch, (left, upper, right, lower)) in enumerate(patches):
            axes[idx].imshow(patch)
            axes[idx].axis('off')

            # 添加红色边框标记（您指定的区域）
            # 这里假设我们要标记的一些区域的索引是 [6, 7, 11, 12, 13]

        plt.suptitle(f"Image {i + 1} - 5x5 Patches")
        plt.subplots_adjust(wspace=0.05, hspace=0.05)  # 调整子图间的间距
        plt.show()


# 文件路径
file_path = 'D:\\MUGE\\valid_imgs.tsv'  # 替换为你的TSV文件路径
process_and_display_images(file_path, num_images=5)
