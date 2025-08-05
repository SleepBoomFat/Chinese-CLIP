import base64
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import csv

# 加载预训练的 Faster R-CNN 模型
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # 设置模型为评估模式


# 加载图像并进行预处理
def load_image_from_base64(base64_str):
    # 解码base64图像数据
    img_data = base64.b64decode(base64_str)
    # 打开图像
    img = Image.open(BytesIO(img_data)).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor格式
    ])
    return transform(img).unsqueeze(0)  # 增加批次维度


# 绘制边界框
def plot_results(image, boxes, labels, scores, threshold=0.5):
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    ax = plt.gca()

    # 过滤低分数的框
    for i in range(len(boxes)):
        if scores[i] > threshold:
            x1, y1, x2, y2 = boxes[i]
            # 绘制红色边界框
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, color='red', linewidth=3))
            # 添加类别标签与分数
            ax.text(x1, y1, f'{labels[i]}: {scores[i]:.2f}',
                    color='yellow', fontsize=12, backgroundcolor='black')

    plt.axis('off')
    plt.show()


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


# 处理并展示前5张图片
def process_and_display_images(file_path, num_images=5):
    images_data = load_base64_images_from_tsv(file_path, num_images)

    for i, base64_image in enumerate(images_data):
        # 解码并加载图像
        image_tensor = load_image_from_base64(base64_image)

        # 将输入图像送入 Faster R-CNN 模型
        with torch.no_grad():
            prediction = model(image_tensor)

        # 获取预测结果
        boxes = prediction[0]['boxes'].cpu().numpy()  # 边界框坐标
        labels = prediction[0]['labels'].cpu().numpy()  # 物体类别标签
        scores = prediction[0]['scores'].cpu().numpy()  # 置信度分数

        # 加载原始图像，用于绘制边界框
        image = Image.open(BytesIO(base64.b64decode(base64_image)))

        # 绘制边界框
        plot_results(image, boxes, labels, scores, threshold=0.5)
        print(f"Processed Image {i + 1}.")


# 调用函数，处理并展示前5张图像
file_path = 'D:\\MUGE\\valid_imgs.tsv'  # 替换为您的TSV文件路径
process_and_display_images(file_path, num_images=5)

