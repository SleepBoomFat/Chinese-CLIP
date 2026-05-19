import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32', ).to(device)
processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32')


# 可视化注意力
def visualize_heatmaps(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions  # (batch, heads, seq_len, seq_len)

    img = inputs.pixel_values.squeeze().permute(1, 2, 0).cpu().numpy()  # (336, 336, 3)
    img = (img - img.min()) / (img.max() - img.min())

    num_layers = len(attentions)
    cols = 4
    rows = (num_layers + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(2 * 4, 2 * rows))
    axs = axs.ravel()

    for i, attn in enumerate(attentions):
        # attn: [batch_size, num_heads, seq_len, seq_len]
        # [cls] atten
        # attn_map = attn.mean(dim=1)[0, 0, 1:].reshape(-1, 1)  # (576, 1)
        # img atten
        attn_map = attn.mean(dim=1)[0, 1:, 1:].mean(dim=0).reshape(-1, 1)

        # 转换为2D注意力图
        config = model.config
        patch_size = config.patch_size
        image_size = config.image_size
        num_patches = (image_size // patch_size) ** 2
        attn_map = attn_map[:num_patches].reshape(1, 1, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))

        # 上采样到原图尺寸
        attn_map = F.interpolate(
            attn_map.to(device),
            scale_factor=patch_size,
            mode="bilinear",
            align_corners=False
        ).squeeze()

        attn_map = attn_map.cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        axs[i].imshow(img)
        axs[i].imshow(attn_map, cmap='jet', alpha=0.5)
        axs[i].set_title(f'Layer {i + 1}')
        axs[i].axis('off')

    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    # plt.colorbar(im, ax=axs)
    plt.tight_layout()
    plt.show()

    # 保存图像
    # # plt.savefig(output_path)

img_path = r"D:\study\multimodal_validpics\pics\103003813.jpg"
visualize_heatmaps(img_path)
