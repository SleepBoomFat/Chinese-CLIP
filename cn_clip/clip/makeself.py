import faiss
import numpy as np
import torch

import torch
from torch import nn

import torch.nn.functional as F

class FeatureMemoryNetwork(nn.Module):
    def __init__(self, feature_dim, output_dim, dropout_prob=0.5):
        super(FeatureMemoryNetwork, self).__init__()
        # MLP定义，用来增强特征
        self.fc1 = nn.Linear(feature_dim * 2, feature_dim * 4)  # 拼接后的特征维度是 feature_dim * 2
        self.fc2 = nn.Linear(feature_dim * 4, output_dim)
        self.dropout = nn.Dropout(dropout_prob)  # 添加 Dropout 层

    def forward(self, fused_features):
        # 将加权融合后的特征通过MLP处理
        x = torch.relu(self.fc1(fused_features))  # (B, hidden_dim)
        x = self.dropout(x)  # 应用 Dropout
        enhanced_features = self.fc2(x)  # (B, output_dim)
        return enhanced_features

def get_memory_feat(image_features, text_features, top_k):
    e_image_features, e_text_features = generate_enhanced_features(image_features, text_features, top_k)
    return e_image_features, e_text_features

def generate_enhanced_features(image_features, text_features, top_k=2):
    """
    :param image_features: 图像特征，形状 (B, D)
    :param text_features: 文本特征，形状 (B, D)
    :param top_k: 选择与每个样本最相似的前K个样本
    :return: 增强后的图像和文本特征
    """
    B, D = image_features.shape
    all_features = torch.cat([image_features, text_features], dim=1)  # (B, 2*D)

    # 计算全局特征的相似度矩阵（使用余弦相似度）
    similarity_matrix = F.cosine_similarity(all_features.unsqueeze(1), all_features.unsqueeze(0), dim=2)  # (B, B)

    # 存储增强后的图像和文本特征
    enhanced_image_features = []
    enhanced_text_features = []

    for i in range(B):
        # 获取与当前样本最相似的top_k个样本的索引（排除自身）
        similarity_scores = similarity_matrix[i]
        _, indices = torch.topk(similarity_scores, top_k + 1)  # 选出前top_k+1个，包括自身
        indices = indices[1:]  # 排除自身

        # 获取相似的图像和文本特征
        similar_image_features = image_features[indices]  # (top_k, D)
        similar_text_features = text_features[indices]  # (top_k, D)

        # 计算加权融合，使用相似度作为加权系数
        # weights = similarity_scores[indices]  # (top_k,)
        # weights = torch.sigmoid(weights)  # 使用sigmoid进行归一化，避免过度放大权重
        weights = F.softmax(similarity_scores[indices], dim=0)  # 使用softmax进行加权

        # 加权融合：使用相似度对图像特征和文本特征进行加权相加
        weighted_image_features = torch.sum(similar_image_features * weights.view(-1, 1), dim=0)  # (D,)
        weighted_text_features = torch.sum(similar_text_features * weights.view(-1, 1), dim=0)  # (D,)

        # 将加权后的历史特征与当前特征相加


        # 将加权后的图像和文本特征送入MLP进行增强
        enhanced_image_features.append(weighted_image_features)
        enhanced_text_features.append(weighted_text_features)

    # 转换为tensor
    # enhanced_image_features = torch.stack(enhanced_image_features)  # (B, D)
    # enhanced_text_features = torch.stack(enhanced_text_features)  # (B, D)

    enhanced_image_features = F.dropout(torch.stack(enhanced_image_features), p=0.5, training=True)
    enhanced_text_features = F.dropout(torch.stack(enhanced_text_features), p=0.5, training=True)

    return enhanced_image_features, enhanced_text_features
