import faiss
import numpy as np
import torch

import torch
from torch import nn

import torch.nn.functional as F

class DynamicMemoryBank:
    def __init__(self, memory_size=10000, feature_dim=1024):
        self.memory_size = memory_size  # 最大存储容量
        self.feature_dim = feature_dim  # 特征维度（图像+文本特征拼接后的维度）
        self.keys = []  # 存储历史特征键（形状：(N, feature_dim)）
        self.labels = []  # 存储匹配标签（0/1，形状：(N,)）
        self.confidences = []  # 存储置信度分数（形状：(N,)）
        # 初始化FAISS索引（用于快速检索）
        self.index = faiss.IndexFlatL2(feature_dim)  # L2距离检索

    def add_sample(self, image_feat, text_feat, label, confidence):
        """添加新样本到记忆库"""
        # 特征键：图像+文本特征拼接（假设image_feat和text_feat已归一化）
        key = torch.cat([image_feat, text_feat], dim=0).detach().cpu().numpy().flatten()  # (feature_dim,)
        self.keys.append(key)
        self.labels.append(label)
        self.confidences.append(confidence)

        # 更新FAISS索引（仅当内存未满时）
        if len(self.keys) <= self.memory_size:
            self.index.add(np.array([key]))  # FAISS索引添加样本

    def retrieve_samples(self, query_feat, top_k=5):
        """检索与查询特征最相关的top_k个历史样本"""
        if len(self.keys) == 0:
            return [], [], []
        # 将查询特征转换为FAISS所需的格式（Numpy数组）
        query_np = query_feat.cpu().numpy().flatten().reshape(1, -1)  # (1, feature_dim)
        # 检索top_k个样本的索引和距离
        distances, indices = self.index.search(query_np, top_k)
        # 提取对应的标签和置信度
        retrieved_labels = [self.labels[i] for i in indices[0]]
        retrieved_confs = [self.confidences[i] for i in indices[0]]

        keys= [self.keys[i] for i in indices[0]]

        return retrieved_labels, retrieved_confs, keys


class MemoryEnhancedMatcher(nn.Module):
    def __init__(self, logit_scale=1.0, memory=DynamicMemoryBank):
        super().__init__()
        self.logit_scale = logit_scale
        self.memory = memory  # 动态记忆库
        self.fusion_weights = nn.Parameter(torch.tensor([0.7, 0.3]))

    def forward(self, image_features, text_features, labels=None):
        # 已有的logits计算（基于CLIP特征）
        logits_per_image = self.logit_scale * (image_features @ text_features.t())  # (B, B)
        logits_per_text = self.logit_scale * (text_features @ image_features.t())  # (B, B)

        # 初始相似度（取对角线元素，每个样本与自身的匹配分数）
        s_init = torch.diag(logits_per_image).clone()  # (B,)

        # ----------------------
        # 步骤1：动态记忆库检索历史样本
        # ----------------------
        # 将当前图文特征拼接为查询向量（B×(D_I+D_T)）
        query_keys = torch.cat([image_features, text_features], dim=1)  # (B, D_I+D_T)
        # 检索top_k个历史样本的标签和置信度
        retrieved_labels, retrieved_confs, topk_features = self.memory.retrieve_samples(query_keys)

        # ----------------------
        # 步骤2：融合历史相似度
        # ----------------------
        if len(retrieved_labels) > 0:
            # 历史样本的特征键（从memory.keys中获取）
            retrieved_keys = torch.tensor(self.memory.keys)[retrieved_labels.astype(bool)]  # (K, D_I+D_T)
            # 计算当前查询与历史查询的相似度（余弦相似度）
            # 注意：query_keys形状为(B, D)，retrieved_keys形状为(K, D)，需广播计算
            sim_hist = F.cosine_similarity(query_keys.unsqueeze(1), retrieved_keys.unsqueeze(0), dim=2)  # (B, K)
            # 取平均历史相似度（每个样本的历史相似度均值）
            s_hist = sim_hist.mean(dim=1)  # (B,)
            # 融合初始相似度与历史相似度（加权求和，权重可调整）
            weights = F.softmax(self.fusion_weights, dim=0)  # 对权重进行softmax归一化
            s_final = weights[0] * s_init + weights[1] * s_hist  # 加权融合
            final_features = query_keys +  topk_features

        else:
            s_final = s_init  # 无历史样本时仅用初始相似度
        BATCH_SIZE = image_features.size(0)  # 假设可以从image_feats获取批次大小
        current_confs = torch.sigmoid(s_final).cpu().tolist()
        if not hasattr(self, 'training_step'):
            self.training_step = 0
        self.training_step += 1
        # 初始阈值较低，随着训练进行逐渐提高
        dynamic_threshold = min(0.8, 0.5 + self.training_step * 0.001)

        for i in range(BATCH_SIZE):
            if current_confs[i] > dynamic_threshold:
                # 提取对角线元素（图像与文本的匹配分数）
                self.memory.add_sample(
                    image_feat=image_features[i],
                    text_feat=text_features[i],
                    label=1,
                    confidence=current_confs[i]
                )
        return s_final, logits_per_image, logits_per_text


def memory_consistency_loss(confidences_pred, labels):
    """记忆一致性损失（均方误差）"""
    # confidences_pred：模型对历史样本的置信度预测（形状：(K,)）
    # labels：历史样本的真实标签（形状：(K,)）
    return F.mse_loss(confidences_pred, labels.float())

class FeatureFusionNetwork(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(FeatureFusionNetwork, self).__init__()
        # MLP定义，用来增强特征
        self.fc1 = nn.Linear(feature_dim * 2, feature_dim * 4)  # 拼接后的特征维度是 feature_dim * 2
        self.fc2 = nn.Linear(feature_dim * 4, output_dim)

    def forward(self, fused_features):
        # 将加权融合后的特征通过MLP处理
        x = torch.relu(self.fc1(fused_features))  # (B, hidden_dim)
        enhanced_features = self.fc2(x)  # (B, output_dim)
        return enhanced_features

def get_new_feat(image_features,text_features,fusion_network):
    e_image_features,e_text_features = generate_enhanced_features(image_features,text_features,3)
    enhanced_all_features = torch.cat([e_image_features, e_text_features], dim=1)
    enhanced_all_features = fusion_network(enhanced_all_features)
    output_i, output_t = torch.split(enhanced_all_features, image_features.shape[1], dim=1)
    output_i = output_i + image_features  # 将原始图像特征与 MLP 输出的图像特征相加
    output_t = output_t + text_features  # 将原始文本特征与 MLP 输出的文本特征相加
    return output_i,output_t


def generate_enhanced_features(image_features, text_features, top_k=3):
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
        weights = similarity_scores[indices]  # (top_k,)
        weights = F.softmax(weights, dim=0)  # 使用softmax进行归一化

        # 加权融合：使用相似度对图像特征和文本特征进行加权相加
        weighted_image_features = torch.sum(similar_image_features * weights.view(-1, 1), dim=0)  # (D,)
        weighted_text_features = torch.sum(similar_text_features * weights.view(-1, 1), dim=0)  # (D,)

        # 将加权后的历史特征与当前特征相加
        fused_image_feature = image_features[i] + weighted_image_features  # (D,)
        fused_text_feature = text_features[i] + weighted_text_features  # (D,)

        # 将加权后的图像和文本特征送入MLP进行增强
        enhanced_image_features.append(fused_image_feature)
        enhanced_text_features.append(fused_text_feature)

    # 转换为tensor
    enhanced_image_features = torch.stack(enhanced_image_features)  # (B, D)
    enhanced_text_features = torch.stack(enhanced_text_features)  # (B, D)

    return enhanced_image_features, enhanced_text_features

def compute_similarity_matrix(image_features, text_features):
    """
    计算图像特征和文本特征之间的相似度矩阵
    :param image_features: 增强后的图像特征，形状 (B, D)
    :param text_features: 增强后的文本特征，形状 (B, D)
    :return: 相似度矩阵，形状 (B, B)
    """
    # 计算图像和文本之间的余弦相似度
    image_text_similarity = F.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)  # (B, B)
    return image_text_similarity
