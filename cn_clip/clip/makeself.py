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
        key = torch.cat([image_feat, text_feat], dim=0).cpu().numpy().flatten()  # (feature_dim,)
        self.keys.append(key)
        self.labels.append(label.item())
        self.confidences.append(confidence.item())

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
        return retrieved_labels, retrieved_confs, distances[0]


class MemoryEnhancedMatcher(nn.Module):
    def __init__(self, logit_scale=1.0, memory=DynamicMemoryBank):
        super().__init__()
        self.logit_scale = logit_scale
        self.memory = memory  # 动态记忆库

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
        retrieved_labels, retrieved_confs, _ = self.memory.retrieve_samples(query_keys)

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
            s_final = 0.7 * s_init + 0.3 * s_hist  # 权重0.7和0.3可自定义
        else:
            s_final = s_init  # 无历史样本时仅用初始相似度
        BATCH_SIZE = image_features.size(0)  # 假设可以从image_feats获取批次大小
        current_confs = torch.sigmoid(s_final).cpu().tolist()

        for i in range(BATCH_SIZE):
            if current_confs[i] > 0.8:
                # 提取对角线元素（图像与文本的匹配分数）
                self.memory.add_sample(
                    image_feat=image_features[i].cpu().numpy(),
                    text_feat=text_features[i].cpu().numpy(),
                    label=1,
                    confidence=current_confs[i]
                )
        return s_final, logits_per_image, logits_per_text


def memory_consistency_loss(confidences_pred, labels):
    """记忆一致性损失（均方误差）"""
    # confidences_pred：模型对历史样本的置信度预测（形状：(K,)）
    # labels：历史样本的真实标签（形状：(K,)）
    return F.mse_loss(confidences_pred, labels.float())

