import torch
import torch.nn as nn
import torch.nn.functional as F

# class-specific loss
'''class SoftMarginHingeEmbeddingLoss(nn.Module):
    def __init__(self, margin=1.0, class_counts=None, gamma=2):
        super(SoftMarginHingeEmbeddingLoss, self).__init__()
        self.margin = margin
        assert self.margin > 0
        self.class_counts = class_counts
        self.gamma = gamma

    def forward(self, inputs, labels):
        # Compute the dot product between the inputs and their corresponding labels
        dot_product = torch.sum(inputs * labels, dim=2)
        margin = self.margin
        if self.class_counts is not None:
            labels = labels.long()
            self.class_counts = self.class_counts.squeeze(dim=0).expand(labels.shape[0], self.class_counts.shape[-1])
            margin = self.margin / torch.sqrt(torch.sqrt(self.class_counts.float()))
        # Compute the hinge loss
        hinge_loss = torch.relu(margin - dot_product)
        if self.class_counts is not None:
            a = (1 / self.class_counts) ** self.gamma
            b = torch.sum((1 / self.class_counts) ** self.gamma, dim=0)
            class_weights = (1 / self.class_counts) ** self.gamma / torch.sum((1 / self.class_counts) ** self.gamma, dim=0)
            hinge_loss = hinge_loss * class_weights
            
        return hinge_loss.mean()'''

class SemanticAwareHingeEmbeddingLoss(nn.Module):
    def __init__(self, margin=1.0, class_counts=None, gamma=2):
        super(SemanticAwareHingeEmbeddingLoss, self).__init__()
        self.margin = margin
        assert self.margin > 0
        self.class_counts = class_counts
        self.gamma = gamma

    def forward(self, inputs, labels, semantic_sim=None):
        """
        inputs: (B, C, D)  # prompt embeddings
        labels: (B, C, D)  # target embeddings, +1 for positive, -1 for negative
        semantic_sim: (C, C) semantic similarity matrix S_ij
        """

        # cosine similarity distance: d(a, b) = 1 - cos(a, b)
        dot_product = torch.sum(inputs * labels, dim=2)  # (B, C)
        margin = self.margin

        if self.class_counts is not None:
            labels_idx = labels.long()
            self.class_counts = self.class_counts.squeeze(dim=0).expand(
                labels_idx.shape[0], self.class_counts.shape[-1]
            )
            margin = self.margin / torch.sqrt(torch.sqrt(self.class_counts.float()))

        # hinge term: relu(margin - dot_product)
        hinge_loss = torch.relu(margin - dot_product)  # (B, C)

        # apply semantic-aware weights
        if semantic_sim is not None:
            # w_ij = 1 - S_ij
            weight = 1.0 - semantic_sim  # (C, C)

            # broadcast到batch
            B, C = labels.shape[0], labels.shape[1]
            weight = weight.unsqueeze(0).expand(B, C, C)  # (B, C, C)

            # 扩展 hinge_loss
            hinge_expanded = hinge_loss.unsqueeze(2).expand(B, C, C)  # (B, C, C)

            # 只对负样本生效: labels = -1
            neg_mask = (labels[:, :, 0] == -1).unsqueeze(2).expand(B, C, C)

            hinge_loss = (hinge_expanded * weight * neg_mask.float()).mean()

        else:
            hinge_loss = hinge_loss.mean()

        return hinge_loss