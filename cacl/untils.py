import torch

def ranking_loss(y_true, y_pred):
    """
    Ranking Loss for multi-label classification
    y_true: (N, C) {0,1} ground-truth labels
    y_pred: (N, C) predicted scores (float)
    """
    N, C = y_true.shape
    loss = 0.0
    valid_count = 0
    for i in range(N):
        pos_idx = (y_true[i] == 1).nonzero(as_tuple=True)[0]
        neg_idx = (y_true[i] == 0).nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        pos_scores = y_pred[i][pos_idx]
        neg_scores = y_pred[i][neg_idx]
        # count how many negative scores are ranked above positives
        pairwise_comp = (neg_scores[:, None] >= pos_scores[None, :]).float()
        loss += pairwise_comp.mean().item()
        valid_count += 1
    return loss / max(valid_count, 1)


def pairwise_accuracy(y_true, y_pred):
    """
    Pairwise Accuracy for multi-label classification
    y_true: (N, C) {0,1} ground-truth labels
    y_pred: (N, C) predicted scores (float)
    """
    N, C = y_true.shape
    acc = 0.0
    valid_count = 0
    for i in range(N):
        pos_idx = (y_true[i] == 1).nonzero(as_tuple=True)[0]
        neg_idx = (y_true[i] == 0).nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        pos_scores = y_pred[i][pos_idx]
        neg_scores = y_pred[i][neg_idx]
        # count how many pairs are correctly ranked
        pairwise_comp = (pos_scores[:, None] > neg_scores[None, :]).float()
        acc += pairwise_comp.mean().item()
        valid_count += 1
    return acc / max(valid_count, 1)