import torch
import torch.nn.functional as F

@torch.no_grad()
def build_text_semantic_matrix(custom_clip, device="cuda"):
    """
    用 CustomCLIP 的 PromptLearner + TextEncoder 构造类别之间的语义相似度矩阵 S (C, C)

    custom_clip: 你的 CustomCLIP 模型
    device: 运行设备
    """
    custom_clip.eval()

    # 从 PromptLearner 得到每个类别的 prompt 序列 (n_cls, ctx_len, ctx_dim)
    prompts = custom_clip.prompt_learner().to(device)
    # token ids: (n_cls, ctx_len)
    tokenized_prompts = custom_clip.tokenized_prompts.to(device)

    # 用 TextEncoder 得到类别嵌入 (n_cls, dim)
    text_features = custom_clip.text_encoder(prompts, tokenized_prompts)
    text_features = F.normalize(text_features, dim=-1)   # 单位化

    # 相似度矩阵 S_ij = cos(t_i, t_j)
    S = text_features @ text_features.t()  # (C, C), [-1, 1]

    # 归一化到 [0,1]
    S = (S + 1.0) / 2.0
    S.fill_diagonal_(1.0)

    return S

@torch.no_grad()
def build_pmi_matrix(train_labels, eps=1.0, normalize_to_01=True, device="cuda"):
    """
    train_labels: (N, C) 的 0/1 张量或 numpy 数组，训练集多标签
    eps: 拉普拉斯平滑
    """
    if not torch.is_tensor(train_labels):
        train_labels = torch.tensor(train_labels)
    train_labels = train_labels.to(device).float()

    N, C = train_labels.shape
    co_counts = train_labels.t() @ train_labels                         # (C, C) 共现次数
    p_i = (co_counts.diag() + eps) / (N + eps * C)                      # (C,)
    p_ij = (co_counts + eps) / (N + eps * C)                            # (C, C)

    # PMI = log( p_ij / (p_i p_j) )
    denom = p_i.unsqueeze(1) * p_i.unsqueeze(0)
    pmi = torch.log(torch.clamp(p_ij / torch.clamp(denom, min=1e-12), min=1e-12))

    # 只保留正 PMI（代表正相关），负的置零
    pmi = torch.clamp(pmi, min=0.0)

    # 归一化到 [0,1]（可选）
    if normalize_to_01:
        maxv = torch.max(pmi)
        if maxv > 0:
            pmi = pmi / maxv

    pmi.fill_diagonal_(1.0)
    return pmi

@torch.no_grad()
def fuse_semantic_matrices(S_text, S_pmi=None, alpha=0.5):
    """
    S = alpha * S_text + (1 - alpha) * S_pmi
    若没有 PMI，就直接返回 S_text
    """
    if S_pmi is None:
        S = S_text.clone()
    else:
        S = alpha * S_text + (1.0 - alpha) * S_pmi
    S = torch.clamp(S, 0.0, 1.0)
    S.fill_diagonal_(1.0)
    return S