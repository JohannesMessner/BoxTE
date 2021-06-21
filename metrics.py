import torch
from boxeloss import score


def rank(positive_embs, negative_embs, filter_idx):
    """
    Ranks embedded facts against provided negative fact embeddings
    :param filter_idx: indicates which of the negative embeddings are actually true and should not be considered
    """
    scores = score(*positive_embs)
    counterscores = score(*negative_embs)
    return ((scores > counterscores) * filter_idx).sum(dim=0) + 1


def mean_rank(ranks_head, ranks_tail):
    return torch.sum(ranks_head + ranks_tail) / (2 * len(ranks_head))


def mean_rec_rank(ranks_head, ranks_tail):
    return torch.sum(1 / ranks_head + 1 / ranks_tail) / (2 * len(ranks_head))


def hits_at_k(ranks_head, ranks_tail, k):
    return (torch.sum(ranks_head <= k) + torch.sum(ranks_tail <= k)) / (2 * len(ranks_head))