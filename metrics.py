import torch
from  model import score


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def retrieval_metrics(positive_embs, head_c_embs, tail_c_embs, filter_head, filter_tail, binscore_fn):
    p_binary_scores = binscore_fn(*positive_embs)
    n_binary_scores = []
    hrh, hth, heh, het, htime_h, htime_t = head_c_embs
    trh, tth, teh, tet, ttime_h, ttime_t = tail_c_embs
    for i in range(len(hrh)):
        n_binary_scores.append(torch.masked_select(binscore_fn(hrh[i], hrh[i], heh[i], het[i], htime_h[i], htime_t[i]),
                                                   mask=filter_head[i, :] == 1))
        n_binary_scores.append(torch.masked_select(binscore_fn(trh[i], trh[i], teh[i], tet[i], ttime_h[i], ttime_t[i]),
                                                   mask=filter_tail[i, :] == 1))
    n_binary_scores = torch.cat(n_binary_scores)

    tp = torch.sum(p_binary_scores)  # true positives
    fn = len(p_binary_scores) - tp  # false negatives
    fp = torch.sum(n_binary_scores)  # false positves
    tn = len(n_binary_scores) - fp  # true negatives
    return tp, tn, fp, fn


def rank(positive_embs, negative_embs, filter_idx):
    scores = score(*positive_embs)
    counterscores = score(*negative_embs)
    return ((scores > counterscores) * filter_idx).sum(dim=0) + 1  # TODO is the +1 needed to meet to def of rank?


def mean_rank(positive_embs, head_c_embs, tail_c_embs, filter_head, filter_tail):
    batch_size = (positive_embs[0].shape)[1]
    return torch.sum(rank(positive_embs, head_c_embs, filter_head) + rank(positive_embs, tail_c_embs, filter_tail)) / (
                2 * batch_size)


def mean_rec_rank(positive_embs, head_c_embs, tail_c_embs, filter_head, filter_tail):
    batch_size = (positive_embs[0].shape)[1]
    return torch.sum(
        1 / rank(positive_embs, head_c_embs, filter_head) + 1 / rank(positive_embs, tail_c_embs, filter_tail)) / (
                       2 * batch_size)


def hits_at_k(positive_embs, head_c_embs, tail_c_embs, filter_head, filter_tail, k):
    batch_size = (positive_embs[0].shape)[1]
    return (torch.sum(rank(positive_embs, head_c_embs, filter_head) <= k) + torch.sum(
        rank(positive_embs, tail_c_embs, filter_tail) <= k)) / (2 * batch_size)