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


'''
@:param ranks_head and @:param ranks_tail can be passed to the function to avoid recalculating the rank for each metric.
    If None, ranks are calculated internally.
'''
def mean_rank(positive_embs, head_c_embs=None, tail_c_embs=None, filter_head=None, filter_tail=None, ranks_head=None, ranks_tail=None):
    if ranks_head is None:
        if head_c_embs is None or filter_head is None:
            raise ValueError("If parameter 'ranks_head' is not specified, 'head_c_embs' and 'filter_head' must be specified.")
        ranks_head = rank(positive_embs, head_c_embs, filter_head)
    if ranks_tail is None:
        if head_c_embs is None or filter_head is None:
            raise ValueError("If parameter 'ranks_tail' is not specified, 'tail_c_embs' and 'filter_tail' must be specified.")
        ranks_tail = rank(positive_embs, tail_c_embs, filter_tail)
    batch_size = (positive_embs[0].shape)[1]
    return torch.sum(ranks_head + ranks_tail) / (2 * batch_size)


'''
@:param ranks and @:param ranks_tail can be passed to the function to avoid recalculating the rank for each metric.
    If None, ranks are calculated internally.
'''
def mean_rec_rank(positive_embs, head_c_embs=None, tail_c_embs=None, filter_head=None, filter_tail=None, ranks_head=None, ranks_tail=None):
    if ranks_head is None:
        if head_c_embs is None or filter_head is None:
            raise ValueError("If parameter 'ranks_head' is not specified, 'head_c_embs' and 'filter_head' must be specified.")
        ranks_head = rank(positive_embs, head_c_embs, filter_head)
    if ranks_tail is None:
        if ranks_tail is None:
            if head_c_embs is None or filter_head is None:
                raise ValueError(
                    "If parameter 'ranks_tail' is not specified, 'tail_c_embs' and 'filter_tail' must be specified.")
        ranks_tail = rank(positive_embs, tail_c_embs, filter_tail)
    batch_size = (positive_embs[0].shape)[1]
    return torch.sum(1 / ranks_head + 1 / ranks_tail) / (2 * batch_size)


'''
@:param ranks and @:param ranks_tail can be passed to the function to avoid recalculating the rank for each metric.
    If None, ranks are calculated internally.
'''
def hits_at_k(positive_embs, k, head_c_embs=None, tail_c_embs=None, filter_head=None, filter_tail=None, ranks_head=None, ranks_tail=None):
    if ranks_head is None:
        if head_c_embs is None or filter_head is None:
            raise ValueError("If parameter 'ranks_head' is not specified, 'head_c_embs' and 'filter_head' must be specified.")
        ranks_head = rank(positive_embs, head_c_embs, filter_head)
    if ranks_tail is None:
        if ranks_tail is None:
            if head_c_embs is None or filter_head is None:
                raise ValueError(
                    "If parameter 'ranks_tail' is not specified, 'tail_c_embs' and 'filter_tail' must be specified.")
        ranks_tail = rank(positive_embs, tail_c_embs, filter_tail)
    batch_size = (positive_embs[0].shape)[1]
    return (torch.sum(ranks_head <= k) + torch.sum(ranks_tail <= k)) / (2 * batch_size)