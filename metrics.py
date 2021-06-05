import torch
from boxeloss import score


def rank(positive_embs, negative_embs, filter_idx, ignore_time=False):
    """
    Ranks embedded facts against provided negative fact embeddings
    :param filter_idx: indicates which of the negative embeddings are actually true and should not be considered
    """
    scores = score(*positive_embs, ignore_time)
    counterscores = score(*negative_embs, ignore_time)
    return ((scores > counterscores) * filter_idx).sum(dim=0) + 1  # TODO is the +1 needed to meet to def of rank?


def mean_rank(positive_embs, head_c_embs=None, tail_c_embs=None, filter_head=None, filter_tail=None, ranks_head=None, ranks_tail=None, ignore_time=False):
    """
    @:param ranks_head and @:param ranks_tail can be passed to the function to avoid recalculating the rank for each metric.
        If None, ranks are calculated internally.
    """
    if ranks_head is None:
        if head_c_embs is None or filter_head is None:
            raise ValueError("If parameter 'ranks_head' is not specified, 'head_c_embs' and 'filter_head' must be specified.")
        ranks_head = rank(positive_embs, head_c_embs, filter_head, ignore_time)
    if ranks_tail is None:
        if head_c_embs is None or filter_head is None:
            raise ValueError("If parameter 'ranks_tail' is not specified, 'tail_c_embs' and 'filter_tail' must be specified.")
        ranks_tail = rank(positive_embs, tail_c_embs, filter_tail, ignore_time)
    batch_size = (positive_embs[0].shape)[1]
    return torch.sum(ranks_head + ranks_tail) / (2 * batch_size)


def mean_rec_rank(positive_embs, head_c_embs=None, tail_c_embs=None, filter_head=None, filter_tail=None, ranks_head=None, ranks_tail=None, ignore_time=False):
    """
    @:param ranks and @:param ranks_tail can be passed to the function to avoid recalculating the rank for each metric.
        If None, ranks are calculated internally.
    """
    if ranks_head is None:
        if head_c_embs is None or filter_head is None:
            raise ValueError("If parameter 'ranks_head' is not specified, 'head_c_embs' and 'filter_head' must be specified.")
        ranks_head = rank(positive_embs, head_c_embs, filter_head, ignore_time)
    if ranks_tail is None:
        if ranks_tail is None:
            if head_c_embs is None or filter_head is None:
                raise ValueError(
                    "If parameter 'ranks_tail' is not specified, 'tail_c_embs' and 'filter_tail' must be specified.")
        ranks_tail = rank(positive_embs, tail_c_embs, filter_tail, ignore_time)
    batch_size = (positive_embs[0].shape)[1]
    return torch.sum(1 / ranks_head + 1 / ranks_tail) / (2 * batch_size)


def hits_at_k(positive_embs, k, head_c_embs=None, tail_c_embs=None, filter_head=None, filter_tail=None, ranks_head=None, ranks_tail=None, ignore_time=False):
    """
    @:param ranks and @:param ranks_tail can be passed to the function to avoid recalculating the rank for each metric.
        If None, ranks are calculated internally.
    """
    if ranks_head is None:
        if head_c_embs is None or filter_head is None:
            raise ValueError("If parameter 'ranks_head' is not specified, 'head_c_embs' and 'filter_head' must be specified.")
        ranks_head = rank(positive_embs, head_c_embs, filter_head, ignore_time)
    if ranks_tail is None:
        if ranks_tail is None:
            if head_c_embs is None or filter_head is None:
                raise ValueError(
                    "If parameter 'ranks_tail' is not specified, 'tail_c_embs' and 'filter_tail' must be specified.")
        ranks_tail = rank(positive_embs, tail_c_embs, filter_tail, ignore_time)
    batch_size = (positive_embs[0].shape)[1]
    return (torch.sum(ranks_head <= k) + torch.sum(ranks_tail <= k)) / (2 * batch_size)