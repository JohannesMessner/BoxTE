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
    rh, th, eh, et, time_h, time_t = negative_embs
    ranks = []
    for i, s in enumerate(scores):
        counterscores = score(rh[i], th[i], eh[i], et[i], time_h[i], time_t[i])
        r = 1
        for i_c, cs in enumerate(counterscores):
            if cs < s and filter_idx[i, i_c]:  # TODO > or < ?
                r += 1
        ranks.append(r)
    return torch.tensor(ranks)


def mean_rank(positive_embs, head_c_embs, tail_c_embs, filter_head, filter_tail):
    return torch.sum(rank(positive_embs, head_c_embs, filter_head) + rank(positive_embs, tail_c_embs, filter_tail)) / (
                2 * len(positive_embs[0]))


def mean_rec_rank(positive_embs, head_c_embs, tail_c_embs, filter_head, filter_tail):
    return torch.sum(
        1 / rank(positive_embs, head_c_embs, filter_head) + 1 / rank(positive_embs, tail_c_embs, filter_tail)) / (
                       2 * len(positive_embs[0]))


def hits_at_k(positive_embs, head_c_embs, tail_c_embs, filter_head, filter_tail, k):
    return (torch.sum(rank(positive_embs, head_c_embs, filter_head) <= k) + torch.sum(
        rank(positive_embs, tail_c_embs, filter_tail) <= k)) / (2 * len(positive_embs[0]))