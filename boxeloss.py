import torch


class BoxELoss():
    """
    Callable that will either perform uniform or self-adversarial loss, depending on the setting in @:param options
    """
    def __init__(self, args, device='cpu'):
        self.use_time_reg = args.use_time_reg
        self.time_reg_weight = args.time_reg_weight
        if args.loss_type in ['uniform', 'u']:
            self.loss_fn = uniform_loss
            self.fn_kwargs = {'gamma': args.margin, 'w': 1.0 / args.num_negative_samples}
        elif args.loss_type in ['adversarial', 'self-adversarial', 'self adversarial', 'a']:
            self.loss_fn = adversarial_loss
            self.fn_kwargs = {'gamma': args.margin, 'alpha': args.adversarial_temp}
        elif args.loss_type in ['cross entropy', 'cross-entropy', 'ce']:
            self.loss_fn = cross_entropy_loss
            self.fn_kwargs = {'ce_loss': torch.nn.CrossEntropyLoss(),
                              'device': device}

    def __call__(self, positive_tuples, negative_tuples, time_bumps=None):
        l = self.loss_fn(positive_tuples, negative_tuples, **self.fn_kwargs)
        if not self.use_time_reg:
            return l
        else:
            return l + self.time_reg_weight * time_reg(time_bumps)


def dist(entity_emb, boxes):
    """
     assumes box is tensor of shape (nb_examples, batch_size, arity, 2, embedding_dim)
     nb_examples is relevant for negative samples; for positive examples it is 1
     so it contains multiple boxes, where each box has lower and upper boundaries in embedding_dim dimensions
     e.g box[0, n, 0, :] is the lower boundary of the n-th box
     entities are of shape (nb_examples, batch_size, arity, embedding_dim)
    """

    ub = boxes[:, :, :, 0, :]  # upper boundaries
    lb = boxes[:, :, :, 1, :]  # lower boundaries
    c = (lb + ub) / 2  # centres
    w = ub - lb + 1  # widths
    k = 0.5 * (w - 1) * (w - (1 / w))
    d = torch.where(torch.logical_and(torch.ge(entity_emb, lb), torch.le(entity_emb, ub)),
                    torch.abs(entity_emb - c) / w,
                    torch.abs(entity_emb - c) * w - k)
    return d


def score(entities, relations, times, order=2, time_weight=0.5):
    d_r = dist(entities, relations).norm(dim=3, p=order).sum(dim=2)
    if times is not None:
        d_t = dist(entities, times).norm(dim=3, p=order).sum(dim=2)
        return time_weight * d_t + (1 - time_weight) * d_r
    else:
        return d_r


def uniform_loss(positives, negatives, gamma, w):
    """
    Calculates uniform negative sampling loss as presented in RotatE, Sun et. al.
    @:param positives tuple (entities, relations, times), for details see return of model.forward
    @:param negatives tuple (entities, relations, times), for details see return of model.forward_negatives
    @:param gamma loss margin
    @:param w hyperparameter, corresponds to 1/k in RotatE paper
    @:param ignore_time if True, then time information is ignored and standard BoxE is executed
    """
    eps = torch.finfo(torch.float32).tiny
    s1 = - torch.log(torch.sigmoid(gamma - score(*positives)) + eps)
    s2 = torch.sum(w * torch.log(torch.sigmoid(score(*negatives) - gamma) + eps), dim=0)
    return torch.sum(s1 - s2)


def triple_probs(negative_triples, alpha):
    scores = ((1 / score(*negative_triples)) * alpha).exp()
    div = scores.sum(dim=0)
    return scores / div


def adversarial_loss(positive_triple, negative_triples, gamma, alpha):
    """
    Calculates self-adversarial negative sampling loss as presented in RotatE, Sun et. al.
    @:param positive_triple tuple (entities, relations, times), for details see return of model.forward
    @:param negative_triple tuple (entities, relations, times), for details see return of model.forward_negatives
    @:param gamma loss margin
    @:param alpha hyperparameter, see RotatE paper
    @:param ignore_time if True, then time information is ignored and standard BoxE is executed
    """
    triple_weights = triple_probs(negative_triples, alpha)
    return uniform_loss(positive_triple, negative_triples, gamma, triple_weights)


def cross_entropy_loss(positive_triple, negative_triples, ce_loss, device='cpu'):
    pos_scores = score(*positive_triple)
    neg_scores = score(*negative_triples)
    combined_inv_scores = torch.cat((-pos_scores, -neg_scores), dim=0).t()
    target = torch.zeros((combined_inv_scores.shape[0]), dtype=torch.long, device=device)
    return ce_loss(combined_inv_scores, target)


def time_reg(time_bumps, norm_ord=4):
    # time_bump.shape = (max_time, nb_timebumps, embedding_dim)
    max_time, nb_timebumps, embedding_dim = time_bumps.shape
    diffs = []
    for time in range(max_time - 1):
        diffs.append(time_bumps[time + 1] - time_bumps[time])
    diffs = torch.stack(diffs)
    return (torch.linalg.norm(diffs, ord=norm_ord, dim=2) ** norm_ord).mean()
