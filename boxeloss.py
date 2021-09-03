import torch


class BoxELoss():
    """
    Callable that will either perform uniform or self-adversarial loss, depending on the setting in @:param options
    """
    def __init__(self, args, device='cpu', timebump_shape=None):
        self.use_time_reg = args.use_time_reg
        self.use_ball_reg = args.use_ball_reg
        self.time_reg_weight = args.time_reg_weight
        self.ball_reg_weight = args.ball_reg_weight
        self.time_reg_order = args.time_reg_order
        self.ball_reg_order = args.ball_reg_order
        if args.loss_type in ['uniform', 'u']:
            self.loss_fn = uniform_loss
            self.fn_kwargs = {'gamma': args.margin, 'w': 1.0 / args.num_negative_samples}
        elif args.loss_type in ['adversarial', 'self-adversarial', 'self adversarial', 'a']:
            self.loss_fn = adversarial_loss
            self.fn_kwargs = {'gamma': args.margin, 'alpha': args.adversarial_temp, 'device': device}
        elif args.loss_type in ['cross entropy', 'cross-entropy', 'ce']:
            self.loss_fn = cross_entropy_loss
            self.fn_kwargs = {'ce_loss': torch.nn.CrossEntropyLoss(reduction=args.ce_reduction),
                              'device': device}
        if self.use_time_reg:
            if timebump_shape is None:
                raise ValueError('Time reg is enabled but timebump shape is not provided.')
            self.diff_matrix = make_diff_matrix(timebump_shape, device=device)

    def __call__(self, positive_tuples, negative_tuples, time_bumps=None):
        l = self.loss_fn(positive_tuples, negative_tuples, **self.fn_kwargs)
        if self.use_time_reg:
            l = l + self.time_reg_weight * self.time_reg(time_bumps, norm_ord=self.time_reg_order)
        if self.use_ball_reg:
            l = l + self.ball_reg_weight * self.ball_reg(entities=positive_tuples[0], relations=positive_tuples[1], norm_ord=self.ball_reg_order)
        return l

    def time_reg(self, time_bumps, norm_ord=4):
        # max_time, nb_timebumps, embedding_dim = time_bumps.shape
        time_bumps = time_bumps.transpose(0, 1)
        diffs = self.diff_matrix.matmul(time_bumps)
        return (torch.linalg.norm(diffs, ord=norm_ord, dim=2) ** norm_ord).mean()

    def ball_reg(self, entities, relations, norm_ord=4):
        heads = entities[:, :, 0, :]
        tails = entities[:, :, 1, :]
        box_centers = (relations[:, :, :, 0, :] + relations[:, :, :, 1, :]) / 2
        head_centers = box_centers[:, :, 0, :]
        tail_centers = box_centers[:, :, 1, :]
        return (torch.linalg.norm(heads, ord=norm_ord, dim=-1) ** norm_ord
                + torch.linalg.norm(tails, ord=norm_ord, dim=-1) ** norm_ord
                + torch.linalg.norm(head_centers, ord=norm_ord, dim=-1) ** norm_ord
                + torch.linalg.norm(tail_centers, ord=norm_ord, dim=-1) ** norm_ord).mean()


def make_diff_matrix(timebump_shape, device):
    (max_time, nb_timebumps, embedding_dim) = timebump_shape
    m = torch.eye(max_time, max_time, requires_grad=False, device=device)
    for i in range(m.shape[0] - 1):
        m[i, i + 1] = -1
    m = m[:-1, :]
    return m.unsqueeze(0)


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


def triple_probs(negative_triples, alpha, device='cpu'):
    eps = torch.finfo(torch.float32).eps
    pre_exp_scores = ((1 / (score(*negative_triples) + eps)) * alpha)
    pre_exp_scores = torch.minimum(pre_exp_scores, torch.tensor([85.0], device=device))  # avoid exp exploding to inf
    scores = pre_exp_scores.exp()
    div = scores.sum(dim=0) + eps
    return scores / div


def adversarial_loss(positive_triple, negative_triples, gamma, alpha, device='cpu'):
    """
    Calculates self-adversarial negative sampling loss as presented in RotatE, Sun et. al.
    @:param positive_triple tuple (entities, relations, times), for details see return of model.forward
    @:param negative_triple tuple (entities, relations, times), for details see return of model.forward_negatives
    @:param gamma loss margin
    @:param alpha hyperparameter, see RotatE paper
    @:param ignore_time if True, then time information is ignored and standard BoxE is executed
    """
    triple_weights = triple_probs(negative_triples, alpha, device)
    return uniform_loss(positive_triple, negative_triples, gamma, triple_weights)


def cross_entropy_loss(positive_triple, negative_triples, ce_loss, device='cpu'):
    pos_scores = score(*positive_triple)
    neg_scores = score(*negative_triples)
    combined_inv_scores = torch.cat((-pos_scores, -neg_scores), dim=0).t()
    target = torch.zeros((combined_inv_scores.shape[0]), dtype=torch.long, device=device)
    return ce_loss(combined_inv_scores, target)
