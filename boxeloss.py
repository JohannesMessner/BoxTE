import torch

class BoxELoss():
    """
    Callable that will either perform uniform or self-adversarial loss, depending on the setting in @:param options
    """
    def __init__(self, args):
        if args.loss_type in ['uniform', 'u']:
            self.loss_fn = uniform_loss
            self.fn_kwargs = {'gamma': args.margin, 'w': 1.0 / args.num_negative_samples, 'ignore_time': args.ignore_time}
        elif args.loss_type in ['adversarial', 'self-adversarial', 'self adversarial', 'a']:
            self.loss_fn = adversarial_loss
            self.fn_kwargs = {'gamma': args.margin, 'alpha': args.adversarial_temp,
                              'ignore_time': args.ignore_time}

    def __call__(self, positive_tuples, negative_tuples):
        return self.loss_fn(positive_tuples, negative_tuples, **self.fn_kwargs)


class BoxEBinScore():
    def __init__(self, options):
        self.ignore_time = options.ignore_time

    def __call__(self, r_headbox, r_tailbox, e_head, e_tail, time_headbox, time_tailbox):
        return binary_score(r_headbox, r_tailbox, e_head, e_tail, time_headbox, time_tailbox,
                            ignore_time=self.ignore_time)


def binary_dist(entity_emb, boxes):
    lb = boxes[:, 0, :]  # lower boundries
    ub = boxes[:, 1, :]  # upper boundries
    # c = (lb + ub)/2  # centres
    # w = ub - lb + 1  # widths
    # k = 0.5*(w - 1) * (w - 1/w)
    return torch.logical_and(torch.ge(entity_emb, lb), torch.le(entity_emb, ub))


def dist(entity_emb, boxes):
    # assumes box is tensor of shape (nb_examples, batch_size, arity, 2, embedding_dim)
    # nb_examples is relevant for negative samples; for positive examples it is 1
    # so it contains multiple boxes, where each box has lower and upper boundries in embdding_dim dimensions
    # e.g box[0, n, 0, :] is the lower boundry of the n-th box
    #
    # entities are of shape (nb_examples, batch_size, arity, embedding_dim)

    lb = boxes[:, :, :, 0, :]  # lower boundaries
    ub = boxes[:, :, :, 1, :]  # upper boundaries
    c = (lb + ub) / 2  # centres
    w = ub - lb + 1  # widths
    k = 0.5 * (w - 1) * (w - (1 / w))
    d = torch.where(torch.logical_and(torch.ge(entity_emb, lb), torch.le(entity_emb, ub)),
                    torch.abs(entity_emb - c) / w,
                    torch.abs(entity_emb - c) * w - k)
    return d


def score(entities, relations, times, ignore_time=False, order=2, time_weight=0.5):
    d_r = dist(entities, relations).norm(dim=3, p=order).sum(dim=2)
    if not ignore_time:
        d_t = dist(entities, times).norm(dim=3, p=order).sum(dim=2)
        return time_weight * d_t + (1 - time_weight) * d_r
    else:
        return d_r


def binary_score(r_headbox, r_tailbox, e_head, e_tail, time_headbox, time_tailbox, ignore_time=False):
    a = torch.all(binary_dist(e_head, r_headbox), dim=1)
    b = torch.all(binary_dist(e_tail, r_tailbox), dim=1)
    c = torch.all(binary_dist(e_head, time_headbox), dim=1)
    d = torch.all(binary_dist(e_tail, time_tailbox), dim=1)
    if ignore_time:
        c = torch.ones_like(c)
        d = torch.ones_like(d)
    return torch.logical_and(a, torch.logical_and(b, torch.logical_and(c, d)))


def uniform_loss(positives, negatives, gamma, w, ignore_time=False):
    """
    Calculates uniform negative sampling loss as presented in RotatE, Sun et. al.
    @:param positives tuple (entities, relations, times), for details see return of model.forward
    @:param negatives tuple (entities, relations, times), for details see return of model.forward_negatives
    @:param gamma loss margin
    @:param w hyperparameter, corresponds to 1/k in RotatE paper
    @:param ignore_time if True, then time information is ignored and standard BoxE is executed
    """
    eps = torch.finfo(torch.float32).tiny
    s1 = - torch.log(torch.sigmoid(gamma - score(*positives, ignore_time=ignore_time)) + eps)
    s2 = torch.sum(w * torch.log(torch.sigmoid(score(*negatives, ignore_time=ignore_time) - gamma) + eps), dim=0)
    return torch.mean(s1 - s2)


def triple_probs(negative_triples, alpha, ignore_time=False):
    scores = ((1 / score(*negative_triples, ignore_time=ignore_time)) * alpha).exp()
    div = scores.sum(dim=0)
    return scores / div


def adversarial_loss(positive_triple, negative_triples, gamma, alpha, ignore_time=False):
    """
    Calculates self-adversarial negative sampling loss as presented in RotatE, Sun et. al.
    @:param positive_triple tuple (entities, relations, times), for details see return of model.forward
    @:param negative_triple tuple (entities, relations, times), for details see return of model.forward_negatives
    @:param gamma loss margin
    @:param alpha hyperparameter, see RotatE paper
    @:param ignore_time if True, then time information is ignored and standard BoxE is executed
    """
    triple_weights = triple_probs(negative_triples, alpha, ignore_time)
    return uniform_loss(positive_triple, negative_triples, gamma, triple_weights, ignore_time=ignore_time)