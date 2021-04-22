import torch
import torch.nn as nn


class BoxTEmp():

    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, weight_init='u', device='cpu'):
        if weight_init == 'u':
            init_f = torch.nn.init.uniform_
            init_args = (0, 0.5)
        elif weight_init == 'n':
            init_f = torch.nn.init.normal_
            init_args = (0, 0.2)
        self.device = device
        self.embedding_dim = embedding_dim
        self.relation_ids = relation_ids
        self.entity_ids = entity_ids
        self.relation_id_offset = relation_ids[0]
        self.timestamps = timestamps
        self.max_time = max(timestamps) + 1
        self.nb_relations = len(relation_ids)
        self.nb_entities = len(entity_ids)
        assert sorted(
            entity_ids + relation_ids) == entity_ids + relation_ids, 'ids need to be ascending ints from 0 with entities coming before relations'

        self.time_head_boxes = nn.Embedding(self.max_time,
                                            2 * embedding_dim)  # lower and upper boundaries, therefore 2*embedding_dim
        self.time_tail_boxes = nn.Embedding(self.max_time, 2 * embedding_dim)
        self.r_head_boxes = nn.Embedding(self.nb_relations, 2 * embedding_dim)
        self.r_tail_boxes = nn.Embedding(self.nb_relations, 2 * embedding_dim)
        self.entity_bases = nn.Embedding(self.nb_entities, embedding_dim)
        self.entity_bumps = nn.Embedding(self.nb_entities, embedding_dim)
        init_f(self.time_head_boxes.weight, *init_args)
        init_f(self.time_tail_boxes.weight, *init_args)
        init_f(self.r_head_boxes.weight, *init_args)
        init_f(self.r_tail_boxes.weight, *init_args)
        init_f(self.entity_bases.weight, *init_args)
        init_f(self.entity_bumps.weight, *init_args)

    def __call__(self, positives, negatives):
        return self.forward(positives, negatives)

    def params(self):
        return [self.r_head_boxes.weight, self.r_tail_boxes.weight, self.entity_bases.weight,
                self.entity_bumps.weight, self.time_head_boxes.weight, self.time_tail_boxes.weight]

    def to(self, device):
        self.device = device
        self.r_head_boxes = self.r_head_boxes.to(device)
        self.r_tail_boxes = self.r_tail_boxes.to(device)
        self.entity_bases = self.entity_bases.to(device)
        self.entity_bumps = self.entity_bumps.to(device)
        self.time_head_boxes = self.time_head_boxes.to(device)
        self.time_tail_boxes = self.time_tail_boxes.to(device)
        return self

    def state_dict(self):
        # TODO need to copy and detach tensors?
        return {'r head box': self.r_head_boxes.state_dict(), 'r tail box': self.r_tail_boxes.state_dict(),
                'entity bases': self.entity_bases.state_dict(), 'entity bumps': self.entity_bumps.state_dict(),
                'time head box': self.time_head_boxes.state_dict(), 'time tail box': self.time_tail_boxes.state_dict()}

    def load_state_dict(self, state_dict):
        self.r_head_boxes.load_state_dict(state_dict['r head box'])
        self.r_tail_boxes.load_state_dict(state_dict['r tail box'])
        self.entity_bases.load_state_dict(state_dict['entity bases'])
        self.entity_bumps.load_state_dict(state_dict['entity bumps'])
        self.time_head_boxes.load_state_dict(state_dict['time head box'])
        self.time_tail_boxes.load_state_dict(state_dict['time tail box'])
        return self

    def get_r_idx_by_id(self, r_ids):
        # r_names: tensor of realtion ids
        return r_ids - self.relation_id_offset

    def get_e_idx_by_id(self, e_ids):
        return e_ids

    def get_embeddings(self):
        # slightly nicer (readable) representation of params
        d = dict()
        for i, t in enumerate(self.head_boxes):
            key = self.relation_names[i] + ' head'
            d[key] = [('lower', self.head_boxes[i, 0]), ('upper', self.head_boxes[i, 1])]
        for i, t in enumerate(self.tail_boxes):
            key = self.relation_names[i] + ' tail'
            d[key] = [('lower', self.tail_boxes[i, 0]), ('upper', self.tail_boxes[i, 1])]
        for i, t in enumerate(self.entity_bases):
            key = self.entity_names[i] + ' base'
            d[key] = self.entity_bases[i]
        for i, t in enumerate(self.entity_bumps):
            key = self.entity_names[i] + ' bump'
            d[key] = self.entity_bumps[i]
        return d

    def compute_embeddings(self, tuples):
        rel_idx = self.get_r_idx_by_id(tuples[1]).to(self.device)
        e_h_idx = self.get_e_idx_by_id(tuples[0]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[2]).to(self.device)
        time_idx = tuples[3]

        # (almost?) all of the below could be one tensor...
        r_head_boxes = self.r_head_boxes(rel_idx).view(
            (len(rel_idx), 2, self.embedding_dim))  # dim 1 distinguishes between upper and lower boundries
        r_tail_boxes = self.r_tail_boxes(rel_idx).view((len(rel_idx), 2, self.embedding_dim))
        head_bases = self.entity_bases(e_h_idx)
        head_bumps = self.entity_bumps(e_h_idx)
        tail_bases = self.entity_bases(e_t_idx)
        tail_bumps = self.entity_bumps(e_t_idx)
        time_head_boxes = self.time_head_boxes(time_idx).view((len(time_idx), 2, self.embedding_dim))
        time_tail_boxes = self.time_tail_boxes(time_idx).view((len(time_idx), 2, self.embedding_dim))
        return r_head_boxes, r_tail_boxes, head_bases + tail_bumps, tail_bases + head_bumps, time_head_boxes, time_tail_boxes

    def forward_negatives(self, negatives):
        rhs = []  # relation head
        rts = []  # relation tail
        ehs = []  # entity head
        ets = []  # entity tail
        ths = []  # time head
        tts = []  # time tail
        for i, n_r in enumerate(negatives[1]):
            rh, rt, eh, et, time_h, time_t = self.compute_embeddings(
                [negatives[0, i], n_r, negatives[2, i], negatives[3, i]])
            rhs.append(rh)
            rts.append(rt)
            ehs.append(eh)
            ets.append(et)
            ths.append(time_h)
            tts.append(time_t)
        return torch.stack(rhs), torch.stack(rts), torch.stack(ehs), torch.stack(ets), torch.stack(ths), torch.stack(
            tts)

    def forward(self, positives, negatives):
        positive_emb = self.compute_embeddings(positives)
        negative_emb = self.forward_negatives(negatives)
        return positive_emb, negative_emb


class BoxELoss():
    def __init__(self, options):
        print(options.ignore_time)
        if options.loss_type in ['uniform', 'u']:
            self.loss_fn = uniform_loss
            self.fn_kwargs = {'gamma': options.margin, 'w': 1 / options.loss_k, 'ignore_time': options.ignore_time}
        elif options.loss_type in ['adversarial', 'self-adversarial', 'self adversarial', 'a']:
            self.loss_fn = adversarial_loss
            self.fn_kwargs = {'gamma': options.margin, 'alpha': options.adversarial_temp,
                              'ignore_time': options.ignore_time}

    def __call__(self, positive_tuples, negative_tuples):
        return self.loss_fn(positive_tuples, negative_tuples, **self.fn_kwargs)


class BoxEBinScore():
    def __init__(self, options):
        self.ignore_time = options.ignore_time

    def __call__(self, r_headbox, r_tailbox, e_head, e_tail, time_headbox, time_tailbox):
        return binary_score(r_headbox, r_tailbox, e_head, e_tail, time_headbox, time_tailbox,
                            ignore_time=self.ignore_time)


def dist(entity_emb, boxes):
    # assumes box is tensor of shape (batch_size, 2, embedding_dim)
    # so it contains multiple boxes, where each box has lower and upper boundries in embdding_dim dimensions
    # e.g box[n, 0, :] is the lower boundry of the n-th box

    lb = boxes[:, 0, :]  # lower boundries
    ub = boxes[:, 1, :]  # upper boundries
    c = (lb + ub) / 2  # centres
    w = ub - lb + 1  # widths
    k = 0.5 * (w - 1) * (w - 1 / w)
    d = torch.where(torch.logical_and(torch.ge(entity_emb, lb), torch.le(entity_emb, ub)),
                    torch.abs(entity_emb - c) / w,
                    torch.abs(entity_emb - c) * w - k)
    return d


def binary_dist(entity_emb, boxes):
    lb = boxes[:, 0, :]  # lower boundries
    ub = boxes[:, 1, :]  # upper boundries
    # c = (lb + ub)/2  # centres
    # w = ub - lb + 1  # widths
    # k = 0.5*(w - 1) * (w - 1/w)
    return torch.logical_and(torch.ge(entity_emb, lb), torch.le(entity_emb, ub))


def score(r_headbox, r_tailbox, e_head, e_tail, time_headbox, time_tailbox, order=2, ignore_time=False):
    # once the representaion of r is known this should probably just take r and the entities
    a = torch.norm(dist(e_head, r_headbox), p=order, dim=1)
    b = torch.norm(dist(e_tail, r_tailbox), p=order, dim=1)
    c = torch.norm(dist(e_head, time_headbox), p=order, dim=1)
    d = torch.norm(dist(e_tail, time_tailbox), p=order, dim=1)
    if ignore_time:
        c = 0
        d = 0
    return a + b + c + d


def binary_score(r_headbox, r_tailbox, e_head, e_tail, time_headbox, time_tailbox, ignore_time=False):
    a = torch.all(binary_dist(e_head, r_headbox), dim=1)
    b = torch.all(binary_dist(e_tail, r_tailbox), dim=1)
    c = torch.all(binary_dist(e_head, time_headbox), dim=1)
    d = torch.all(binary_dist(e_tail, time_tailbox), dim=1)
    if ignore_time:
        c = torch.ones_like(c)
        d = torch.ones_like(d)
    return torch.logical_and(a, torch.logical_and(b, torch.logical_and(c, d)))


def uniform_loss(positive_tuple, negative_tuples, gamma, w, ignore_time=False):
    # An example of the following description can be found in our google doc (https://docs.google.com/document/d/1XltuO8IeyYSb8gLNA0lO8nLOQSqhe-Ub5gOnIc0-89w/edit?usp=sharing)
    # If there is a more natural way to represent this data, please let me know :)
    # positive triple: tuple of form (head_boxes, tail_boxes, head_entities, tail_entities)
    #     head_boxes: tensor of shape (batch_size, 2, embdding_dims), to represent lower and upper boundries
    #     tail_boxes: same as head_boxes
    #     head_entities: tensor of shape (batch_size, embedding_dims), so one tensor for each head entity
    #     tail_entities: same as head_entities
    # negative_triples: tuple of form (n_head_boxes, n_tail_boxes, n_head_entities, n_tail_entities)
    #     n_head_boxes: tensor of shape (nb_samples, batch_size, 2, embdding_dims), so each positive triple has nb_samples negative samples associated with it
    #     n_tail_boxes: same as head_boxes
    #     n_head_entities: tensor of shape (nb_samples, batch_size, embedding_dims)
    #     n_tail_entities: same as head_entities
    # w == 1/k (see RotatE-paper)
    # headbox, tailbox, e_head, e_tail = positive_triple

    s1 = - torch.log(torch.sigmoid(gamma - score(*positive_tuple, ignore_time=ignore_time)))
    s2_terms = []
    n_head_boxes, n_tail_boxes, n_head_e, n_tail_e, n_time_headboxes, n_time_tailboxes = negative_tuples
    if not torch.is_tensor(w):
        w = torch.tensor([w]).repeat(len(n_head_boxes))
    for i in range(len(n_head_boxes)):
        s2_terms.append(w[i] * torch.log(torch.sigmoid(
            score(n_head_boxes[i], n_tail_boxes[i], n_head_e[i], n_tail_e[i], n_time_headboxes[i], n_time_tailboxes[i],
                  ignore_time=ignore_time) - gamma)))
    s2 = torch.sum(torch.stack(s2_terms), dim=1)
    return torch.mean(s1 - s2)


def triple_probs(negative_triples, alpha):
    n_head_boxes, n_tail_boxes, n_head_e, n_tail_e, n_time_head, n_time_tail = negative_triples
    scores = []
    for i in range(len(n_head_boxes)):
        scores.append(torch.exp(
            alpha * score(n_head_boxes[i], n_tail_boxes[i], n_head_e[i], n_tail_e[i], n_time_head[i], n_time_tail[i])))
    scores = torch.stack(scores)
    div = torch.repeat_interleave(torch.sum(scores, dim=1).unsqueeze(1), repeats=scores.shape[1], dim=1)
    return torch.div(scores, div)


def adversarial_loss(positive_triple, negative_triples, gamma, alpha, ignore_time=False):
    triple_weights = triple_probs(negative_triples, alpha)
    return uniform_loss(positive_triple, negative_triples, gamma, triple_weights, ignore_time=ignore_time)