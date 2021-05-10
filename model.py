import torch
import torch.nn as nn
import copy

'''
BoxE model extended with boxes for timestamps.
Can do interpolation completion on TKGs.
'''
class BoxTEmp():

    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, weight_init='u', device='cpu'):
        if weight_init == 'u':
            init_f = torch.nn.init.uniform_
            init_args = (0, 0.5)
        elif weight_init == 'n':
            init_f = torch.nn.init.normal_
            init_args = (0, 0.2)
        else:
            raise ValueError("Invalid value for argument 'weight_init'. Use 'u' for uniform or 'n' for normal weight initialization.")
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
        nb_examples, _, batch_size = tuples.shape
        e_h_idx = self.get_e_idx_by_id(tuples[:, 0]).to(self.device)
        rel_idx = self.get_r_idx_by_id(tuples[:, 1]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[:, 2]).to(self.device)
        time_idx = tuples[:, 3]

        # (almost?) all of the below could be one tensor...
        r_head_boxes = self.r_head_boxes(rel_idx).view((nb_examples, batch_size, 2, self.embedding_dim))  # dim 2 distinguishes between upper and lower boundaries
        r_tail_boxes = self.r_tail_boxes(rel_idx).view((nb_examples, batch_size, 2, self.embedding_dim))
        head_bases = self.entity_bases(e_h_idx)
        head_bumps = self.entity_bumps(e_h_idx)
        tail_bases = self.entity_bases(e_t_idx)
        tail_bumps = self.entity_bumps(e_t_idx)
        time_head_boxes = self.time_head_boxes(time_idx).view((nb_examples, batch_size, 2, self.embedding_dim))
        time_tail_boxes = self.time_tail_boxes(time_idx).view((nb_examples, batch_size, 2, self.embedding_dim))
        return torch.stack((head_bases + tail_bumps, tail_bases + head_bumps), dim=2),\
               torch.stack((r_head_boxes, r_tail_boxes), dim=2),\
               torch.stack((time_head_boxes, time_tail_boxes), dim=2)

    '''
    @:return tuple (entities, relations, times) containing embeddings with
        entities.shape = (nb_negative_samples, batch_size, arity, embedding_dim)
        relations.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
        times.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
    '''
    def forward_negatives(self, negatives):
        return self.compute_embeddings(negatives)

    def forward_positives(self, positives):
        return self.compute_embeddings(positives)

    '''
    @:param positives tensor containing id's for entities, relations and times of shape (1, 4, batch_size)
        and where dim 1 indicates 0 -> head, 1 -> relation, 2 -> tail, 3 -> time
    @:param negatives tensor containing id's for entities, relations and times of shape (nb_negative_samples, 4, batch_size)
        and where dim 1 indicates 0 -> head, 1 -> relation, 2 -> tail, 3 -> time
    @:return tuple ((p_entities, p_relations, p_times), (n_entities, n_relations, n_times)) with
        p_entities.shape = (1, batch_size, arity, embedding_dim)
        p_relations.shape = (1, batch_size, arity, 2, embedding_dim)
        p_times.shape = (1, batch_size, arity, 2, embedding_dim)
        n_entities.shape = (nb_negative_samples, batch_size, arity, embedding_dim)
        n_relations.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
        n_times.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
    '''
    def forward(self, positives, negatives):
        positive_emb = self.forward_positives(positives)
        negative_emb = self.forward_negatives(negatives)
        return positive_emb, negative_emb


'''
Extension of the base BoxTEmp model, where time boxes are approximated by MLP.
Enables extrapolation on TKGs.
'''
class BoxTEmpMLP():

    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, weight_init='u', nn_depth=3, nn_width=300, lookback=1, device='cpu'):
        if weight_init == 'u':
            init_f = torch.nn.init.uniform_
            init_args = (0, 0.5)
        elif weight_init == 'n':
            init_f = torch.nn.init.normal_
            init_args = (0, 0.2)
        else:
            raise ValueError("Invalid value for argument 'weight_init'. Use 'u' for uniform or 'n' for normal weight initialization.")
        self.device = device
        self.lookback = lookback
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
        # initialize model parameters
        self.entity_bases = nn.Embedding(self.nb_entities, embedding_dim)
        self.entity_bumps = nn.Embedding(self.nb_entities, embedding_dim)
        self.r_head_boxes = nn.Embedding(self.nb_relations, 2 * embedding_dim)
        self.r_tail_boxes = nn.Embedding(self.nb_relations, 2 * embedding_dim)
        self.initial_time_head_boxes = nn.Embedding(self.lookback, 2 * embedding_dim)  # lower and upper boundaries, therefore 2*embedding_dim
        self.initial_time_tail_boxes = nn.Embedding(self.lookback, 2 * embedding_dim)
        init_f(self.entity_bases.weight, *init_args)
        init_f(self.entity_bumps.weight, *init_args)
        init_f(self.r_head_boxes.weight, *init_args)
        init_f(self.r_tail_boxes.weight, *init_args)
        init_f(self.initial_time_head_boxes.weight, *init_args)
        init_f(self.initial_time_tail_boxes.weight, *init_args)
        mlp_layers = [nn.Linear(4*self.embedding_dim*lookback, nn_width), nn.ReLU()]  # 4* because of lower/upper and head/tail
        for i in range(nn_depth):
            mlp_layers.append(nn.Linear(nn_width, nn_width))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(nn_width, 4*self.embedding_dim))
        self.time_transition = nn.Sequential(*mlp_layers)
        self.to(device)

    def __call__(self, positives, negatives):
        return self.forward(positives, negatives)

    def params(self):
        return [self.r_head_boxes.weight, self.r_tail_boxes.weight, self.entity_bases.weight,
                self.entity_bumps.weight, self.initial_time_head_boxes.weight, self.initial_time_tail_boxes.weight]\
               + list(self.time_transition.parameters())

    def to(self, device):
        self.device = device
        self.r_head_boxes = self.r_head_boxes.to(device)
        self.r_tail_boxes = self.r_tail_boxes.to(device)
        self.entity_bases = self.entity_bases.to(device)
        self.entity_bumps = self.entity_bumps.to(device)
        self.time_transition = self.time_transition.to(device)
        return self

    def state_dict(self):
        return {'r head box': self.r_head_boxes.state_dict(), 'r tail box': self.r_tail_boxes.state_dict(),
                'entity bases': self.entity_bases.state_dict(), 'entity bumps': self.entity_bumps.state_dict(),
                'init time head': self.initial_time_head_boxes.state_dict(),
                'init time tail': self.initial_time_tail_boxes.state_dict(),
                'time transition': self.time_transition.state_dict()}

    def load_state_dict(self, state_dict):
        self.r_head_boxes.load_state_dict(state_dict['r head box'])
        self.r_tail_boxes.load_state_dict(state_dict['r tail box'])
        self.entity_bases.load_state_dict(state_dict['entity bases'])
        self.entity_bumps.load_state_dict(state_dict['entity bumps'])
        self.initial_time_head_boxes.load_state_dict(state_dict['init time head'])
        self.initial_time_tail_boxes.load_state_dict(state_dict['init time tail'])
        self.time_transition.load_state_dict(state_dict['time transition'])
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

    def unroll_time(self):
        initial_times = torch.arange(0, self.lookback)
        init_head_boxes = self.initial_time_head_boxes(initial_times)
        init_tail_boxes = self.initial_time_tail_boxes(initial_times)
        current_state = torch.stack((init_head_boxes, init_tail_boxes), dim=1).flatten()
        time_head_boxes, time_tail_boxes = [], []
        for t in range(self.max_time):
            next_time = self.time_transition(current_state)
            time_head_boxes.append(next_time[:2*self.embedding_dim])
            time_tail_boxes.append(next_time[2*self.embedding_dim:])
            current_state = torch.cat((current_state[4*self.embedding_dim:], next_time))  # cut off upper/lower and head/
        return nn.Embedding.from_pretrained(torch.cat((init_head_boxes, torch.stack(time_head_boxes)))),\
               nn.Embedding.from_pretrained(torch.cat((init_tail_boxes, torch.stack(time_tail_boxes))))

    def compute_embeddings(self, tuples):
        nb_examples, _, batch_size = tuples.shape
        e_h_idx = self.get_e_idx_by_id(tuples[:, 0]).to(self.device)
        rel_idx = self.get_r_idx_by_id(tuples[:, 1]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[:, 2]).to(self.device)
        time_idx = tuples[:, 3]

        r_head_boxes = self.r_head_boxes(rel_idx).view((nb_examples, batch_size, 2, self.embedding_dim))  # dim 2 distinguishes between upper and lower boundaries
        r_tail_boxes = self.r_tail_boxes(rel_idx).view((nb_examples, batch_size, 2, self.embedding_dim))
        head_bases = self.entity_bases(e_h_idx)
        head_bumps = self.entity_bumps(e_h_idx)
        tail_bases = self.entity_bases(e_t_idx)
        tail_bumps = self.entity_bumps(e_t_idx)
        all_time_head_boxes, all_time_tail_boxes = self.unroll_time()

        time_head_boxes = all_time_head_boxes(time_idx).view((nb_examples, batch_size, 2, self.embedding_dim))
        time_tail_boxes = all_time_tail_boxes(time_idx).view((nb_examples, batch_size, 2, self.embedding_dim))
        return torch.stack((head_bases + tail_bumps, tail_bases + head_bumps), dim=2),\
               torch.stack((r_head_boxes, r_tail_boxes), dim=2),\
               torch.stack((time_head_boxes, time_tail_boxes), dim=2)

    '''
    @:return tuple (entities, relations, times) containing embeddings with
        entities.shape = (nb_negative_samples, batch_size, arity, embedding_dim)
        relations.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
        times.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
    '''
    def forward_negatives(self, negatives):
        return self.compute_embeddings(negatives)

    def forward_positives(self, positives):
        return self.compute_embeddings(positives)

    '''
    @:param positives tensor containing id's for entities, relations and times of shape (1, 4, batch_size)
        and where dim 1 indicates 0 -> head, 1 -> relation, 2 -> tail, 3 -> time
    @:param negatives tensor containing id's for entities, relations and times of shape (nb_negative_samples, 4, batch_size)
        and where dim 1 indicates 0 -> head, 1 -> relation, 2 -> tail, 3 -> time
    @:return tuple ((p_entities, p_relations, p_times), (n_entities, n_relations, n_times)) with
        p_entities.shape = (1, batch_size, arity, embedding_dim)
        p_relations.shape = (1, batch_size, arity, 2, embedding_dim)
        p_times.shape = (1, batch_size, arity, 2, embedding_dim)
        n_entities.shape = (nb_negative_samples, batch_size, arity, embedding_dim)
        n_relations.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
        n_times.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
    '''
    def forward(self, positives, negatives):
        positive_emb = self.forward_positives(positives)
        negative_emb = self.forward_negatives(negatives)
        return positive_emb, negative_emb


'''
Callable that will either perform uniform or self-adversarial loss, depending on the setting in @:param options
'''
class BoxELoss():
    def __init__(self, options):
        if options.loss_type in ['uniform', 'u']:
            self.loss_fn = uniform_loss
            self.fn_kwargs = {'gamma': options.margin, 'w': 1.0 / options.loss_k, 'ignore_time': options.ignore_time}
        elif options.loss_type in ['adversarial', 'self-adversarial', 'self adversarial', 'a']:
            self.loss_fn = adversarial_loss_old
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
    k = 0.5 * (w - 1) * (w - 1 / w)
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


'''
Calculates uniform negative sampling loss as presented in RotatE, Sun et. al.
@:param positives tuple (entities, relations, times), for details see return of model.forward
@:param negatives tuple (entities, relations, times), for details see return of model.forward_negatives
@:param gamma loss margin
@:param w hyperparameter, corresponds to 1/k in RotatE paper
@:param ignore_time if True, then time information is ignored and standard BoxE is executed
'''
def uniform_loss(positives, negatives, gamma, w, ignore_time=False):
    s1 = - torch.log(torch.sigmoid(gamma - score(*positives, ignore_time=ignore_time)))
    s2 = torch.sum(w * torch.log(torch.sigmoid(score(*negatives, ignore_time=ignore_time) - gamma)), dim=0)
    return torch.mean(s1 - s2)


def triple_probs(negative_triples, alpha):
    scores = (score(*negative_triples) * alpha).exp()
    div = scores.sum()
    return scores / div

'''
Calculates self-adversarial negative sampling loss as presented in RotatE, Sun et. al.
@:param positive_triple tuple (entities, relations, times), for details see return of model.forward
@:param negative_triple tuple (entities, relations, times), for details see return of model.forward_negatives
@:param gamma loss margin
@:param alpha hyperparameter, see RotatE paper
@:param ignore_time if True, then time information is ignored and standard BoxE is executed
'''
def adversarial_loss_old(positive_triple, negative_triples, gamma, alpha, ignore_time=False):
    triple_weights = triple_probs(negative_triples, alpha)
    return uniform_loss(positive_triple, negative_triples, gamma, triple_weights, ignore_time=ignore_time)