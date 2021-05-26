import torch
import torch.nn as nn
import copy

class BaseBoxE():
    """
        BoxE model extended with boxes for timestamps.
        Can do interpolation completion on TKGs.
        """

    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, weight_init='u', device='cpu',
                 weight_init_args=(0, 1)):
        if weight_init == 'u':
            self.init_f = torch.nn.init.uniform_
        elif weight_init == 'n':
            self.init_f = torch.nn.init.normal_
        else:
            raise ValueError(
                "Invalid value for argument 'weight_init'. Use 'u' for uniform or 'n' for normal weight initialization.")
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

        self.r_head_boxes = nn.Embedding(self.nb_relations, 2 * embedding_dim)
        self.r_tail_boxes = nn.Embedding(self.nb_relations, 2 * embedding_dim)
        self.entity_bases = nn.Embedding(self.nb_entities, embedding_dim)
        self.entity_bumps = nn.Embedding(self.nb_entities, embedding_dim)
        self.init_f(self.r_head_boxes.weight, *weight_init_args)
        self.init_f(self.r_tail_boxes.weight, *weight_init_args)
        self.init_f(self.entity_bases.weight, *weight_init_args)
        self.init_f(self.entity_bumps.weight, *weight_init_args)

    def __call__(self, positives, negatives):
        return self.forward(positives, negatives)

    def params(self):
        return [self.r_head_boxes.weight, self.r_tail_boxes.weight, self.entity_bases.weight, self.entity_bumps.weight]

    def to(self, device):
        self.device = device
        self.r_head_boxes = self.r_head_boxes.to(device)
        self.r_tail_boxes = self.r_tail_boxes.to(device)
        self.entity_bases = self.entity_bases.to(device)
        self.entity_bumps = self.entity_bumps.to(device)
        return self

    def state_dict(self):
        return {'r head box': self.r_head_boxes.state_dict(), 'r tail box': self.r_tail_boxes.state_dict(),
                'entity bases': self.entity_bases.state_dict(), 'entity bumps': self.entity_bumps.state_dict()}

    def load_state_dict(self, state_dict):
        self.r_head_boxes.load_state_dict(state_dict['r head box'])
        self.r_tail_boxes.load_state_dict(state_dict['r tail box'])
        self.entity_bases.load_state_dict(state_dict['entity bases'])
        self.entity_bumps.load_state_dict(state_dict['entity bumps'])
        return self

    def get_r_idx_by_id(self, r_ids):
        """@:param r_names tensor of realtion ids"""
        return r_ids - self.relation_id_offset

    def get_e_idx_by_id(self, e_ids):
        return e_ids

    def compute_embeddings(self, tuples):
        nb_examples, _, batch_size = tuples.shape
        e_h_idx = self.get_e_idx_by_id(tuples[:, 0]).to(self.device)
        rel_idx = self.get_r_idx_by_id(tuples[:, 1]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[:, 2]).to(self.device)

        # (almost?) all of the below could be one tensor...
        r_head_boxes = self.r_head_boxes(rel_idx).view(
            (nb_examples, batch_size, 2, self.embedding_dim))  # dim 2 distinguishes between upper and lower boundaries
        r_tail_boxes = self.r_tail_boxes(rel_idx).view((nb_examples, batch_size, 2, self.embedding_dim))
        head_bases = self.entity_bases(e_h_idx)
        head_bumps = self.entity_bumps(e_h_idx)
        tail_bases = self.entity_bases(e_t_idx)
        tail_bumps = self.entity_bumps(e_t_idx)
        return torch.stack((head_bases + tail_bumps, tail_bases + head_bumps), dim=2), \
               torch.stack((r_head_boxes, r_tail_boxes), dim=2)

    def forward_negatives(self, negatives):
        """
        @:return tuple (entities, relations, times) containing embeddings with
            entities.shape = (nb_negative_samples, batch_size, arity, embedding_dim)
            relations.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
            times.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
        """
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


class BoxTEmp(BaseBoxE):
    """
    BoxE model extended with boxes for timestamps.
    Can do interpolation completion on TKGs.
    """
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, weight_init='u', device='cpu', weight_init_args=(0, 1)):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, weight_init, device, weight_init_args)
        self.time_head_boxes = nn.Embedding(self.max_time,
                                            2 * embedding_dim)  # lower and upper boundaries, therefore 2*embedding_dim
        self.time_tail_boxes = nn.Embedding(self.max_time, 2 * embedding_dim)
        self.init_f(self.time_head_boxes.weight, *weight_init_args)
        self.init_f(self.time_tail_boxes.weight, *weight_init_args)

    def __call__(self, positives, negatives):
        return self.forward(positives, negatives)

    def params(self):
        return super().params() + [self.time_head_boxes.weight, self.time_tail_boxes.weight]

    def to(self, device):
        super().to(device)
        self.time_head_boxes = self.time_head_boxes.to(device)
        self.time_tail_boxes = self.time_tail_boxes.to(device)
        return self

    def state_dict(self):
        d = super().state_dict()
        d['time head box'] = self.time_head_boxes.state_dict()
        d['time tail box'] = self.time_tail_boxes.state_dict()
        return d

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.time_head_boxes.load_state_dict(state_dict['time head box'])
        self.time_tail_boxes.load_state_dict(state_dict['time tail box'])
        return self

    def compute_embeddings(self, tuples):
        entity_embs, relation_embs = super().compute_embeddings(tuples)
        nb_examples, _, batch_size = tuples.shape
        time_idx = tuples[:, 3]
        time_head_boxes = self.time_head_boxes(time_idx).view((nb_examples, batch_size, 2, self.embedding_dim))
        time_tail_boxes = self.time_tail_boxes(time_idx).view((nb_examples, batch_size, 2, self.embedding_dim))
        return entity_embs, relation_embs, torch.stack((time_head_boxes, time_tail_boxes), dim=2)

    def forward_negatives(self, negatives):
        """
        @:return tuple (entities, relations, times) containing embeddings with
            entities.shape = (nb_negative_samples, batch_size, arity, embedding_dim)
            relations.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
            times.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
        """
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


class BoxTEmpMLP(BaseBoxE):
    """
    Extension of the base BoxTEmp model, where time boxes are approximated by MLP.
    Enables extrapolation on TKGs.
    """
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, weight_init='u', nn_depth=3, nn_width=300, lookback=1, device='cpu', weight_init_args=(0, 1)):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, weight_init, device, weight_init_args)
        self.lookback = lookback
        self.nn_depth = nn_depth
        self.nn_width = nn_width
        self.initial_time_head_boxes = nn.Embedding(self.lookback, 2 * embedding_dim)  # lower and upper boundaries, therefore 2*embedding_dim
        self.initial_time_tail_boxes = nn.Embedding(self.lookback, 2 * embedding_dim)
        self.init_f(self.initial_time_head_boxes.weight, *weight_init_args)
        self.init_f(self.initial_time_tail_boxes.weight, *weight_init_args)
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
        return super().params() + [self.initial_time_head_boxes.weight, self.initial_time_tail_boxes.weight]\
               + list(self.time_transition.parameters())

    def to(self, device):
        super().to(device)
        self.time_transition = self.time_transition.to(device)
        self.initial_time_head_boxes = self.initial_time_head_boxes.to(device)
        self.initial_time_tail_boxes = self.initial_time_tail_boxes.to(device)
        return self

    def state_dict(self):
        d = super().state_dict()
        d['init time head'] = self.initial_time_head_boxes.state_dict()
        d['init time tail'] = self.initial_time_tail_boxes.state_dict()
        d['time transition'] = self.time_transition.state_dict()
        return d

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.initial_time_head_boxes.load_state_dict(state_dict['init time head'])
        self.initial_time_tail_boxes.load_state_dict(state_dict['init time tail'])
        self.time_transition.load_state_dict(state_dict['time transition'])
        return self

    def unroll_time(self):
        initial_times = torch.arange(0, self.lookback, device=self.device)
        init_head_boxes = self.initial_time_head_boxes(initial_times)
        init_tail_boxes = self.initial_time_tail_boxes(initial_times)
        current_state = torch.stack((init_head_boxes, init_tail_boxes), dim=1).flatten().to(self.device)
        time_head_boxes, time_tail_boxes = [], []
        for t in range(self.max_time):
            next_time = self.time_transition(current_state)
            time_head_boxes.append(next_time[:2*self.embedding_dim])
            time_tail_boxes.append(next_time[2*self.embedding_dim:])
            current_state = torch.cat((current_state[4*self.embedding_dim:], next_time))  # cut off upper/lower and head/
        return nn.Embedding.from_pretrained(torch.cat((init_head_boxes, torch.stack(time_head_boxes)))).to(self.device),\
               nn.Embedding.from_pretrained(torch.cat((init_tail_boxes, torch.stack(time_tail_boxes)))).to(self.device)

    def compute_embeddings(self, tuples):
        entity_embs, relation_embs = super().compute_embeddings(tuples)
        nb_examples, _, batch_size = tuples.shape
        time_idx = tuples[:, 3]
        all_time_head_boxes, all_time_tail_boxes = self.unroll_time()
        time_head_boxes = all_time_head_boxes(time_idx).view((nb_examples, batch_size, 2, self.embedding_dim))
        time_tail_boxes = all_time_tail_boxes(time_idx).view((nb_examples, batch_size, 2, self.embedding_dim))
        return entity_embs, relation_embs, torch.stack((time_head_boxes, time_tail_boxes), dim=2)

    def forward_negatives(self, negatives):
        """
        @:return tuple (entities, relations, times) containing embeddings with
            entities.shape = (nb_negative_samples, batch_size, arity, embedding_dim)
            relations.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
            times.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
        """
        return self.compute_embeddings(negatives)

    def forward_positives(self, positives):
        return self.compute_embeddings(positives)

    def forward(self, positives, negatives):
        """
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
        """
        positive_emb = self.forward_positives(positives)
        negative_emb = self.forward_negatives(negatives)
        return positive_emb, negative_emb


class BoxTEmpRelationMLP(BaseBoxE):
    """
    Extension of the base BoxE for TKGC, where relation boxes can move as function of time.
    Enables extrapolation on TKGs.
    """
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, weight_init='u', nn_depth=3, nn_width=300, lookback=1, device='cpu', weight_init_args=(0, 1)):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, weight_init, device, weight_init_args)
        self.lookback = lookback
        self.nn_depth = nn_depth
        self.nn_width = nn_width
        self.initial_r_head_boxes = torch.empty((len(relation_ids), self.lookback, 2 * embedding_dim), device=device)  # lower and upper boundaries, therefore 2*embedding_dim
        self.initial_r_tail_boxes = torch.empty((len(relation_ids), self.lookback, 2 * embedding_dim), device=device)
        self.r_head_boxes = self.initial_r_head_boxes
        self.r_tail_boxes = self.initial_r_tail_boxes
        self.init_f(self.initial_r_head_boxes, *weight_init_args)
        self.init_f(self.initial_r_tail_boxes, *weight_init_args)
        mlp_layers = [nn.Linear(4*self.embedding_dim*lookback, nn_width), nn.ReLU()]  # 4* because of lower/upper and head/tail
        for i in range(nn_depth):
            mlp_layers.append(nn.Linear(nn_width, nn_width))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(nn_width, 4*self.embedding_dim))
        time_transition = nn.Sequential(*mlp_layers)
        self.time_transition_list = nn.ModuleList()
        for r_id in relation_ids:
            self.time_transition_list.append(copy.deepcopy(time_transition))
        self.to(device)

    def __call__(self, positives, negatives):
        return self.forward(positives, negatives)

    def params(self):
        return [self.entity_bases.weight, self.entity_bumps.weight, self.initial_r_head_boxes, self.initial_r_tail_boxes] \
               + list(self.time_transition_list.parameters())

    def to(self, device):
        super().to(device)
        self.time_transition_list.to(device)
        self.initial_r_head_boxes.to(device)
        return self

    def state_dict(self):
        d = {'entity bases': self.entity_bases.state_dict(), 'entity bumps': self.entity_bumps.state_dict(),
             'time transitions': self.time_transition_list.state_dict(), 'init r head': self.initial_r_head_boxes,
             'init r tail': self.initial_r_tail_boxes}
        return d

    def load_state_dict(self, state_dict):
        self.entity_bases.load_state_dict(state_dict['entity bases'])
        self.entity_bumps.load_state_dict(state_dict['entity bumps'])
        self.time_transition_list.load_state_dict(state_dict['time transitions'])
        self.initial_r_head_boxes = state_dict['init r head']
        self.initial_r_tail_boxes = state_dict['init r tail']
        return self

    def unroll_time(self):
        all_head_boxes = []
        all_tail_boxes = []
        for r_id in self.relation_ids:
            i_r = self.get_r_idx_by_id(r_id)
            init_h, init_t = self.initial_r_head_boxes[i_r, :, :], self.initial_r_tail_boxes[i_r, :, :]  # shape (lookback, 2*emb_dim)
            current_state = torch.stack((init_h, init_t), dim=1).flatten().to(self.device)
            time_head_boxes, time_tail_boxes = [], []
            for t in range(self.max_time):
                next_time = self.time_transition_list[i_r](current_state)
                time_head_boxes.append(next_time[:2 * self.embedding_dim])
                time_tail_boxes.append(next_time[2 * self.embedding_dim:])
                current_state = torch.cat(
                    (current_state[4 * self.embedding_dim:], next_time))  # cut off upper/lower and head/
            all_head_boxes.append(torch.cat((init_h, torch.stack(time_head_boxes))))
            all_tail_boxes.append(torch.cat((init_h, torch.stack(time_tail_boxes))))
        return torch.stack(all_head_boxes), torch.stack(all_tail_boxes)

    def compute_embeddings(self, tuples):
        nb_examples, _, batch_size = tuples.shape
        e_h_idx = self.get_e_idx_by_id(tuples[:, 0]).to(self.device)
        rel_idx = self.get_r_idx_by_id(tuples[:, 1]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[:, 2]).to(self.device)
        time_idx = tuples[:, 3]
        all_r_head_boxes, all_r_tail_boxes = self.unroll_time()  # shape (relation, timestamp, 2*embedding_dim)

        r_head_boxes = all_r_head_boxes[rel_idx, time_idx, :].view((nb_examples, batch_size, 2, self.embedding_dim))
        r_tail_boxes = all_r_tail_boxes[rel_idx, time_idx, :].view((nb_examples, batch_size, 2, self.embedding_dim))
        head_bases = self.entity_bases(e_h_idx)
        head_bumps = self.entity_bumps(e_h_idx)
        tail_bases = self.entity_bases(e_t_idx)
        tail_bumps = self.entity_bumps(e_t_idx)
        entity_embs, relation_embs = torch.stack((head_bases + tail_bumps, tail_bases + head_bumps), dim=2), torch.stack((r_head_boxes, r_tail_boxes), dim=2)
        return entity_embs, relation_embs, torch.zeros_like(relation_embs)  # return dummy for time boxes

    def forward_negatives(self, negatives):
        """
        @:return tuple (entities, relations, times) containing embeddings with
            entities.shape = (nb_negative_samples, batch_size, arity, embedding_dim)
            relations.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
            times.shape = (nb_negative_samples, batch_size, arity, 2, embedding_dim)
        """
        return self.compute_embeddings(negatives)

    def forward_positives(self, positives):
        return self.compute_embeddings(positives)

    def forward(self, positives, negatives):
        """
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
        """
        positive_emb = self.forward_positives(positives)
        negative_emb = self.forward_negatives(negatives)
        return positive_emb, negative_emb