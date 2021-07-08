import torch
import torch.nn as nn
import copy


class BaseBoxE(nn.Module):

    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, device='cpu',
                 weight_init_args=(0, 1), norm_embeddings=False):
        super(BaseBoxE, self).__init__()
        self.init_f = torch.nn.init.uniform_
        if norm_embeddings:
            self.embedding_norm_fn = nn.Tanh()
        else:
            self.embedding_norm_fn = nn.Identity()
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

        self.r_head_base_points = nn.Embedding(self.nb_relations, embedding_dim)
        self.r_head_widths = nn.Embedding(self.nb_relations, embedding_dim)
        self.r_head_size_scales = nn.Embedding(self.nb_relations, 1)
        self.r_tail_base_points = nn.Embedding(self.nb_relations, embedding_dim)
        self.r_tail_widths = nn.Embedding(self.nb_relations, embedding_dim)
        self.r_tail_size_scales = nn.Embedding(self.nb_relations, 1)

        self.entity_bases = nn.Embedding(self.nb_entities, embedding_dim)
        self.entity_bumps = nn.Embedding(self.nb_entities, embedding_dim)

        self.init_f(self.r_head_base_points.weight, *weight_init_args)
        self.init_f(self.r_head_size_scales.weight, -1, 1)
        self.init_f(self.r_head_widths.weight, *weight_init_args)
        self.init_f(self.r_tail_base_points.weight, *weight_init_args)
        self.init_f(self.r_tail_size_scales.weight, -1, 1)
        self.init_f(self.r_tail_widths.weight, *weight_init_args)
        self.init_f(self.entity_bases.weight, *weight_init_args)
        self.init_f(self.entity_bumps.weight, *weight_init_args)

    def shape_norm(self, t, dim):
        # taken from original BoxE implementation (https://github.com/ralphabb/BoxE)
        step1_tensor = torch.abs(t)
        step2_tensor = step1_tensor + (10 ** -8)
        log_norm_tensor = torch.log(step2_tensor)
        step3_tensor = torch.mean(log_norm_tensor, dim=dim, keepdim=True)

        norm_volume = torch.exp(step3_tensor)
        return t / norm_volume

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def get_r_idx_by_id(self, r_ids):
        """@:param r_names tensor of realtion ids"""
        return r_ids - self.relation_id_offset

    def get_e_idx_by_id(self, e_ids):
        return e_ids

    def compute_relation_embeddings(self, tuples):
        nb_examples, _, batch_size = tuples.shape
        rel_idx = self.get_r_idx_by_id(tuples[:, 1]).to(self.device)
        # get relevant embeddings
        r_head_bases = self.r_head_base_points(rel_idx)
        r_tail_bases = self.r_tail_base_points(rel_idx)

        r_head_widths = self.shape_norm(self.r_head_widths(rel_idx), dim=2)  # normalize relative widths
        r_tail_widths = self.shape_norm(self.r_tail_widths(rel_idx), dim=2)

        r_head_scales = nn.functional.elu(self.r_head_size_scales(rel_idx)) + 1  # ensure scales > 0
        r_tail_scales = nn.functional.elu(self.r_tail_size_scales(rel_idx)) + 1
        # compute scaled widths
        head_deltas = torch.multiply(r_head_widths, r_head_scales)
        tail_deltas = torch.multiply(r_tail_widths, r_tail_scales)
        # compute corners from base and width
        head_corner_1 = r_head_bases + head_deltas
        head_corner_2 = r_head_bases - head_deltas
        tail_corner_1 = r_tail_bases + tail_deltas
        tail_corner_2 = r_tail_bases - tail_deltas
        # determine upper and lower corners
        head_upper = torch.maximum(head_corner_1, head_corner_2)
        head_lower = torch.minimum(head_corner_1, head_corner_2)
        tail_upper = torch.maximum(tail_corner_1, tail_corner_2)
        tail_lower = torch.minimum(tail_corner_1, tail_corner_2)
        # assemble boxes
        r_head_boxes = torch.stack((head_upper, head_lower), dim=2)
        r_tail_boxes = torch.stack((tail_upper, tail_lower), dim=2)
        return self.embedding_norm_fn(torch.stack((r_head_boxes, r_tail_boxes), dim=2))

    def compute_entity_embeddings(self, tuples):
        e_h_idx = self.get_e_idx_by_id(tuples[:, 0]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[:, 2]).to(self.device)
        head_bases = self.entity_bases(e_h_idx)
        head_bumps = self.entity_bumps(e_h_idx)
        tail_bases = self.entity_bases(e_t_idx)
        tail_bumps = self.entity_bumps(e_t_idx)
        return self.embedding_norm_fn(torch.stack((head_bases + tail_bumps, tail_bases + head_bumps), dim=2))

    def compute_embeddings(self, tuples):

        # get relevant entity bases and bumps

        return self.compute_entity_embeddings(tuples), self.compute_relation_embeddings(tuples)

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


class NaiveBaseBoxE():
    """
        BoxE model extended with boxes for timestamps.
        Can do interpolation completion on TKGs.
        """

    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, weight_init='u', device='cpu',
                 weight_init_args=(0, 1), norm_embeddings=False):
        if weight_init == 'u':
            self.init_f = torch.nn.init.uniform_
        elif weight_init == 'n':
            self.init_f = torch.nn.init.normal_
        elif weight_init == 'default':
            self.init_f = lambda x, *args: x  # don't change initialization
        else:
            raise ValueError(
                "Invalid value for argument 'weight_init'. Use 'u' for uniform or 'n' for normal weight initialization.")
        if norm_embeddings:
            self.embedding_norm_fn = nn.Tanh()
        else:
            self.embedding_norm_fn = nn.Identity()
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
        return self.embedding_norm_fn(torch.stack((head_bases + tail_bumps, tail_bases + head_bumps), dim=2)), \
               self.embedding_norm_fn(torch.stack((r_head_boxes, r_tail_boxes), dim=2))

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


class StaticBoxE(BaseBoxE):
    """
    BoxE model extended with boxes for timestamps.
    Can do interpolation completion on TKGs.
    """
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, device='cpu', weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings)

    def compute_embeddings(self, tuples):
        entity_embs, relation_embs = super().compute_embeddings(tuples)
        return entity_embs, relation_embs, None


class TempBoxE_S(BaseBoxE):
    """
    BoxE model extended with boxes for timestamps.
    Can do interpolation completion on TKGs.
    """
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, device='cpu', weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings)
        self.time_head_base_points = nn.Embedding(self.max_time, embedding_dim)
        self.time_head_widths = nn.Embedding(self.max_time, embedding_dim)
        self.time_head_size_scales = nn.Embedding(self.max_time, 1)
        self.time_tail_base_points = nn.Embedding(self.max_time, embedding_dim)
        self.time_tail_widths = nn.Embedding(self.max_time, embedding_dim)
        self.time_tail_size_scales = nn.Embedding(self.max_time, 1)
        self.init_f(self.time_head_base_points.weight, *weight_init_args)
        self.init_f(self.time_head_widths.weight, *weight_init_args)
        self.init_f(self.time_head_size_scales.weight, -1, 1)
        self.init_f(self.time_tail_base_points.weight, *weight_init_args)
        self.init_f(self.time_tail_widths.weight, *weight_init_args)
        self.init_f(self.time_tail_size_scales.weight, -1, 1)

    def compute_embeddings(self, tuples):
        entity_embs, relation_embs = super().compute_embeddings(tuples)
        nb_examples, _, batch_size = tuples.shape
        time_idx = tuples[:, 3]
        time_head_bases = self.time_head_base_points(time_idx)
        time_tail_bases = self.time_tail_base_points(time_idx)

        time_head_widths = self.shape_norm(self.time_head_widths(time_idx), dim=2)  # normalize relative widths
        time_tail_widths = self.shape_norm(self.time_tail_widths(time_idx), dim=2)

        time_head_scales = nn.functional.elu(self.time_head_size_scales(time_idx)) + 1  # ensure scales > 0
        time_tail_scales = nn.functional.elu(self.time_tail_size_scales(time_idx)) + 1
        # compute scaled widths
        head_deltas = torch.multiply(time_head_widths, time_head_scales)
        tail_deltas = torch.multiply(time_tail_widths, time_tail_scales)
        # compute corners from base and width
        head_corner_1 = time_head_bases + head_deltas
        head_corner_2 = time_head_bases - head_deltas
        tail_corner_1 = time_tail_bases + tail_deltas
        tail_corner_2 = time_tail_bases - tail_deltas
        # determine upper and lower corners
        head_upper = torch.maximum(head_corner_1, head_corner_2)
        head_lower = torch.minimum(head_corner_1, head_corner_2)
        tail_upper = torch.maximum(tail_corner_1, tail_corner_2)
        tail_lower = torch.minimum(tail_corner_1, tail_corner_2)
        # assemble boxes
        time_head_boxes = torch.stack((head_upper, head_lower), dim=2)
        time_tail_boxes = torch.stack((tail_upper, tail_lower), dim=2)
        return entity_embs, relation_embs, self.embedding_norm_fn(torch.stack((time_head_boxes, time_tail_boxes), dim=2))


class TempBoxE_R(BaseBoxE):
    """
    BoxE model extended with boxes for timestamps.
    Can do interpolation completion on TKGs.
    """
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, device='cpu', weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings)
        self.r_head_base_points = nn.Parameter(torch.empty((self.max_time, self.nb_relations, self.embedding_dim)))
        self.r_head_widths = nn.Parameter(torch.empty((self.max_time, self.nb_relations, self.embedding_dim)))
        self.r_head_size_scales = nn.Parameter(torch.empty((self.max_time, self.nb_relations, 1)))
        self.r_tail_base_points = nn.Parameter(torch.empty((self.max_time, self.nb_relations, self.embedding_dim)))
        self.r_tail_widths = nn.Parameter(torch.empty((self.max_time, self.nb_relations, self.embedding_dim)))
        self.r_tail_size_scales = nn.Parameter(torch.empty((self.max_time, self.nb_relations, 1)))
        self.init_f(self.r_head_base_points, *weight_init_args)
        self.init_f(self.r_head_widths, *weight_init_args)
        self.init_f(self.r_head_size_scales, -1, 1)
        self.init_f(self.r_tail_base_points, *weight_init_args)
        self.init_f(self.r_tail_widths, *weight_init_args)
        self.init_f(self.r_tail_size_scales, -1, 1)

    def compute_embeddings(self, tuples):
        nb_examples, _, batch_size = tuples.shape
        e_h_idx = self.get_e_idx_by_id(tuples[:, 0]).to(self.device)
        rel_idx = self.get_r_idx_by_id(tuples[:, 1]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[:, 2]).to(self.device)
        time_idx = tuples[:, 3]
        r_head_bases = self.r_head_base_points[time_idx, rel_idx, :]  # shape (nb_examples, batch_size, embedding_dim)
        r_tail_bases = self.r_tail_base_points[time_idx, rel_idx, :]

        r_head_widths = self.shape_norm(self.r_head_widths[time_idx, rel_idx, :], dim=2)  # normalize relative widths
        r_tail_widths = self.shape_norm(self.r_tail_widths[time_idx, rel_idx, :], dim=2)

        r_head_scales = nn.functional.elu(self.r_head_size_scales[time_idx, rel_idx, :]) + 1  # ensure scales > 0
        r_tail_scales = nn.functional.elu(self.r_tail_size_scales[time_idx, rel_idx, :]) + 1
        # compute scaled widths
        head_deltas = torch.multiply(r_head_widths, r_head_scales)
        tail_deltas = torch.multiply(r_tail_widths, r_tail_scales)
        # compute corners from base and width
        head_corner_1 = r_head_bases + head_deltas
        head_corner_2 = r_head_bases - head_deltas
        tail_corner_1 = r_tail_bases + tail_deltas
        tail_corner_2 = r_tail_bases - tail_deltas
        # determine upper and lower corners
        head_upper = torch.maximum(head_corner_1, head_corner_2)  # shape (nb_examples, batch_size, embedding_dim)
        head_lower = torch.minimum(head_corner_1, head_corner_2)
        tail_upper = torch.maximum(tail_corner_1, tail_corner_2)
        tail_lower = torch.minimum(tail_corner_1, tail_corner_2)
        # assemble boxes
        r_head_boxes = torch.stack((head_upper, head_lower), dim=2)  # shape (nb_examples,
        r_tail_boxes = torch.stack((tail_upper, tail_lower), dim=2)
        # entity embeddings
        head_bases = self.entity_bases(e_h_idx)
        head_bumps = self.entity_bumps(e_h_idx)
        tail_bases = self.entity_bases(e_t_idx)
        tail_bumps = self.entity_bumps(e_t_idx)
        # stack everything
        entity_embs, relation_embs = torch.stack((head_bases + tail_bumps, tail_bases + head_bumps), dim=2), torch.stack((r_head_boxes, r_tail_boxes), dim=2)
        return self.embedding_norm_fn(entity_embs), self.embedding_norm_fn(relation_embs), None  # return no time boxes


class TempBoxE_SMLP(BaseBoxE):
    """
    Extension of the base BoxTEmp model, where time boxes are approximated by MLP.
    Enables extrapolation on TKGs.
    """
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, nn_depth=3, nn_width=300,
                 lookback=1, device='cpu', weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings)
        self.lookback = lookback
        self.nn_depth = nn_depth
        self.nn_width = nn_width
        self.init_time_head_base_points = nn.Embedding(self.lookback, embedding_dim)
        self.init_time_head_widths = nn.Embedding(self.lookback, embedding_dim)
        self.init_time_head_size_scales = nn.Embedding(self.lookback, 1)
        self.init_time_tail_base_points = nn.Embedding(self.lookback, embedding_dim)
        self.init_time_tail_widths = nn.Embedding(self.lookback, embedding_dim)
        self.init_time_tail_size_scales = nn.Embedding(self.lookback, 1)
        self.init_f(self.init_time_head_base_points.weight, *weight_init_args)
        self.init_f(self.init_time_head_widths.weight, *weight_init_args)
        self.init_f(self.init_time_head_size_scales.weight, -1, 1)
        self.init_f(self.init_time_tail_base_points.weight, *weight_init_args)
        self.init_f(self.init_time_tail_widths.weight, *weight_init_args)
        self.init_f(self.init_time_tail_size_scales.weight, -1, 1)
        mlp_layers = [nn.Linear(4*self.embedding_dim*lookback, nn_width), nn.ReLU()]  # 4* because of lower/upper and head/tail
        for i in range(nn_depth):
            mlp_layers.append(nn.Linear(nn_width, nn_width))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(nn_width, 4*self.embedding_dim))
        self.time_transition = nn.Sequential(*mlp_layers)
        self.to(device)

    def unroll_time(self, init_head_boxes, init_tail_boxes):
        current_state = torch.stack((init_head_boxes, init_tail_boxes), dim=1).flatten().to(self.device)
        time_head_boxes, time_tail_boxes = [], []
        for t in range(self.max_time):
            next_time = self.embedding_norm_fn(self.time_transition(current_state))
            time_head_boxes.append(next_time[:2*self.embedding_dim])
            time_tail_boxes.append(next_time[2*self.embedding_dim:])
            current_state = torch.cat((current_state[4*self.embedding_dim:], next_time))  # cut off upper/lower and head/
        return nn.Embedding.from_pretrained(torch.cat((init_head_boxes, torch.stack(time_head_boxes)))).to(self.device),\
               nn.Embedding.from_pretrained(torch.cat((init_tail_boxes, torch.stack(time_tail_boxes)))).to(self.device)

    def compute_embeddings(self, tuples):
        entity_embs, relation_embs = super().compute_embeddings(tuples)
        nb_examples, _, batch_size = tuples.shape
        time_idx = tuples[:, 3]
        initial_times = torch.arange(0, self.lookback, device=self.device)
        init_time_head_bases = self.init_time_head_base_points(initial_times)
        init_time_tail_bases = self.init_time_tail_base_points(initial_times)

        init_time_head_widths = self.shape_norm(self.init_time_head_widths(initial_times), dim=1)  # normalize relative widths
        init_time_tail_widths = self.shape_norm(self.init_time_tail_widths(initial_times), dim=1)

        init_time_head_scales = nn.functional.elu(self.init_time_head_size_scales(initial_times)) + 1  # ensure scales > 0
        init_time_tail_scales = nn.functional.elu(self.init_time_tail_size_scales(initial_times)) + 1
        # compute scaled widths
        head_deltas = torch.multiply(init_time_head_widths, init_time_head_scales)
        tail_deltas = torch.multiply(init_time_tail_widths, init_time_tail_scales)
        # compute corners from base and width
        head_corner_1 = init_time_head_bases + head_deltas
        head_corner_2 = init_time_head_bases - head_deltas
        tail_corner_1 = init_time_tail_bases + tail_deltas
        tail_corner_2 = init_time_tail_bases - tail_deltas
        # determine upper and lower corners
        head_upper = torch.maximum(head_corner_1, head_corner_2)
        head_lower = torch.minimum(head_corner_1, head_corner_2)
        tail_upper = torch.maximum(tail_corner_1, tail_corner_2)
        tail_lower = torch.minimum(tail_corner_1, tail_corner_2)
        # assemble boxes
        init_time_head_boxes = torch.stack((head_upper, head_lower), dim=2).flatten(1,2)
        init_time_tail_boxes = torch.stack((tail_upper, tail_lower), dim=2).flatten(1,2)
        all_time_head_boxes, all_time_tail_boxes = self.unroll_time(init_time_head_boxes, init_time_tail_boxes)
        time_head_boxes = all_time_head_boxes(time_idx).view((nb_examples, batch_size, 2, self.embedding_dim))
        time_tail_boxes = all_time_tail_boxes(time_idx).view((nb_examples, batch_size, 2, self.embedding_dim))
        return entity_embs, relation_embs, self.embedding_norm_fn(torch.stack((time_head_boxes, time_tail_boxes), dim=2))


class TempBoxE_RMLP_multi(BaseBoxE):
    """
    Extension of the base BoxE for TKGC, where relation boxes can move as function of time.
    Enables extrapolation on TKGs.
    """
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, nn_depth=3, nn_width=300,
                 lookback=1, device='cpu', weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings)
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
        self.time_transition = nn.ModuleList()
        for r_id in relation_ids:
            self.time_transition.append(copy.deepcopy(time_transition))
        self.to(device)

    def unroll_time(self):
        all_head_boxes = []
        all_tail_boxes = []
        for r_id in self.relation_ids:
            i_r = self.get_r_idx_by_id(r_id)
            init_h, init_t = self.initial_r_head_boxes[i_r, :, :].to(self.device), self.initial_r_tail_boxes[i_r, :, :].to(self.device)  # shape (lookback, 2*emb_dim)
            current_state = torch.stack((init_h, init_t), dim=1).flatten().to(self.device)
            time_head_boxes, time_tail_boxes = [], []
            for t in range(self.max_time):
                next_time = self.embedding_norm_fn(self.time_transition[i_r](current_state))
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
        return self.embedding_norm_fn(entity_embs), self.embedding_norm_fn(relation_embs), None  # return no time boxes


class TempBoxE_RMLP(TempBoxE_RMLP_multi):
    """
    Extension of the base BoxE for TKGC, where relation boxes can move as function of time.
    Enables extrapolation on TKGs.
    """
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, nn_depth=3, nn_width=300,
                 lookback=1, device='cpu', weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, nn_depth, nn_width, lookback,
                         device, weight_init_args, norm_embeddings)
        self.nb_relations = len(self.relation_ids)
        self.init_r_head_base_points = nn.Parameter(torch.empty((self.lookback, self.nb_relations, embedding_dim)))
        self.init_r_head_widths = nn.Parameter(torch.empty((self.lookback, self.nb_relations, embedding_dim)))
        self.init_r_head_size_scales = nn.Parameter(torch.empty((self.lookback, self.nb_relations, 1)))
        self.init_r_tail_base_points = nn.Parameter(torch.empty((self.lookback, self.nb_relations, embedding_dim)))
        self.init_r_tail_widths = nn.Parameter(torch.empty((self.lookback, self.nb_relations, embedding_dim)))
        self.init_r_tail_size_scales = nn.Parameter(torch.empty((self.lookback, self.nb_relations, 1)))
        self.init_f(self.init_r_head_base_points, *weight_init_args)
        self.init_f(self.init_r_head_widths, *weight_init_args)
        self.init_f(self.init_r_head_size_scales, -1, 1)
        self.init_f(self.init_r_tail_base_points, *weight_init_args)
        self.init_f(self.init_r_tail_widths, *weight_init_args)
        self.init_f(self.init_r_tail_size_scales, -1, 1)
        mlp_layers = [nn.Linear(4*self.embedding_dim*lookback*len(self.relation_ids), nn_width), nn.ReLU()]  # 4* because of lower/upper and head/tail
        for i in range(nn_depth):
            mlp_layers.append(nn.Linear(nn_width, nn_width))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(nn_width, 4*self.embedding_dim*len(self.relation_ids)))
        self.time_transition = nn.Sequential(*mlp_layers)
        self.to(device)

    def unroll_time(self, init_head_boxes, init_tail_boxes):
        init_state = torch.stack((init_head_boxes, init_tail_boxes), dim=1).flatten().to(self.device)
        current_state = init_state
        time_head_boxes, time_tail_boxes = [], []
        for t in range(self.max_time):
            next_time = self.embedding_norm_fn(self.time_transition(current_state))
            time_head_boxes.append(next_time[:self.nb_relations * 2 * self.embedding_dim].view((self.nb_relations, 2*self.embedding_dim)))
            time_tail_boxes.append(next_time[self.nb_relations * 2 * self.embedding_dim:].view((self.nb_relations, 2*self.embedding_dim)))
            current_state = torch.cat(
                (current_state[self.nb_relations * 4 * self.embedding_dim:], next_time))  # cut off upper/lower and head/
        return torch.cat((init_head_boxes, torch.stack(time_head_boxes))).to(self.device), \
               torch.cat((init_tail_boxes, torch.stack(time_tail_boxes))).to(self.device)

    def compute_embeddings(self, tuples):
        nb_examples, _, batch_size = tuples.shape
        e_h_idx = self.get_e_idx_by_id(tuples[:, 0]).to(self.device)
        rel_idx = self.get_r_idx_by_id(tuples[:, 1]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[:, 2]).to(self.device)
        time_idx = tuples[:, 3]
        initial_times = torch.arange(0, self.lookback, device=self.device)
        init_r_head_bases = self.init_r_head_base_points[initial_times,:,:]
        init_r_tail_bases = self.init_r_tail_base_points[initial_times,:,:]

        init_r_head_widths = self.shape_norm(self.init_r_head_widths[initial_times], dim=2)  # normalize relative widths
        init_r_tail_widths = self.shape_norm(self.init_r_tail_widths[initial_times], dim=2)

        init_r_head_scales = nn.functional.elu(
            self.init_r_head_size_scales[initial_times]) + 1  # ensure scales > 0
        init_r_tail_scales = nn.functional.elu(self.init_r_tail_size_scales[initial_times]) + 1
        # compute scaled widths
        head_deltas = torch.multiply(init_r_head_widths, init_r_head_scales)
        tail_deltas = torch.multiply(init_r_tail_widths, init_r_tail_scales)
        # compute corners from base and width
        head_corner_1 = init_r_head_bases + head_deltas
        head_corner_2 = init_r_head_bases - head_deltas
        tail_corner_1 = init_r_tail_bases + tail_deltas
        tail_corner_2 = init_r_tail_bases - tail_deltas
        # determine upper and lower corners
        head_upper = torch.maximum(head_corner_1, head_corner_2)
        head_lower = torch.minimum(head_corner_1, head_corner_2)
        tail_upper = torch.maximum(tail_corner_1, tail_corner_2)
        tail_lower = torch.minimum(tail_corner_1, tail_corner_2)
        # assemble boxes
        init_r_head_boxes = torch.stack((head_upper, head_lower), dim=2).flatten(2, 3)
        init_r_tail_boxes = torch.stack((tail_upper, tail_lower), dim=2).flatten(2, 3)
        all_r_head_boxes, all_r_tail_boxes = self.unroll_time(init_r_head_boxes, init_r_tail_boxes)  # shape (timestamp, relation, 2*embedding_dim)

        r_head_boxes = all_r_head_boxes[time_idx, rel_idx, :].view((nb_examples, batch_size, 2, self.embedding_dim))
        r_tail_boxes = all_r_tail_boxes[time_idx, rel_idx, :].view((nb_examples, batch_size, 2, self.embedding_dim))
        head_bases = self.entity_bases(e_h_idx)
        head_bumps = self.entity_bumps(e_h_idx)
        tail_bases = self.entity_bases(e_t_idx)
        tail_bumps = self.entity_bumps(e_t_idx)
        entity_embs, relation_embs = torch.stack((head_bases + tail_bumps, tail_bases + head_bumps), dim=2), torch.stack((r_head_boxes, r_tail_boxes), dim=2)
        return self.embedding_norm_fn(entity_embs), self.embedding_norm_fn(relation_embs), None  # return no time boxes


class TempBoxE_SMLP_Plus(TempBoxE_SMLP):
    """
    Extension of the base BoxTEmp model, where time boxes are approximated by MLP.
    Enables extrapolation on TKGs.
    """
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, nn_depth=3, nn_width=300,
                 lookback=1, device='cpu', weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, nn_depth, nn_width,
                 lookback, device, weight_init_args, norm_embeddings)
        self.time_embeddings = nn.Embedding(self.max_time, embedding_dim)
        self.init_f(self.time_embeddings.weight, *weight_init_args)
        mlp_layers = [nn.Linear(4*self.embedding_dim*lookback + self.embedding_dim, nn_width), nn.ReLU()]  # 4* because of lower/upper, head/tail
        for i in range(nn_depth):
            mlp_layers.append(nn.Linear(nn_width, nn_width))
            mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(nn_width, 4*self.embedding_dim))
        self.time_transition = nn.Sequential(*mlp_layers)
        self.to(device)

    def unroll_time(self, init_head_boxes, init_tail_boxes):
        current_state = torch.stack((init_head_boxes, init_tail_boxes), dim=1).flatten().to(self.device)
        time_head_boxes, time_tail_boxes = [], []
        for t in range(self.max_time):
            time_emb = self.time_embeddings(torch.tensor(t, device=self.device)).squeeze()
            current_state = torch.cat((time_emb, current_state))
            next_time = self.embedding_norm_fn(self.time_transition(current_state))
            time_head_boxes.append(next_time[:2*self.embedding_dim])
            time_tail_boxes.append(next_time[2*self.embedding_dim:])
            current_state = torch.cat((current_state[5*self.embedding_dim:], next_time))  # cut off upper/lower, head/tail and time_emb
        return nn.Embedding.from_pretrained(torch.cat((init_head_boxes, torch.stack(time_head_boxes)))).to(self.device),\
               nn.Embedding.from_pretrained(torch.cat((init_tail_boxes, torch.stack(time_tail_boxes)))).to(self.device)


class TempBoxE_RMLP_Plus(TempBoxE_RMLP):
    """
    Extension of the base BoxE for TKGC, where relation boxes can move as function of time.
    Enables extrapolation on TKGs.
    """
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, nn_depth=3, nn_width=300,
                 lookback=1, device='cpu', weight_init_args=(0, 1), norm_embeddings=False, layer_norm=True, layer_affine=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, nn_depth, nn_width,
                 lookback, device, weight_init_args, norm_embeddings)
        self.time_embeddings = nn.Embedding(self.max_time, embedding_dim)
        self.init_f(self.time_embeddings.weight, *weight_init_args)
        mlp_layers = [nn.Linear(4*self.embedding_dim*lookback*len(self.relation_ids)+self.embedding_dim, nn_width), nn.Tanh()]   # 4* because of lower/upper, head/tail
        if layer_norm:
            mlp_layers.append(nn.LayerNorm(self.nn_width, elementwise_affine=layer_affine))
        for i in range(nn_depth):
            mlp_layers.append(nn.Linear(nn_width, nn_width))
            mlp_layers.append(nn.Tanh())
            if layer_norm:
                mlp_layers.append(nn.LayerNorm(self.nn_width, elementwise_affine=layer_affine))
        mlp_layers.append(nn.Linear(nn_width, 4*self.embedding_dim*len(self.relation_ids)))
        mlp_layers.append(nn.Tanh())
        self.time_transition = nn.Sequential(*mlp_layers)
        self.to(device)

    def unroll_time(self, init_head_boxes, init_tail_boxes):
        init_state = torch.stack((init_head_boxes, init_tail_boxes), dim=1).flatten().to(self.device)
        current_state = init_state
        time_head_boxes, time_tail_boxes = [], []
        for t in range(self.max_time):
            time_emb = self.time_embeddings(torch.tensor(t, device=self.device)).squeeze()
            current_state = torch.cat((time_emb, current_state))
            next_time = self.embedding_norm_fn(self.time_transition(current_state))
            time_head_boxes.append(next_time[:self.nb_relations * 2 * self.embedding_dim].view((self.nb_relations, 2*self.embedding_dim)))
            time_tail_boxes.append(next_time[self.nb_relations * 2 * self.embedding_dim:].view((self.nb_relations, 2*self.embedding_dim)))
            current_state = torch.cat(
                (current_state[(self.nb_relations * 4 * self.embedding_dim + self.embedding_dim):], next_time))  # cut off upper/lower and head/
        return torch.cat((init_head_boxes, torch.stack(time_head_boxes))).to(self.device), \
               torch.cat((init_tail_boxes, torch.stack(time_tail_boxes))).to(self.device)


class TempBoxE_M(BaseBoxE):
    """
    Generalization of TempBoxE^S that employes multiples time boxes, where #timeboxes == #relations
    Can do interpolation completion on TKGs.
    """
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, device='cpu', weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings)
        self.time_head_base_points = nn.Parameter(torch.empty((self.max_time, self.nb_relations, self.embedding_dim)))
        self.time_head_widths = nn.Parameter(torch.empty((self.max_time, self.nb_relations, self.embedding_dim)))
        self.time_head_size_scales = nn.Parameter(torch.empty((self.max_time, self.nb_relations, 1)))
        self.time_tail_base_points = nn.Parameter(torch.empty((self.max_time, self.nb_relations, self.embedding_dim)))
        self.time_tail_widths = nn.Parameter(torch.empty((self.max_time, self.nb_relations, self.embedding_dim)))
        self.time_tail_size_scales = nn.Parameter(torch.empty((self.max_time, self.nb_relations, 1)))
        self.init_f(self.time_head_base_points, *weight_init_args)
        self.init_f(self.time_head_widths, *weight_init_args)
        self.init_f(self.time_head_size_scales, -1, 1)
        self.init_f(self.time_tail_base_points, *weight_init_args)
        self.init_f(self.time_tail_widths, *weight_init_args)
        self.init_f(self.time_tail_size_scales, -1, 1)

    def compute_time_embeddings(self, tuples):
        nb_examples, _, batch_size = tuples.shape
        rel_idx = self.get_r_idx_by_id(tuples[:, 1]).to(self.device)
        time_idx = tuples[:, 3]
        time_head_bases = self.time_head_base_points[time_idx, rel_idx, :]
        time_tail_bases = self.time_tail_base_points[time_idx, rel_idx, :]

        time_head_widths = self.shape_norm(self.time_head_widths[time_idx, rel_idx, :],
                                           dim=2)  # normalize relative widths
        time_tail_widths = self.shape_norm(self.time_tail_widths[time_idx, rel_idx, :], dim=2)

        time_head_scales = nn.functional.elu(self.time_head_size_scales[time_idx, rel_idx, :]) + 1  # ensure scales > 0
        time_tail_scales = nn.functional.elu(self.time_tail_size_scales[time_idx, rel_idx, :]) + 1
        # compute scaled widths
        head_deltas = torch.multiply(time_head_widths, time_head_scales)
        tail_deltas = torch.multiply(time_tail_widths, time_tail_scales)
        # compute corners from base and width
        head_corner_1 = time_head_bases + head_deltas
        head_corner_2 = time_head_bases - head_deltas
        tail_corner_1 = time_tail_bases + tail_deltas
        tail_corner_2 = time_tail_bases - tail_deltas
        # determine upper and lower corners
        head_upper = torch.maximum(head_corner_1, head_corner_2)
        head_lower = torch.minimum(head_corner_1, head_corner_2)
        tail_upper = torch.maximum(tail_corner_1, tail_corner_2)
        tail_lower = torch.minimum(tail_corner_1, tail_corner_2)
        # assemble boxes
        time_head_boxes = torch.stack((head_upper, head_lower), dim=2)
        time_tail_boxes = torch.stack((tail_upper, tail_lower), dim=2)
        return self.embedding_norm_fn(torch.stack((time_head_boxes, time_tail_boxes), dim=2))

    def compute_embeddings(self, tuples):
        entity_embs, relation_embs = super().compute_embeddings(tuples)
        time_embs = self.compute_time_embeddings(tuples)
        return entity_embs, relation_embs, time_embs


class DEBoxE_TimeEntEmb(BaseBoxE):
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, device='cpu',
                 weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings=False)
        if norm_embeddings:
            self.embedding_norm_fn_ = nn.Tanh()
        else:
            self.embedding_norm_fn_ = nn.Identity()
        self.time_entity_vectors = nn.Parameter(torch.empty(self.max_time, self.nb_entities, self.embedding_dim))
        self.init_f(self.time_entity_vectors, *weight_init_args)

    def compute_embeddings(self, tuples):
        entity_embs, relation_embs = super().compute_embeddings(tuples)
        e_h_idx = self.get_e_idx_by_id(tuples[:, 0]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[:, 2]).to(self.device)
        time_idx = tuples[:, 3]
        head_time_vecs = self.time_entity_vectors[time_idx, e_h_idx, :]
        tail_time_vecs = self.time_entity_vectors[time_idx, e_t_idx, :]
        time_vecs = torch.stack((head_time_vecs, tail_time_vecs), dim=2)
        entity_embs = entity_embs + time_vecs
        return self.embedding_norm_fn_(entity_embs), self.embedding_norm_fn_(relation_embs), None


class DEBoxE_OneBumpPerTime(BaseBoxE):
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, device='cpu',
                 weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings=False)
        if norm_embeddings:
            self.embedding_norm_fn_ = nn.Tanh()
        else:
            self.embedding_norm_fn_ = nn.Identity()
        self.time_entity_vectors = nn.Parameter(torch.empty(self.max_time, self.embedding_dim))
        self.init_f(self.time_entity_vectors, *weight_init_args)

    def compute_embeddings(self, tuples):
        entity_embs, relation_embs = super().compute_embeddings(tuples)
        time_idx = tuples[:, 3]
        time_vecs = self.time_entity_vectors[time_idx, :]
        time_vecs = torch.stack((time_vecs, time_vecs), dim=2)  # apply to both heads and tails
        entity_embs = entity_embs + time_vecs
        return self.embedding_norm_fn_(entity_embs), self.embedding_norm_fn_(relation_embs), None


class DEBoxE_TimeBump(BaseBoxE):
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, device='cpu',
                 weight_init_args=(0, 1), norm_embeddings=False, activation='sine', time_weight=0.5):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings=False)
        if norm_embeddings:
            self.embedding_norm_fn_ = nn.Tanh()
        else:
            self.embedding_norm_fn_ = nn.Identity()
        if activation == 'sine':
            self.activation_fn = torch.sin
        if activation == 'sigmoid':
            self.activation_fn = torch.sigmoid
        self.time_weight = time_weight
        self.time_bumps = nn.Embedding(self.nb_entities, self.embedding_dim)
        self.time_w = nn.Embedding(self.nb_entities, self.embedding_dim)
        self.time_b = nn.Embedding(self.nb_entities, self.embedding_dim)
        self.init_f(self.time_bumps.weight, *weight_init_args)
        self.init_f(self.time_w.weight, *weight_init_args)
        self.init_f(self.time_b.weight, *weight_init_args)

    def compute_embeddings(self, tuples):
        entity_embs, relation_embs = super().compute_embeddings(tuples)
        e_h_idx = self.get_e_idx_by_id(tuples[:, 0]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[:, 2]).to(self.device)
        time_idx = tuples[:, 3]
        time_idx = torch.stack([time_idx for _ in range(self.embedding_dim)], dim=2)
        head_time_bumps = self.time_bumps(e_h_idx) * self.activation_fn(self.time_w(e_h_idx)*time_idx + self.time_b(e_h_idx))
        tail_time_bumps = self.time_bumps(e_t_idx) * self.activation_fn(self.time_w(e_t_idx)*time_idx + self.time_b(e_t_idx))
        time_vecs = torch.stack((head_time_bumps, tail_time_bumps), dim=2)
        entity_embs = entity_embs + self.time_weight * time_vecs
        return self.embedding_norm_fn_(entity_embs), self.embedding_norm_fn_(relation_embs), None


class DEBoxE_EntityEmb(BaseBoxE):
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, time_proportion, activation='sine', device='cpu',
                 weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings=False)
        if norm_embeddings:
            self.embedding_norm_fn_ = nn.Tanh()
        else:
            self.embedding_norm_fn_ = nn.Identity()
        if activation == 'sine':
            self.activation_fn = torch.sin
        if activation == 'sigmoid':
            self.activation_fn = torch.sigmoid
        self.time_features_mask = (torch.arange(self.embedding_dim, device=self.device) < (time_proportion * self.embedding_dim))
        self.time_w = nn.Parameter(torch.empty((self.nb_entities, embedding_dim), device=self.device))
        self.time_b = nn.Parameter(torch.empty((self.nb_entities, embedding_dim), device=self.device))
        nn.init.normal_(self.time_w, 0, 0.5)
        nn.init.normal_(self.time_b, 0, 0.5)

    def compute_embeddings(self, tuples):
        entity_embs, relation_embs = super().compute_embeddings(tuples)
        nb_examples, _, batch_size = tuples.shape
        time_idx = tuples[:, 3]
        time_idx = torch.stack([time_idx for _ in range(self.embedding_dim)], dim=2)
        e_h_idx = self.get_e_idx_by_id(tuples[:, 0]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[:, 2]).to(self.device)
        head_embeddings = entity_embs[:, :, 0, :]
        tail_embeddings = entity_embs[:, :, 1, :]
        mask = self.time_features_mask.repeat((nb_examples, batch_size, 1))
        head_time_features = mask * head_embeddings * self.activation_fn(
            time_idx * self.time_w[e_h_idx, :] + self.time_b[e_h_idx, :])
        head_static_features = ~mask * head_embeddings
        tail_time_features = mask * tail_embeddings * self.activation_fn(
            time_idx * self.time_w[e_t_idx, :] + self.time_b[e_t_idx, :])
        tail_static_features = ~mask * tail_embeddings
        time_features = torch.stack((head_time_features, tail_time_features), dim=2)
        static_features = torch.stack((head_static_features, tail_static_features), dim=2)
        entity_embs = time_features + static_features
        return self.embedding_norm_fn_(entity_embs), self.embedding_norm_fn_(relation_embs), None


class DEBoxE_EntityBump(BaseBoxE):
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, time_proportion, activation='sine', device='cpu',
                 weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings=False)
        if norm_embeddings:
            self.embedding_norm_fn_ = nn.Tanh()
        else:
            self.embedding_norm_fn_ = nn.Identity()
        if activation == 'sine':
            self.activation_fn = torch.sin
        if activation == 'sigmoid':
            self.activation_fn = torch.sigmoid
        self.time_features_mask = (torch.arange(self.embedding_dim, device=self.device) < (time_proportion * self.embedding_dim))
        self.time_w = nn.Parameter(torch.empty((self.nb_entities, embedding_dim), device=self.device))
        self.time_b = nn.Parameter(torch.empty((self.nb_entities, embedding_dim), device=self.device))
        nn.init.normal_(self.time_w, 0, 0.5)
        nn.init.normal_(self.time_b, 0, 0.5)

    def compute_embeddings(self, tuples):
        _, relation_embs = super().compute_embeddings(tuples)
        nb_examples, _, batch_size = tuples.shape
        e_h_idx = self.get_e_idx_by_id(tuples[:, 0]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[:, 2]).to(self.device)
        time_idx = tuples[:, 3]
        time_idx = torch.stack([time_idx for _ in range(self.embedding_dim)], dim=2)
        # get relevant entity bases and bumps
        head_bases = self.entity_bases(e_h_idx)
        head_bumps = self.entity_bumps(e_h_idx)
        tail_bases = self.entity_bases(e_t_idx)
        tail_bumps = self.entity_bumps(e_t_idx)
        # perform DE on bump vectors
        mask = self.time_features_mask.repeat((nb_examples, batch_size, 1))
        head_bump_time_features = mask * head_bumps * self.activation_fn(
            time_idx * self.time_w[e_h_idx, :] + self.time_b[e_h_idx, :])
        head_bump_static_features = ~mask * head_bumps
        tail_bump_time_features = mask * tail_bumps * self.activation_fn(
            time_idx * self.time_w[e_t_idx, :] + self.time_b[e_t_idx, :])
        tail_bump_static_features = ~mask * tail_bumps
        head_bumps = head_bump_time_features + head_bump_static_features
        tail_bumps = tail_bump_time_features + tail_bump_static_features
        entity_embs = self.embedding_norm_fn(torch.stack((head_bases + tail_bumps, tail_bases + head_bumps), dim=2))
        return entity_embs, self.embedding_norm_fn_(relation_embs), None


class DEBoxE_EntityBase(BaseBoxE):
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, time_proportion, activation='sine', device='cpu',
                 weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings=False)
        if norm_embeddings:
            self.embedding_norm_fn_ = nn.Tanh()
        else:
            self.embedding_norm_fn_ = nn.Identity()
        if activation == 'sine':
            self.activation_fn = torch.sin
        if activation == 'sigmoid':
            self.activation_fn = torch.sigmoid
        self.time_features_mask = (torch.arange(self.embedding_dim, device=self.device) < (time_proportion * self.embedding_dim))
        self.time_w = nn.Parameter(torch.empty((self.nb_entities, embedding_dim), device=self.device))
        self.time_b = nn.Parameter(torch.empty((self.nb_entities, embedding_dim), device=self.device))
        nn.init.normal_(self.time_w, 0, 0.5)
        nn.init.normal_(self.time_b, 0, 0.5)

    def compute_entity_embeddings(self, tuples):
        nb_examples, _, batch_size = tuples.shape
        e_h_idx = self.get_e_idx_by_id(tuples[:, 0]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[:, 2]).to(self.device)
        time_idx = tuples[:, 3]
        time_idx = torch.stack([time_idx for _ in range(self.embedding_dim)], dim=2)
        # get relevant entity bases and bumps
        head_bases = self.entity_bases(e_h_idx)
        head_bumps = self.entity_bumps(e_h_idx)
        tail_bases = self.entity_bases(e_t_idx)
        tail_bumps = self.entity_bumps(e_t_idx)
        # perform DE on bump vectors
        mask = self.time_features_mask.repeat((nb_examples, batch_size, 1))
        head_base_time_features = mask * head_bases * self.activation_fn(
            time_idx * self.time_w[e_h_idx, :] + self.time_b[e_h_idx, :])
        head_base_static_features = ~mask * head_bases
        tail_base_time_features = mask * tail_bases * self.activation_fn(
            time_idx * self.time_w[e_t_idx, :] + self.time_b[e_t_idx, :])
        tail_base_static_features = ~mask * tail_bases
        head_bases = head_base_time_features + head_base_static_features
        tail_bases = tail_base_time_features + tail_base_static_features
        return self.embedding_norm_fn(torch.stack((head_bases + tail_bumps, tail_bases + head_bumps), dim=2))

    def compute_embeddings(self, tuples):
        _, relation_embs = super().compute_embeddings(tuples)
        entity_embs = self.compute_entity_embeddings(tuples)
        return entity_embs, self.embedding_norm_fn_(relation_embs), None


class DEBoxE_BaseM(BaseBoxE):
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, time_proportion, activation='sine',
                 device='cpu',
                 weight_init_args=(0, 1), norm_embeddings=False):
        super(DEBoxE_BaseM, self).__init__(embedding_dim, relation_ids, entity_ids, timestamps, device,
                                           weight_init_args, norm_embeddings)
        # delete not needed inherited parameters to minimize memory overhead
        self.entity_bases, self.entity_bumps = None, None

        self.de_boxe = DEBoxE_EntityBase(embedding_dim, relation_ids, entity_ids, timestamps, time_proportion, activation, device,
                                         weight_init_args, norm_embeddings)
        self.de_boxe.r_head_base_points, self.de_boxe.r_head_widths, self.de_boxe.r_head_size_scales = None, None, None
        self.de_boxe.r_tail_base_points, self.de_boxe.r_tail_widths, self.de_boxe.r_tail_size_scales = None, None, None
        self.de_boxe.r_head_base_points, self.de_boxe.r_head_widths, self.de_boxe.r_head_size_scales = None, None, None
        self.de_boxe.r_tail_base_points, self.de_boxe.r_tail_widths, self.de_boxe.r_tail_size_scales = None, None, None

        self.tempboxe_m = TempBoxE_M(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args,
                                     norm_embeddings)
        self.tempboxe_m.entity_bases, self.tempboxe_m.entity_bumps = None, None
        self.tempboxe_m.r_head_base_points, self.tempboxe_m.r_head_widths, self.tempboxe_m.r_head_size_scales = None, None, None
        self.tempboxe_m.r_tail_base_points, self.tempboxe_m.r_tail_widths, self.tempboxe_m.r_tail_size_scales = None, None, None
        self.tempboxe_m.r_head_base_points, self.tempboxe_m.r_head_widths, self.tempboxe_m.r_head_size_scales = None, None, None
        self.tempboxe_m.r_tail_base_points, self.tempboxe_m.r_tail_widths, self.tempboxe_m.r_tail_size_scales = None, None, None

    def compute_embeddings(self, tuples):
        relation_embs = super().compute_relation_embeddings(tuples)
        entity_embs = self.de_boxe.compute_entity_embeddings(tuples)
        time_box_embs = self.tempboxe_m.compute_time_embeddings(tuples)
        return entity_embs, relation_embs, time_box_embs


class TimeLSTM(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, output_dim):
        super(TimeLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, time_embeddings):
        max_time, embedding_dim = time_embeddings.shape
        out, _ = self.lstm(time_embeddings.view(max_time, 1, -1))  # shape (max_time, 1, hidden_dim)
        return self.linear(out.view(max_time, -1))


class TempBoxE_SLSTM_Plus(TempBoxE_SMLP_Plus):
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, nn_depth=3, nn_width=300,
                     lookback=1, device='cpu', weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, nn_depth, nn_width,
                             lookback, device, weight_init_args, norm_embeddings)
        self.time_transition = TimeLSTM(hidden_dim=nn_width, embedding_dim=self.embedding_dim,
                                        output_dim=4*self.embedding_dim)
        self.to(device)

    def unroll_time(self, init_head_boxes, init_tail_boxes):  # arguments just for compatibility with base class
        embs = self.time_embeddings(torch.arange(self.max_time, device=self.device))  # get all time embeddings
        time_boxes_flat = self.time_transition(embs)
        _, num_box_embs = time_boxes_flat.shape
        heads, tails = time_boxes_flat[:, :int(num_box_embs/2)], time_boxes_flat[:, int(num_box_embs/2):]
        return nn.Embedding.from_pretrained(heads), nn.Embedding.from_pretrained(tails)


class TempBoxE_RLSTM_Plus(TempBoxE_RMLP_Plus):
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, nn_depth=3, nn_width=300,
                     lookback=1, device='cpu', weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, nn_depth, nn_width,
                             lookback, device, weight_init_args, norm_embeddings)
        self.time_transition = TimeLSTM(hidden_dim=nn_width, embedding_dim=self.embedding_dim,
                                           output_dim=4*self.embedding_dim*self.nb_relations)
        self.to(device)

    def unroll_time(self, init_head_boxes, init_tail_boxes):  # arguments just for compatibility with base class
        embs = self.time_embeddings(torch.arange(self.max_time, device=self.device))  # get all time embeddings
        relation_boxes_flat = self.time_transition(embs)
        num_timesteps, num_box_embs = relation_boxes_flat.shape
        heads, tails = relation_boxes_flat[:, :int(num_box_embs/2)], relation_boxes_flat[:, int(num_box_embs/2):]
        return heads.view(num_timesteps, self.nb_relations, -1), tails.view(num_timesteps, self.nb_relations, -1)


class TempBoxE_MLSTM_Plus(BaseBoxE):
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, nn_depth=3, nn_width=300,
                     lookback=1, device='cpu', weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings)
        self.time_embeddings = nn.Embedding(self.max_time, embedding_dim)
        self.init_f(self.time_embeddings.weight, *weight_init_args)
        self.time_transition = TimeLSTM(hidden_dim=nn_width, embedding_dim=self.embedding_dim,
                                           output_dim=4*self.embedding_dim*self.nb_relations)
        self.to(device)

    def unroll_time(self):
        embs = self.time_embeddings(torch.arange(self.max_time, device=self.device))  # get all time embeddings
        relation_boxes_flat = self.time_transition(embs)
        num_timesteps, num_box_embs = relation_boxes_flat.shape
        heads, tails = relation_boxes_flat[:, :int(num_box_embs/2)], relation_boxes_flat[:, int(num_box_embs/2):]
        return heads.view(num_timesteps, self.nb_relations, -1), tails.view(num_timesteps, self.nb_relations, -1)

    def compute_embeddings(self, tuples):
        entity_embs, relation_embs = super().compute_embeddings(tuples)
        nb_examples, _, batch_size = tuples.shape
        rel_idx = self.get_r_idx_by_id(tuples[:, 1]).to(self.device)
        time_idx = tuples[:, 3]
        all_r_head_boxes, all_r_tail_boxes = self.unroll_time()  # shape (timestamp, relation, 2*embedding_dim)
        time_head_boxes = all_r_head_boxes[time_idx, rel_idx, :].view((nb_examples, batch_size, 2, self.embedding_dim))
        time_tail_boxes = all_r_tail_boxes[time_idx, rel_idx, :].view((nb_examples, batch_size, 2, self.embedding_dim))
        return entity_embs, relation_embs, self.embedding_norm_fn(torch.stack((time_head_boxes, time_tail_boxes), dim=2))
