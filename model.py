import math

import torch
import torch.nn as nn
import copy


class BaseBoxE(nn.Module):
    """
    Base class for BoxTE, BoxE, TBoxE, and DEBoxE
    """
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

    def forward(self, positives, negatives):
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
        positive_emb = self.forward_positives(positives)
        negative_emb = self.forward_negatives(negatives)
        return positive_emb, negative_emb


class BoxE(BaseBoxE):
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, device='cpu', weight_init_args=(0, 1), norm_embeddings=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings)

    def compute_embeddings(self, tuples):
        entity_embs, relation_embs = super().compute_embeddings(tuples)
        return entity_embs, relation_embs, None


class TBoxE(BaseBoxE):
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


def to_cartesian(vecs, device):
    nb_examples, batch_size, nb_timebumps, emb_dim = vecs.shape
    og_shape = (nb_examples, batch_size, nb_timebumps, emb_dim)
    flat_shape = (nb_examples * batch_size * nb_timebumps, emb_dim)
    vecs = vecs.view(flat_shape)

    r = vecs[:, 0]
    angles = vecs[:, 1:]
    cos_vec = angles.cos()
    sin_vec = angles.sin()
    xs = []
    running_sin = torch.ones(len(vecs), device=device)
    for i_a, a in enumerate(angles.t()):  # iterate over embedding_dim-1
        xs.append(r * running_sin * cos_vec[:, i_a])
        running_sin = running_sin * sin_vec[:, i_a].clone()
    xs.append(r * running_sin)
    return torch.stack(xs, dim=1).view(og_shape)


def to_angle_interval(angles):
    '''maps angles to [0, 2*pi) interval '''
    angles_by_twopi = angles/(2*math.pi)
    return (angles_by_twopi - torch.floor(angles_by_twopi)) * 2 * math.pi


class BoxTE(BaseBoxE):
    def __init__(self, embedding_dim, relation_ids, entity_ids, timestamps, device='cpu',
                 weight_init_args=(0, 1), norm_embeddings=False, time_weight=1, use_r_factor=False, use_e_factor=False,
                 nb_timebumps=1, use_r_rotation=False, use_e_rotation=False, nb_time_basis_vecs=-1,
                 norm_time_basis_vecs=False, use_r_t_factor=False, dropout_p=0.0, arity_spec_timebumps=False):
        super().__init__(embedding_dim, relation_ids, entity_ids, timestamps, device, weight_init_args, norm_embeddings=False)
        if norm_embeddings:
            self.embedding_norm_fn_ = nn.Tanh()
        else:
            self.embedding_norm_fn_ = nn.Identity()
        self.nb_time_basis_vecs = nb_time_basis_vecs
        self.norm_time_basis_vecs = norm_time_basis_vecs
        self.use_r_factor, self.use_e_factor, self.use_r_t_factor = use_r_factor, use_e_factor, use_r_t_factor
        self.use_r_rotation, self.use_e_rotation = use_r_rotation, use_e_rotation
        self.nb_timebumps = nb_timebumps
        self.time_weight = time_weight
        self.droput_p = dropout_p
        self.droput = nn.Dropout(dropout_p)
        self.arity_spec_timebumps = arity_spec_timebumps
        if not self.nb_time_basis_vecs > 0:  # don't factorize time bumps, learn them directly/explicitly
            self.factorize_time = False
            if self.arity_spec_timebumps:
                self.head_time_bumps = nn.Parameter(torch.empty(self.max_time, self.nb_timebumps, self.embedding_dim))
                self.tail_time_bumps = nn.Parameter(torch.empty(self.max_time, self.nb_timebumps, self.embedding_dim))
                self.init_f(self.head_time_bumps, *weight_init_args)
                self.init_f(self.tail_time_bumps, *weight_init_args)
            else:
                self.time_bumps = nn.Parameter(torch.empty(self.max_time, self.nb_timebumps, self.embedding_dim))
                self.init_f(self.time_bumps, *weight_init_args)
        else:  # factorize time bumps into two tensors
            self.factorize_time = True
            if self.arity_spec_timebumps:
                self.head_time_bumps_a = nn.Parameter(torch.empty(self.nb_timebumps, self.max_time, self.nb_time_basis_vecs))
                self.head_time_bumps_b = nn.Parameter(
                    torch.empty(self.nb_timebumps, self.nb_time_basis_vecs, self.embedding_dim))
                self.tail_time_bumps_a = nn.Parameter(
                    torch.empty(self.nb_timebumps, self.max_time, self.nb_time_basis_vecs))
                self.tail_time_bumps_b = nn.Parameter(
                    torch.empty(self.nb_timebumps, self.nb_time_basis_vecs, self.embedding_dim))
                self.init_f(self.head_time_bumps_a, *weight_init_args)
                self.init_f(self.head_time_bumps_b, *weight_init_args)
                self.init_f(self.tail_time_bumps_a, *weight_init_args)
                self.init_f(self.tail_time_bumps_b, *weight_init_args)

            else:
                self.time_bumps_a = nn.Parameter(torch.empty(self.nb_timebumps, self.max_time, self.nb_time_basis_vecs))
                self.time_bumps_b = nn.Parameter(torch.empty(self.nb_timebumps, self.nb_time_basis_vecs, self.embedding_dim))
                self.init_f(self.time_bumps_a, *weight_init_args)
                self.init_f(self.time_bumps_b, *weight_init_args)
        if self.use_r_factor:
            if self.arity_spec_timebumps:
                self.head_r_factor = nn.Parameter(torch.empty(self.nb_relations, self.nb_timebumps, 1))
                self.tail_r_factor = nn.Parameter(torch.empty(self.nb_relations, self.nb_timebumps, 1))
                torch.nn.init.normal_(self.head_r_factor, 1, 0.1)
                torch.nn.init.normal_(self.tail_r_factor, 1, 0.1)
            else:
                self.r_factor = nn.Parameter(torch.empty(self.nb_relations, self.nb_timebumps, 1))
                torch.nn.init.normal_(self.r_factor, 1, 0.1)
                self.head_r_factor = self.r_factor
                self.tail_r_factor = self.r_factor
        if self.use_r_t_factor:
            self.r_t_factor = nn.Parameter(torch.empty(self.nb_relations, self.max_time, self.nb_timebumps, 1))
            torch.nn.init.normal_(self.r_t_factor, 1, 0.1)
        if self.use_e_factor:
            self.e_factor = nn.Parameter(torch.empty(self.nb_entities, self.nb_timebumps, 1))
            torch.nn.init.normal_(self.e_factor, 1, 0.1)
        if self.use_r_rotation:
            self.r_angles = nn.Parameter(torch.empty(self.nb_relations, self.nb_timebumps, 1))
            torch.nn.init.normal_(self.r_angles, 0, 0.1)
        if self.use_e_rotation:
            self.e_angles = nn.Parameter(torch.empty(self.nb_entities, self.nb_timebumps, 1))
            torch.nn.init.normal_(self.e_angles, 0, 0.1)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)
        if not self.arity_spec_timebumps:
            if self.use_r_factor:
                self.head_r_factor = self.r_factor
                self.tail_r_factor = self.r_factor
            if not self.factorize_time:
                self.head_time_bumps = self.time_bumps
                self.tail_time_bumps = self.time_bumps
            else:
                self.head_time_bumps_a = self.time_bumps_a
                self.tail_time_bumps_a = self.tail_time_bumps
                self.head_time_bumps_b = self.time_bumps_b
                self.tail_time_bumps_b = self.time_bumps_b

    def dropout_timebump(self, bumps):
        if self.droput_p == 0:
            return bumps
        max_time, nb_timebumps, embedding_dim = bumps.shape
        bump_mask = self.droput(torch.ones(max_time, nb_timebumps, device=self.device))
        if self.training:  # undo dropout scaling, not needed here since we're using a mask
            bump_mask = bump_mask * (1 - self.droput_p)
        bump_mask = bump_mask.unsqueeze(-1).expand(-1, -1, embedding_dim)
        return bumps * bump_mask

    def compute_timebumps(self, is_tail=False, ignore_dropout=False):
        if not self.factorize_time:
            if self.arity_spec_timebumps:
                bumps = self.tail_time_bumps if is_tail else self.head_time_bumps
            else:
                bumps = self.time_bumps
            return bumps if ignore_dropout else self.dropout_timebump(bumps)
        ####
        if self.arity_spec_timebumps:
            bumps_a = self.tail_time_bumps_a if is_tail else self.head_time_bumps_a
            bumps_b = self.tail_time_bumps_b if is_tail else self.tail_time_bumps_b
        else:
            bumps_a = self.time_bumps_a
            bumps_b = self.time_bumps_b
        if self.norm_time_basis_vecs:
            bumps_a = torch.nn.functional.softmax(bumps_a, dim=1)
        bumps = torch.matmul(bumps_a, bumps_b).transpose(0, 1)
        return bumps if ignore_dropout else self.dropout_timebump(bumps)

    def compute_combined_timebumps(self, ignore_dropout=False):
        if not self.arity_spec_timebumps:
            return self.compute_timebumps(ignore_dropout=ignore_dropout)
        else:
            head_bumps = self.compute_timebumps(is_tail=False, ignore_dropout=ignore_dropout)
            tail_bumps = self.compute_timebumps(is_tail=True, ignore_dropout=ignore_dropout)
            return torch.cat((head_bumps, tail_bumps), dim=1)

    def apply_rotation(self, vecs, angles):
        nb_examples, batch_size, nb_timebumps, emb_dim = vecs.shape
        og_shape = (nb_examples, batch_size, nb_timebumps, emb_dim)
        flat_shape = (nb_examples * batch_size * nb_timebumps, emb_dim)
        angles = to_angle_interval(angles)
        angles = torch.cat([angles for _ in range(emb_dim - 1)], dim=3)  # same angle for all dims
        vecs_sph = vecs.view(flat_shape)  # interpret given vectors as spherical coordinates
        vecs_sph[:, 1:] = to_angle_interval(vecs_sph[:, 1:])  # angles need to be in [0, 2pi)
        vecs_sph[:, 1:] += angles.view((nb_examples * batch_size * nb_timebumps, emb_dim-1))  # apply angles
        vecs_sph[:, 1:] = to_angle_interval(vecs_sph[:, 1:])  # angles need to be in [0, 2pi)
        return vecs_sph.view(og_shape).abs()  # radii need to be >= 0

    def index_bumps(self, bumps, idx):
        ''' For atemporal facts, return zero bump; for temporal fact, return appropriate time bump '''
        zeros = torch.zeros(self.nb_timebumps, self.embedding_dim, device=self.device)
        ones = torch.ones(self.nb_timebumps, self.embedding_dim, device=self.device)
        zero_one = torch.stack((zeros, ones))
        mask_idx = torch.where(idx > 0, torch.tensor([1], device=self.device), torch.tensor([0], device=self.device))
        temp_fact_mask = zero_one[mask_idx]
        return bumps[idx] * temp_fact_mask

    def compute_embeddings(self, tuples):
        entity_embs, relation_embs = super().compute_embeddings(tuples)
        time_idx = tuples[:, 3]
        rel_idx = self.get_r_idx_by_id(tuples[:, 1]).to(self.device)
        e_h_idx = self.get_e_idx_by_id(tuples[:, 0]).to(self.device)
        e_t_idx = self.get_e_idx_by_id(tuples[:, 2]).to(self.device)
        time_bumps_h = self.compute_timebumps(is_tail=False)
        time_bumps_t = self.compute_timebumps(is_tail=True)
        time_vecs_h = self.index_bumps(time_bumps_h, time_idx)
        time_vecs_t = self.index_bumps(time_bumps_t, time_idx)
        if self.use_r_rotation:
            time_vecs_h = self.apply_rotation(time_vecs_h, self.r_angles[rel_idx, :, :])
            time_vecs_t = self.apply_rotation(time_vecs_t, self.r_angles[rel_idx, :, :])
        if self.use_e_rotation:
            time_vecs_h = self.apply_rotation(time_vecs_h, self.e_angles[e_h_idx, :, :])
            time_vecs_t = self.apply_rotation(time_vecs_t, self.e_angles[e_t_idx, :, :])
        if self.use_r_rotation or self.use_e_rotation:
            # if rotations are used, we interpret saved time bumps as spherical coordinates
            # so we need to transform to cartesian before applying the bumps
            time_vecs_h = to_cartesian(time_vecs_h, device=self.device)
            time_vecs_t = to_cartesian(time_vecs_t, device=self.device)
        if self.use_r_factor:
            time_vecs_h *= self.head_r_factor[rel_idx, :, :]
            time_vecs_t *= self.tail_r_factor[rel_idx, :, :]
        if self.use_e_factor:
            time_vecs_h *= self.e_factor[e_h_idx, :, :]
            time_vecs_t *= self.e_factor[e_t_idx, :, :]
        if self.use_r_t_factor:
            time_vecs_h *= self.r_t_factor[rel_idx, time_idx, :, :]
            time_vecs_t *= self.r_t_factor[rel_idx, time_idx, :, :]
        time_vecs_h, time_vecs_t = time_vecs_h.sum(dim=2), time_vecs_t.sum(dim=2)  # sum over all time bumps
        time_vecs = torch.stack((time_vecs_h, time_vecs_t), dim=2)  # apply to both heads and tails
        entity_embs = entity_embs + self.time_weight * time_vecs
        return self.embedding_norm_fn_(entity_embs), self.embedding_norm_fn_(relation_embs), None


class DEBoxE(BaseBoxE):
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
