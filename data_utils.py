import torch
import numpy as np


class Temp_kg_loader():  # wrapper for dataloader
    ## TODO check ids need to be consistend across train, test, val

    '''
    @param truncate positive int indicating how many training examples to consider. Negative int uses all examples.
    @param data_format: some datasets (i.e. the temporal ones) are in a format that need special parsing.
      That format can be specified via this string
    '''

    def __init__(self, train_path, test_path, valid_path, truncate=-1, data_format='', no_time_info=False, device='cpu'):
        self.device = device
        self.train_data_raw = self.parse_dataset(train_path, truncate, no_time_info=no_time_info)
        self.test_data_raw = self.parse_dataset(test_path, truncate, no_time_info=no_time_info)
        self.valid_data_raw = self.parse_dataset(valid_path, truncate, no_time_info=no_time_info)
        self.entity_ids, self.relation_ids, self.id_to_name, self.name_to_id = self.compute_ids()
        self.train_data = self.dates_to_days(
            [(self.name_to_id[h], self.name_to_id[r], self.name_to_id[t], temp) for (h, r, t, temp) in
             self.train_data_raw])
        self.test_data = self.dates_to_days(
            [(self.name_to_id[h], self.name_to_id[r], self.name_to_id[t], temp) for (h, r, t, temp) in
             self.test_data_raw])
        self.valid_data = self.dates_to_days(
            [(self.name_to_id[h], self.name_to_id[r], self.name_to_id[t], temp) for (h, r, t, temp) in
             self.valid_data_raw])
        self.train_data_no_timestamps = [(h, r, t) for (h, r, t, _) in self.train_data]
        self.test_data_no_timestamps = [(h, r, t) for (h, r, t, _) in self.test_data]
        self.valid_data_no_timestamps = [(h, r, t) for (h, r, t, _) in self.valid_data]
        self.max_time_train = max([time for [_, _, _, time] in self.train_data])
        self.train_fact_set = set(self.train_data)
        self.train_fact_set_no_timestamps = set(self.train_data_no_timestamps)
        self.fact_set = set(self.test_data + self.valid_data)

    def get_testloader(self, data='ids', **kwargs):
        if data == 'ids':
            return torch.utils.data.DataLoader(dataset=self.test_data, **kwargs)
        if data == 'raw':
            return torch.utils.data.Dataloader(dataset=self.test_data_raw, **kwargs)

    def get_trainloader(self, data='ids', **kwargs):
        if data == 'ids':
            return torch.utils.data.DataLoader(dataset=self.train_data, **kwargs)
        if data == 'raw':
            return torch.utils.data.Dataloader(dataset=self.train_data_raw, **kwargs)

    def get_validloader(self, data='ids', **kwargs):
        if data == 'ids':
            return torch.utils.data.DataLoader(dataset=self.valid_data, **kwargs)
        if data == 'raw':
            return torch.utils.data.Dataloader(dataset=self.valid_data_raw, **kwargs)

    def get_combined_loader(self, datasets, data='ids', **kwargs):
        d = []
        if data == 'ids':
            if 'train' in datasets:
                d += self.train_data
            if 'val' in datasets or 'valid' in datasets:
                d += self.valid_data
            if 'test' in datasets:
                d += self.test_data
            return torch.utils.data.DataLoader(dataset=d, **kwargs)
        if data == 'raw':
            if 'train' in datasets:
                d += self.train_data_raw
            if 'val' in datasets or 'valid' in datasets:
                d += self.valid_data_raw
            if 'test' in datasets:
                d += self.test_data_raw
            return torch.utils.data.DataLoader(dataset=d, **kwargs)

    def compute_ids(self):
        id_to_name = dict()
        name_to_id = dict()
        e_ids = []
        r_ids = []
        # it is internal convention to have entity ids start at 0; don't change!
        id = 0
        for e_name in self.get_entity_names():
            id_to_name[id] = e_name
            name_to_id[e_name] = id
            e_ids.append(id)
            id += 1
        for r_name in self.get_relation_names():
            id_to_name[id] = r_name
            name_to_id[r_name] = id
            r_ids.append(id)
            id += 1
        return e_ids, r_ids, id_to_name, name_to_id

    def parse_dataset(self, path_to_file, truncate=-1, data_format='', no_time_info=False):
        tuples = []
        with open(path_to_file, 'r') as f:
            lines = f.read().splitlines()
            if truncate > 0:
                lines = lines[:truncate]
            for line in lines:
                line_split = tuple(line.split('\t'))
                if no_time_info or len(line_split) == 3:  # add dummy time information
                    h, r, t = line_split
                    line_split = (h, r, t, '0000-00-00')
                    # data_format = 'ICEWS'
                tuples.append(line_split)
        if data_format == 'ICEWS':
            tuples = self.dates_to_days(tuples)
        return tuples

    '''
    Transform ICEWS timestamps to days, where the earliest day in the dataset is day 0
    '''

    def dates_to_days(self, data_tuples):
        cumm_days_year_1500 = 548229  # hard coded base case avoids exceeding max recursion depth
        stamp_to_nums = lambda x: list(map(int, x.split('-')))
        is_leap = lambda x: True if (x % 4 == 0 and x % 100 != 0) or x % 400 == 0 else False  # algorithm from Wikipedia
        days_per_year = lambda x: 366 if is_leap(x) else 365
        days_per_month = lambda year, month: 29 if (month == 2 and is_leap(year)) else \
        [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month]
        cumm_days_year = lambda year: 0 if year < 0 else cumm_days_year_1500 if year == 1500\
            else days_per_year(year) + cumm_days_year(year - 1)
        cumm_days_month = lambda year, month: 0 if month == 0 else days_per_month(year, month) + cumm_days_month(year,
                                                                                                                 month - 1)
        calculate_days = lambda l: cumm_days_year(0 if l[0] == 0 else l[0] - 1) + cumm_days_month(l[0], l[1]) + l[2]
        data_days = [(h, r, t, calculate_days(stamp_to_nums(stamp))) for (h, r, t, stamp) in data_tuples]
        offset = min([d for (_, _, _, d) in data_days])
        return [(h, r, t, d - offset) for (h, r, t, d) in data_days]

    def get_entity_names(self):
        # unsorted list of entity names
        train_head_names = set([h for (h, _, _, _) in self.train_data_raw])
        other_names = []
        other_names.append(set([t for (_, _, t, _) in self.train_data_raw]))
        other_names.append(set([h for (h, _, _, _) in self.test_data_raw]))
        other_names.append(set([t for (_, _, t, _) in self.test_data_raw]))
        other_names.append(set([h for (h, _, _, _) in self.valid_data_raw]))
        other_names.append(set([t for (_, _, t, _) in self.valid_data_raw]))
        return list(train_head_names.union(*other_names))

    def get_relation_names(self):
        # unsorted list of relation names
        r_train = set([r for (_, r, _, _) in self.train_data_raw])
        other_rs = []
        other_rs.append(set([r for (_, r, _, _) in self.test_data_raw]))
        other_rs.append(set([r for (_, r, _, _) in self.valid_data_raw]))
        return list(r_train.union(*other_rs))

    def get_timestamps(self):
        return [time for (_, _, _, time) in self.train_data] + [time for (_, _, _, time) in self.test_data] + [time for
                                                                                                               (_, _, _,
                                                                                                                time) in
                                                                                                               self.valid_data]

    '''
    @param sampling_mode: 'd','dependent' -> time Dependent sampling; 'a','agnostic' -> time Agnostic sampling (see HyTE paper for details)
    '''

    def needs_resample(self, sample, sampling_mode):
        if sampling_mode in ['a', 'agnostic']:
            return (sample[0], sample[1], sample[2]) in self.train_fact_set_no_timestamps
        if sampling_mode in ['d', 'dependent']:
            return (sample[0], sample[1], sample[2], sample[3]) in self.train_fact_set
        raise ValueError("Invalid sampling mode. Use 'd' or 'a'")

    '''
    @param sampling_mode: 'd','dependent' -> time Dependent sampling; 'a','agnostic' -> time Agnostic sampling (see HyTE paper for details)
    '''

    def resample(self, tuples, sampling_mode):
        max_e_id = len(self.entity_ids)
        no_replacement_performed = True
        for i, head in enumerate(tuples[0]):
            if self.needs_resample([head, tuples[1, i], tuples[2, i], tuples[3, i]], sampling_mode):
                no_replacement_performed = False
                new_e = torch.randint(max_e_id, (1,))
                is_head = torch.randint(2, (1,)) == 1
                if head.item():
                    tuples[0, i] = new_e.item()
                else:
                    tuples[2, i] = new_e.item()
        return tuples, no_replacement_performed

    '''
    @param sampling_mode: 'd','dependent' -> time Dependent sampling; 'a','agnostic' -> time Agnostic sampling (see HyTE paper for details)
    '''

    def sample_negatives(self, tuples, nb_samples, sampling_mode='d'):
        batch_size = len(tuples[0])
        # tuples_rep = torch.repeat_interleave(torch.stack(tuples), nb_samples, dim=1)
        tuples_rep = torch.repeat_interleave(tuples, nb_samples, dim=1)
        # we assume entity ids to start at 0
        max_e_id = len(self.entity_ids)
        # sample random entities
        sample_ids = torch.randint(max_e_id, size=(batch_size * nb_samples,)).to(self.device)
        is_head = (torch.randint(2, size=(
        batch_size * nb_samples,)) == 1)  # indicate if head is being replaced (otherwise, replace tail)
        # create sampled triples from sampled entities
        replace_mask = torch.stack((is_head, torch.zeros(len(is_head)), ~is_head, torch.zeros(len(is_head)))).to(self.device)
        inverse_replace_mask = torch.stack((~is_head, torch.ones(len(is_head)), is_head, torch.ones(len(is_head)))).to(
            self.device)
        replacements = torch.stack((sample_ids, tuples_rep[1], sample_ids, tuples_rep[3])).to(self.device)
        sampled_tuples = replace_mask * replacements + inverse_replace_mask * tuples_rep
        # filter out and replace known positive triples
        filtering_done = False
        while not filtering_done:
            sampled_triples, filtering_done = self.resample(sampled_tuples, sampling_mode)
        return sampled_tuples.reshape((4, batch_size, nb_samples)).long()

    def compute_filter_idx(self, tuples):
        idx = torch.ones_like(tuples[0])
        for i, l in enumerate(tuples[0]):
            for j, head in enumerate(tuples[0, i]):
                if (tuples[0, i, j], tuples[1, i, j], tuples[2, i, j], tuples[3, i, j]) in self.fact_set:
                    idx[i, j] = 0
                    idx[i, j] = 0
                    idx[i, j] = 0
                    idx[i, j] = 0
        return idx

    '''
    Replaces head by all other entities and filters out known positives
    @return tensor of shape (3, batch_size, nb_entities) where filtered out triples contain -1
    '''

    def corrupt_head(self, tuples):
        batch_size = len(tuples[0])
        # we assume entity ids to start at 0
        max_e_id = len(self.entity_ids)
        tuples_rep = torch.repeat_interleave(tuples, max_e_id, dim=1)
        l = len(tuples_rep[0])
        e_permutations = torch.repeat_interleave(torch.from_numpy(np.arange(max_e_id)), batch_size).to(self.device)

        replace_mask = torch.stack((torch.ones(l), torch.zeros(l), torch.zeros(l), torch.zeros(l))).to(self.device)
        inverse_replace_mask = torch.stack((torch.zeros(l), torch.ones(l), torch.ones(l), torch.ones(l))).to(self.device)
        replacements = torch.stack((e_permutations, tuples_rep[1], e_permutations, tuples_rep[3])).to(self.device)
        sampled_tuples = replace_mask * replacements + inverse_replace_mask * tuples_rep

        sampled_tuples = sampled_tuples.reshape((4, batch_size, max_e_id)).long()
        filter_idx = self.compute_filter_idx(
            sampled_tuples)  # indices of the tuples that are positive facts and get filtered out
        return sampled_tuples, filter_idx

    '''
    Replaces head by all other entities and filters out known positives
    @return tensor of shape (3, batch_size, nb_entities) where filtered out triples contain -1
    '''

    def corrupt_tail(self, tuples):
        batch_size = len(tuples[0])
        # we assume entity ids to start at 0
        max_e_id = len(self.entity_ids)
        tuples_rep = torch.repeat_interleave(tuples, max_e_id, dim=1)
        l = len(tuples_rep[0])
        e_permutations = torch.repeat_interleave(torch.from_numpy(np.arange(max_e_id)), batch_size).to(self.device)

        replace_mask = torch.stack((torch.zeros(l), torch.zeros(l), torch.ones(l), torch.ones(l))).to(self.device)
        inverse_replace_mask = torch.stack((torch.ones(l), torch.ones(l), torch.zeros(l), torch.zeros(l))).to(self.device)
        replacements = torch.stack((e_permutations, tuples_rep[1], e_permutations, tuples_rep[3])).to(self.device)
        sampled_tuples = replace_mask * replacements + inverse_replace_mask * tuples_rep

        sampled_tuples = sampled_tuples.reshape((4, batch_size, max_e_id)).long()
        filter_idx = self.compute_filter_idx(
            sampled_tuples)  # indices of the tuples that are positive facts and get filtered out
        return sampled_tuples, filter_idx

    def to(self, device):
        self.device = device
