import logging
import torch
import numpy as np
import numbers


class TempKgLoader():
    """Loads datasets, holds data, provides dataloaders, and samples negative/corrupted facts"""

    def __init__(self, train_path, test_path, valid_path, truncate=-1, data_format='', no_time_info=False, device='cpu', entity_subset=-1, kg_is_static=False):
        """
        @param truncate positive int indicating how many training examples to consider. Negative int uses all examples.
        @param data_format: some datasets (i.e. the temporal ones) are in a format that need special parsing.
          That format can be specified via this string
        """
        self.device = device
        self.kg_is_static = kg_is_static
        self.train_data_raw = self.parse_dataset(train_path, truncate, no_time_info=no_time_info)
        self.test_data_raw = self.parse_dataset(test_path, truncate, no_time_info=no_time_info)
        self.valid_data_raw = self.parse_dataset(valid_path, truncate, no_time_info=no_time_info)
        if entity_subset > 0:
            self.subset_data_by_entities(entity_subset)
        self.entity_ids, self.relation_ids, self.id_to_name, self.name_to_id = self.compute_ids()
        if data_format == 'YAGO':
            self.max_time, self.min_time = self.get_max_min_time(self.train_data_raw + self.test_data_raw + self.valid_data_raw)
            self.year_to_id, self.id_to_year = self.compute_year_ids()
        if data_format == 'ICEWS' or data_format == 'GDELT':
            self.train_data = self.dates_to_days(
                [(self.name_to_id[h], self.name_to_id[r], self.name_to_id[t], temp) for (h, r, t, temp) in
                self.train_data_raw])
            self.test_data = self.dates_to_days(
                [(self.name_to_id[h], self.name_to_id[r], self.name_to_id[t], temp) for (h, r, t, temp) in
                self.test_data_raw])
            self.valid_data = self.dates_to_days(
                [(self.name_to_id[h], self.name_to_id[r], self.name_to_id[t], temp) for (h, r, t, temp) in
                self.valid_data_raw])
        elif data_format == 'YAGO':
            self.train_data = [(self.name_to_id[h], self.name_to_id[r], self.name_to_id[t], self.year_to_id[temp])
                               for (h, r, t, temp) in self.train_data_raw]
            self.test_data =[(self.name_to_id[h], self.name_to_id[r], self.name_to_id[t], self.year_to_id[temp])
                             for (h, r, t, temp) in self.test_data_raw]
            self.valid_data = [(self.name_to_id[h], self.name_to_id[r], self.name_to_id[t], self.year_to_id[temp])
                               for (h, r, t, temp) in self.valid_data_raw]
        self.train_data_no_timestamps = [(h, r, t) for (h, r, t, _) in self.train_data]
        self.test_data_no_timestamps = [(h, r, t) for (h, r, t, _) in self.test_data]
        self.valid_data_no_timestamps = [(h, r, t) for (h, r, t, _) in self.valid_data]
        self.max_time_train = max([time for [_, _, _, time] in self.train_data])
        self.train_fact_set = set(self.train_data)
        self.train_fact_set_no_timestamps = set(self.train_data_no_timestamps)
        self.fact_set_no_timestamps = set(self.test_data_no_timestamps + self.valid_data_no_timestamps).union(self.train_fact_set_no_timestamps)
        self.fact_set = set(self.test_data + self.valid_data).union(self.train_fact_set)

    def get_max_min_time(self, tuples):
        max_time = -300
        min_time = 3000
        for h, r, t, time in tuples:
            if time == '0000-00-00':
                continue
            else:
                int_time = int(time)
                if int_time > max_time:
                    max_time = int_time
                if int_time < min_time:
                    min_time = int_time
        return max_time, min_time

    def subset_data_by_entities(self, nb_entities):
        accepted_es = []
        for i, (h, r, t, time) in enumerate(self.train_data_raw):
            if len(accepted_es) >= nb_entities:
                break
            if h not in accepted_es:
                accepted_es.append(h)
            if len(accepted_es) >= nb_entities:
                break
            if t not in accepted_es:
                accepted_es.append(t)
        total_data = self.train_data_raw + self.test_data_raw + self.valid_data_raw
        l_total_data = len(total_data)
        train_prop, test_prop, valid_prop = len(self.train_data_raw)/l_total_data, len(self.test_data_raw)/l_total_data, len(self.valid_data_raw)/l_total_data
        total_filtered_data = [(h,r,t,time) for (h,r,t,time) in total_data if (h in accepted_es and t in accepted_es)]
        self.train_data_raw = total_filtered_data[:int(train_prop*len(total_filtered_data))]
        self.test_data_raw = total_filtered_data[int(train_prop*len(total_filtered_data)):int((train_prop+test_prop)*len(total_filtered_data))]
        self.valid_data_raw = total_filtered_data[int((train_prop+test_prop)*len(total_filtered_data)):]

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

    def compute_year_ids(self):
        id_to_year = dict()
        year_to_id = dict()
        for h, r, t, time in (self.train_data_raw + self.test_data_raw + self.valid_data_raw):
            if time == '0000-00-00':  # dummy time inserted for non-temporal facts
                year_to_id[time] = -1
            else:
                id = int(time) - self.min_time
                id_to_year[id] = time
                year_to_id[time] = id
        return year_to_id, id_to_year

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
                    # make entities and relations different from each other
                    r = '@' + r
                    h, t = '#' + h, '#' + t
                    line_split = (h, r, t, '0000-00-00')
                elif len(line_split) == 4:
                    h, r, t, time = line_split
                    # make entities and relations different from each other
                    r = '@' + r
                    h, t = '#' + h, '#' + t
                    line_split = (h, r, t, time)
                elif line_split == ('',):
                    continue  # empty line can be ignored
                else:
                    raise ValueError('Line is not parsable')
                tuples.append(line_split)
        if data_format == 'ICEWS':
            tuples = self.dates_to_days(tuples)
        return tuples

    def dates_to_days(self, data_tuples):
        """
        Transform ICEWS timestamps to days, where the earliest day in the dataset is day 0
        """
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
        # list of entity names
        name_lists = []
        name_lists.append([fact[0] for fact in self.train_data_raw])
        name_lists.append([fact[2] for fact in self.train_data_raw])
        name_lists.append([fact[0] for fact in self.test_data_raw])
        name_lists.append([fact[2] for fact in self.test_data_raw])
        name_lists.append([fact[0] for fact in self.valid_data_raw])
        name_lists.append([fact[2] for fact in self.valid_data_raw])
        flat_names_list = [item for sublist in name_lists for item in sublist]  # from Alex Martelli on stackoverflow https://stackoverflow.com/a/952952
        return list(dict.fromkeys(flat_names_list))  # remove duplicates, requires python >= 3.7 to maintain ordering

    def get_relation_names(self):
        # list of relation names
        name_lists = []
        name_lists.append([r for (_, r, _, _) in self.train_data_raw])
        name_lists.append([r for (_, r, _, _) in self.test_data_raw])
        name_lists.append([r for (_, r, _, _) in self.valid_data_raw])
        flat_names_list = [item for sublist in name_lists for item in sublist]  # from Alex Martelli on stackoverflow https://stackoverflow.com/a/952952
        return list(dict.fromkeys(flat_names_list))  # remove duplicates, requires python >= 3.7 to maintain ordering

    def get_timestamps(self):
        return [time for (_, _, _, time) in self.train_data] + [time for (_, _, _, time) in self.test_data] + [time for
                                                                                                               (_, _, _,
                                                                                                                time) in
                                                                                                               self.valid_data]

    def resample_agnostic(self, sample):
        return (sample[0], sample[1], sample[2]) in self.train_fact_set_no_timestamps

    def resample_dependent(self, sample):
        return (sample[0], sample[1], sample[2], sample[3]) in self.train_fact_set

    def get_resample_func(self, sampling_mode):
        if sampling_mode in ['a', 'agnostic']:
            return self.resample_agnostic
        if sampling_mode in ['d', 'dependent']:
            return self.resample_dependent

    def needs_resample(self, sample, sampling_mode):
        """
        @:param sampling_mode: 'd','dependent' -> time Dependent sampling; 'a','agnostic' -> time Agnostic sampling (see HyTE paper for details)
        """
        if sampling_mode in ['a', 'agnostic'] or self.kg_is_static:
            return (sample[0], sample[1], sample[2]) in self.train_fact_set_no_timestamps
        if sampling_mode in ['d', 'dependent']:
            return (sample[0], sample[1], sample[2], sample[3]) in self.train_fact_set
        raise ValueError("Invalid sampling mode. Use 'd' or 'a'")

    def resample_known_positives(self, tuples, sampling_mode):
        """
        @:param sampling_mode: 'd','dependent' -> time Dependent sampling; 'a','agnostic' -> time Agnostic sampling (see HyTE paper for details)
        """
        nb_examples, _, batch_size = tuples.shape
        max_e_id = len(self.entity_ids)
        tuples_t = tuples.transpose(1, 2).reshape((nb_examples * batch_size, 4)).cpu().numpy()
        resample_func = self.get_resample_func(sampling_mode)

        def func(row):
            t = (row[0].item(), row[1].item(), row[2].item(), row[3].item())
            if not resample_func(t):
                return row
            while resample_func(t):
                new_e = torch.randint(max_e_id, (1,)).item()
                is_head = torch.randint(2, (1,)) == 1
                if is_head.item():
                    t = ([new_e, row[1], row[2], row[3]])
                else:
                    t = ([row[0], row[1], new_e, row[3]])
            return np.array(t)

        tuples_t = np.apply_along_axis(func, 1, tuples_t)
        if tuples_t.dtype not in ['float64', 'float32', 'float16', 'complex64', 'complex128', 'int64', 'int32', 'int16', 'int8', 'uint8', 'bool']:
            logging.warning('Array dtype not supported. No filtering in this iteration. Dtype: {}'.format(tuples_t.dtype))
            for row in tuples_t:
                for e in row:
                    if not isinstance(e, numbers.Number):
                        logging.info(str(e))
        tuples_t = torch.from_numpy(tuples_t).reshape((nb_examples, batch_size, 4)).transpose(1,2).to(self.device)
        return tuples_t

    def sample_negatives(self, tuples, nb_samples, sampling_mode='d', corrupt_what='e'):
        if corrupt_what == 'e':
            return self.sample_negative_entities(tuples, nb_samples, sampling_mode)
        elif corrupt_what == 't':
            return self.sample_negative_timestamp(tuples, nb_samples, sampling_mode)
        elif corrupt_what in ['et', 'te', 't+e', 'e+t']:
            return self.sample_negative_time_and_entities(tuples, nb_samples, sampling_mode)

    def sample_negative_entities(self, tuples, nb_samples, sampling_mode='d'):
        _, _, batch_size = tuples.shape
        tuples_rep = torch.repeat_interleave(tuples, nb_samples, dim=0)
        max_e_id = len(self.entity_ids)  # we assume entity ids to start at 0
        sample_ids = torch.randint(max_e_id, size=(nb_samples, 1, batch_size)).to(self.device)  # sample random entities
        replacements = torch.cat((sample_ids, tuples_rep[:,1,:].unsqueeze(1), sample_ids, tuples_rep[:,3,:].unsqueeze(1)), dim=1).to(self.device)
        is_head = (torch.randint(2, size=(nb_samples, batch_size)) == 1).unsqueeze(1)  # indicate if head is being replaced (otherwise, replace tail)
        replace_mask = torch.cat((is_head, torch.zeros(nb_samples, 1, batch_size), ~is_head, torch.zeros(nb_samples, 1, batch_size)), dim=1).to(self.device)
        inverse_replace_mask = torch.cat((~is_head, torch.ones(nb_samples, 1, batch_size), is_head, torch.ones(nb_samples, 1, batch_size)), dim=1).to(self.device)
        sampled_tuples = replace_mask * replacements + inverse_replace_mask * tuples_rep
        # filter out and replace known positive triples
        sampled_tuples = self.resample_known_positives(sampled_tuples, sampling_mode)
        return sampled_tuples.long()

    def sample_negative_time_and_entities(self, tuples, nb_samples, sampling_mode='d'):
        _, _, batch_size = tuples.shape
        tuples_rep = torch.repeat_interleave(tuples, nb_samples, dim=0)
        max_e_id = len(self.entity_ids)  # we assume entity ids to start at 0
        sample_ids = torch.randint(max_e_id, size=(nb_samples, 1, batch_size)).to(self.device)  # sample random entities
        replacement_es = torch.cat((sample_ids, tuples_rep[:,1,:].unsqueeze(1), sample_ids, tuples_rep[:,3,:].unsqueeze(1)), dim=1).to(self.device)
        sample_times = torch.randint(self.max_time_train, size=(nb_samples, 1, batch_size)).to(self.device)  # sample random time stamps
        replacement_times = torch.cat(
            (tuples_rep[:, 0, :].unsqueeze(1), tuples_rep[:, 1, :].unsqueeze(1), tuples_rep[:, 2, :].unsqueeze(1), sample_times), dim=1).to(
            self.device)
        rands = torch.randint(3, size=(nb_samples, batch_size))
        is_head = (rands == 1).unsqueeze(1)  # indicate if head is being replaced
        is_tail = (rands == 0).unsqueeze(1)
        is_time = (rands == 2).unsqueeze(1)
        zeros = torch.zeros(nb_samples, 1, batch_size)
        ones = torch.ones(nb_samples, 1, batch_size)
        replace_mask = torch.cat((is_head, zeros, is_tail, torch.zeros(nb_samples, 1, batch_size)), dim=1).to(self.device)
        inverse_replace_mask = torch.cat((~is_head, ones, ~is_tail, torch.ones(nb_samples, 1, batch_size)), dim=1).to(self.device)
        sampled_tuples = replace_mask * replacement_es + inverse_replace_mask * tuples_rep
        time_replace_mask = torch.cat((zeros, zeros, zeros, is_time), dim=1).to(self.device)
        inverse_time_replace_mask = torch.cat((ones, ones, ones, ~is_time), dim=1).to(self.device)
        sampled_tuples = time_replace_mask * replacement_times + inverse_time_replace_mask * sampled_tuples
        # filter out and replace known positive triples
        sampled_tuples = self.resample_known_positives(sampled_tuples, sampling_mode)
        return sampled_tuples.long()

    def sample_negative_timestamp(self, tuples, nb_samples, sampling_mode='d'):
        _, _, batch_size = tuples.shape
        sample_times = torch.randint(self.max_time_train, size=(nb_samples, batch_size)).to(self.device)  # sample random time stamps
        tuples_rep = torch.repeat_interleave(tuples, nb_samples, dim=0)
        tuples_rep[:, 3, :] = sample_times
        # filter out and replace known positive triples
        sampled_tuples = self.resample_known_positives(tuples_rep, sampling_mode)
        return sampled_tuples.long()

    def compute_filter_idx(self, tuples):
        nb_examples, _, batch_size = tuples.shape
        tuples_t = tuples.transpose(1,2).reshape((nb_examples*batch_size, 4)).cpu().numpy()
        if self.kg_is_static:
            func = lambda row: tuple([row[i].item() for i in range(3)]) not in self.fact_set_no_timestamps
        else:
            func = lambda row: tuple([row[i].item() for i in range(4)]) not in self.fact_set
        idx = np.apply_along_axis(func, 1, tuples_t)
        return torch.from_numpy(idx).reshape((nb_examples, batch_size)).to(self.device)

    def corrupt_tuple(self, tuples, head_or_tail, return_batch_size=-1, kg_is_static=False):
        _, _, batch_size = tuples.shape
        max_e_id = len(self.entity_ids) # we assume entity ids to start at 0
        tuples_rep = torch.repeat_interleave(tuples, max_e_id, dim=0)
        sample_ids = torch.arange(max_e_id, device=self.device).repeat([batch_size, 1]).t().unsqueeze(1)  #shape (max_e_id, 1, batch_size)
        replacements = torch.cat((sample_ids, tuples_rep[:,1,:].unsqueeze(1), sample_ids, tuples_rep[:,3,:].unsqueeze(1)), dim=1).to(self.device)
        if head_or_tail in ['head', 'h']:
            is_head = torch.ones((max_e_id, 1, batch_size)) == 1
        elif head_or_tail in ['tail', 't']:
            is_head = torch.zeros((max_e_id, 1, batch_size)) == 1
        else:
            raise ValueError("Argument 'head_or_tail' must be 'h', 'head', 't' or 'tail'")
        replace_mask = torch.cat((is_head, torch.zeros(max_e_id, 1, batch_size), ~is_head, torch.zeros(max_e_id, 1, batch_size)), dim=1).to(self.device)
        inverse_replace_mask = torch.cat((~is_head, torch.ones(max_e_id, 1, batch_size), is_head, torch.ones(max_e_id, 1, batch_size)), dim=1).to(self.device)
        sampled_tuples = (replace_mask * replacements + inverse_replace_mask * tuples_rep).long()

        filter_idx = self.compute_filter_idx(sampled_tuples)
        if return_batch_size > 0:
            return torch.split(sampled_tuples, return_batch_size), torch.split(filter_idx, return_batch_size)
        return (sampled_tuples,), (filter_idx,)
