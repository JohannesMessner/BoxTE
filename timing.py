import time
import logging
import torch

class ExecutionTimer():

    def __init__(self):
        self.active = False

    def activate(self):
        self.active = True
        self.epoch_times = []
        self.sampling_times_epoch = []
        self.for_times_epoch = []
        self.back_times_epoch = []

    def deactivate(self):
        self.active = False

    def log(self, action):
        if not self.active:
            return
        if action == 'start_epoch':
            self.sampling_time = 0
            self.for_time = 0
            self.back_time = 0
            self.epoch_start_time = time.time()
        elif action == 'start_neg_sampling':
            self.sampling_start_time = time.time()
        elif action == 'end_neg_sampling':
            self.sampling_end_time = time.time()
            self.sampling_time += (self.sampling_end_time-self.sampling_start_time)
        elif action == 'start_forward':
            self.for_start_time = time.time()
        elif action == 'end_forward':
            self.for_end_time = time.time()
            self.for_time += (self.for_end_time - self.for_start_time)
        elif action == 'start_backward':
            self.back_start_time = time.time()
        elif action == 'end_backward':
            self.back_end_time = time.time()
            self.back_time += (self.back_end_time - self.back_start_time)
        elif action == 'end_epoch':
            self.sampling_times_epoch.append(self.sampling_time)
            self.for_times_epoch.append(self.for_time)
            self.back_times_epoch.append(self.back_time)
            self.epoch_end_time = time.time()
            self.epoch_times.append(self.epoch_end_time-self.epoch_start_time)
        elif action == 'start_validation':
            logging.info('average epoch time: {} seconds'.format(torch.tensor(self.epoch_times).mean()))
            self.epoch_times = []
            logging.info('average sampling time per epoch: {} seconds'.format(torch.tensor(self.sampling_times_epoch).mean()))
            logging.info('average forward time per epoch: {} seconds'.format(torch.tensor(self.for_times_epoch).mean()))
            logging.info('average backward time per epoch: {} seconds'.format(torch.tensor(self.back_times_epoch).mean()))
            self.sampling_times_epoch = []
            self.metrics_start_time = time.time()
        elif action == 'end_validation':
            self.metrics_end_time = time.time()
            logging.info('metrics calculation time: {} seconds'.format(self.metrics_end_time-self.metrics_start_time))
        else:
            logging.warning('unrecognized action "{}" during timing.'.format(action))


