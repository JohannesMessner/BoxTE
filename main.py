import torch
import numpy as np
import copy
import logging
import os
import argparse
import pprint
import time
import warnings
from datetime import datetime
from metrics import mean_rank
from metrics import mean_rec_rank
from metrics import hits_at_k
from metrics import retrieval_metrics
from metrics import precision
from metrics import recall
from metrics import rank
from model import BoxTEmp
from model import BoxTEmpMLP
from model import BoxELoss
from model import BoxEBinScore
from data_utils import Temp_kg_loader


def parse_args(args):
    # Hyper-Parameters
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--load_params_path', default='',
                        help='Specifies path to model parameters will be loaded. Default initializes a new model.')
    parser.add_argument('--train_path', default='./train.txt',
                        help='Path to training dataset')
    parser.add_argument('--valid_path', default='./valid.txt',
                        help='Path to validation dataset')
    parser.add_argument('--test_path', default='./test.txt',
                        help='Path to test dataset')
    parser.add_argument('--log_filename', default='',
                        help='Filename given to log file. Default prints to stderr. Timestamp added automatically.')
    parser.add_argument('--progress_filename', default='progress',
                        help='Filename given to validation progress data. File extension and timestamp are added automatically.')
    parser.add_argument('--params_filename', default='params',
                        help='Filename given to save model parameters / state dict.  File extension and timestamp are added automatically.')
    parser.add_argument('--results_filename', default='test_results',
                        help='Filename given to final results/metrics. File extension and timestamp are added automatically.')
    parser.add_argument('--info_filename', default='info',
                        help='Filename given to information file. File extension and timestamp are added automatically.')
    parser.add_argument('--log_dir', default='',
                        help='Directory results, params, info and log will be saved into.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Loss margin.')
    parser.add_argument('--num_epochs', default=10, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training batch size.')
    parser.add_argument('--embedding_dim', default=300, type=int,
                        help='Dimensionality of the embedding.')
    parser.add_argument('--learning_rate', default=.0001, type=float,
                        help='Learning rate.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--validation_step', default=500, type=int,
                        help='Number of epochs in between validations.')
    parser.add_argument('--normed_bumps', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--truncate_datasets', default=-1, type=int,
                        help='Truncate datasets to a subset of entries.')
    parser.add_argument('--entity_subset', default=-1, type=int,
                        help='Truncate datasets to a number of entities. Train/test/val split will be maintained.')
    parser.add_argument('--adversarial_temp', default=1, type=float,
                        help='Alpha parameter for adversarial negative sampling loss.')
    parser.add_argument('--loss_type', default='u', type=str,
                        help="Toggle between uniform ('u') and self-adversarial ('a') loss.")
    parser.add_argument('--num_negative_samples', default=10, type=int,
                        help="Number of negative samples per positive (true) triple.")
    parser.add_argument('--weight_init', default='u', type=str,
                        help="Type of weight initialization for the model.")
    parser.add_argument('--weight_init_args', default=[0, 0.5], nargs=2, type=float,
                        help="Parameters to be passed to weight initialization.")
    parser.add_argument('--print_loss_step', default=-1, type=int,
                        help="Number of epochs in between printing of current training loss.")
    parser.add_argument('--neg_sampling_type', default='a', type=str,
                        help="Toggle between time agnostic ('a') and time dependent ('d') negative sampling.")
    parser.add_argument('--nn_depth', default=3, type=int,
                        help="Number of hidden layers in the time-approximating MLP. Only relevant if '--extrapolate' is set.")
    parser.add_argument('--nn_width', default=300, type=int,
                        help="Width of the time-approximating MLP. Only relevant if '--extrapolate' is set.")
    parser.add_argument('--lookback', default=1, type=int,
                        help="Number of past time steps considered to predict next time. Only relevant if '--extrapolate' is set.")
    parser.add_argument('--metrics_batch_size', default=-1, type=int,
                        help="Perform metrics calculation in batches of given size. Default is no batching / a single batch.")
    parser.add_argument('--ignore_time', dest='ignore_time', action='store_true',
                        help='Ignores time information present in the data and performs standard BoxE.')
    parser.add_argument('--extrapolate', dest='extrapolate', action='store_true',
                        help='Enabled temporal extrapolation by approximating time boxes with an MLP.')
    parser.add_argument('--no_initial_validation', dest='no_initial_validation', action='store_true',
                        help='Disable validation after first epoch.')
    parser.set_defaults(ignore_time=False)
    parser.set_defaults(extrapolate=False)
    parser.set_defaults(no_initial_validation=False)
    return parser.parse_args(args)


def train_test_binary(kg, trainloader, testloader, model, loss_fn, binscore_fn, optimizer, args,
                      combined_loader=None, device='cpu'):
    logging.info('Training started')
    loss_progress = []
    validation_progress = []
    for i_epoch in range(args.num_epochs):
        epoch_losses = []
        for i_batch, data in enumerate(trainloader):
            data = torch.stack(data).to(device).unsqueeze(0)
            optimizer.zero_grad()
            negatives = kg.sample_negatives(data, args.num_negative_samples, args.neg_sampling_type)
            positive_emb, negative_emb = model(data, negatives)
            loss = loss_fn(positive_emb, negative_emb)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        loss_progress.append(np.mean(epoch_losses))
        if args.print_loss_step > 0 and i_epoch % args.print_loss_step == 0:
            logging.info('MEAN EPOCH LOSS: {}'.format(loss_progress[-1]))
        if i_epoch == 0:
            logging.info('first epoch done')
        if i_epoch % args.validation_step == 0:  # validation step
            logging.info('validation checkpoint reached')
            precision, recall = test_retrieval(kg, testloader, model, loss_fn, binscore_fn, optimizer, args, device=device)
            logging.info('PRECISION: {}, RECALL: {}'.format(precision, recall))
            validation_progress.append((precision, recall))
            if combined_loader is not None:
                uf_precision, uf_recall = test_retrieval(kg, testloader, model, loss_fn, binscore_fn, optimizer,
                                                         args, device=device)
                logging.info('UNFILTERED PRECISION: {}, RECALL: {}'.format(uf_precision, uf_recall))

    precision, recall = test_retrieval(kg, testloader, model, loss_fn, binscore_fn, optimizer, args, device=device)
    logging.info('PRECISION: {}, RECALL: {}'.format(precision, recall))
    validation_progress.append((precision, recall))
    if combined_loader is not None:
        uf_precision, uf_recall = test_retrieval(kg, testloader, model, loss_fn, binscore_fn, optimizer, args, device=device)
        logging.info('UNFILTERED PRECISION: {}, RECALL: {}'.format(uf_precision, uf_recall))
    return precision, recall, validation_progress


def train_validate(kg, trainloader, valloader, model, loss_fn, optimizer, args, device='cpu'):
    logging.info('training started')
    best_mrr = -1
    best_params = None
    loss_progress = []
    validation_progress = []
    epoch_times = []
    sampling_times_epoch = []
    for i_epoch in range(args.num_epochs):
        sampling_time = 0
        epoch_start_time = time.time()
        epoch_losses = []
        for i_batch, data in enumerate(trainloader):
            data = torch.stack(data).to(device).unsqueeze(0)
            optimizer.zero_grad()
            sampling_start_time = time.time()
            negatives = kg.sample_negatives(data, args.num_negative_samples, args.neg_sampling_type)
            sampling_end_time = time.time()
            sampling_time += (sampling_end_time-sampling_start_time)
            positive_emb, negative_emb = model(data, negatives)
            loss = loss_fn(positive_emb, negative_emb)
            if not loss.isfinite():
                logging.warning('Loss is {}. Skipping to next mini batch.'.format(loss.item()))
                continue
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        sampling_times_epoch.append(sampling_time)
        loss_progress.append(np.mean(epoch_losses))
        epoch_end_time = time.time()
        epoch_times.append(epoch_end_time-epoch_start_time)
        if args.print_loss_step > 0 and i_epoch % args.print_loss_step == 0:
            logging.info('MEAN EPOCH LOSS: {}'.format(loss_progress[-1]))
        if i_epoch == 0:
            logging.info('first epoch done')
        if i_epoch % args.validation_step == 0 and (i_epoch != 0 or (not args.no_initial_validation)):  # validation step
            logging.info('validation checkpoint reached')
            logging.info('average epoch time: {} seconds'.format(torch.tensor(epoch_times).mean()))
            epoch_times = []
            logging.info('average sampling time per epoch: {} seconds'.format(torch.tensor(sampling_times_epoch).mean()))
            sampling_times_epoch = []
            metrics_start_time = time.time()
            metrics = test(kg, valloader, model, args, device=device, corrupt_triples_batch_size=args.metrics_batch_size)
            metrics_end_time = time.time()
            logging.info('metrics calculation time: {} seconds'.format(metrics_end_time-metrics_start_time))
            logging.info('METRICS: {}'.format(metrics))
            validation_progress.append(metrics)
            if metrics['mrr'] > best_mrr:
                best_mrr = metrics['mrr']
                best_params = copy.deepcopy(model.state_dict())
    logging.info('final validation')
    metrics = test(kg, valloader, model, args, device=device, corrupt_triples_batch_size=args.metrics_batch_size)
    logging.info('METRICS: {}'.format(metrics))
    validation_progress.append(metrics)
    if metrics['mrr'] > best_mrr:
        best_mrr = metrics['mrr']
        best_params = copy.deepcopy(model.state_dict())
    return best_params, best_mrr, {'loss': loss_progress, 'metrics': validation_progress}


def test(kg, dataloader, model, args, device='cpu', corrupt_triples_batch_size=1024):
    with torch.no_grad():
        batch_sizes = []
        mr = []
        mrr = []
        h_at_1 = []
        h_at_3 = []
        h_at_5 = []
        h_at_10 = []
        times = []
        for i_batch, batch in enumerate(dataloader):
            batch = torch.stack(batch).to(device).unsqueeze(0)
            getting_corrupts_start_time = time.time()
            head_corrupts, head_f = kg.corrupt_tuple(batch, 'h', corrupt_triples_batch_size)
            tail_corrupts, tail_f = kg.corrupt_tuple(batch, 't', corrupt_triples_batch_size)
            getting_corrupts_end_time = time.time()
            times.append(getting_corrupts_end_time-getting_corrupts_start_time)
            embeddings = model.forward_positives(batch)
            ranks_head, ranks_tail = 1, 1
            for i, c_batch_head in enumerate(head_corrupts):
                c_batch_tail, head_f_batch, tail_f_batch = tail_corrupts[i], head_f[i], tail_f[i]
                head_c_embs = model.forward_negatives(c_batch_head)
                tail_c_embs = model.forward_negatives(c_batch_tail)
                ranks_head += rank(embeddings, head_c_embs, head_f_batch, args.ignore_time) - 1
                ranks_tail += rank(embeddings, tail_c_embs, tail_f_batch, args.ignore_time) - 1
            batch_sizes.append(batch.shape[2])
            mr.append(mean_rank(embeddings, ranks_head=ranks_head, ranks_tail=ranks_tail))
            mrr.append(mean_rec_rank(embeddings, ranks_head=ranks_head, ranks_tail=ranks_tail))
            h_at_1.append(hits_at_k(embeddings, ranks_head=ranks_head, ranks_tail=ranks_tail, k=1))
            h_at_3.append(hits_at_k(embeddings, ranks_head=ranks_head, ranks_tail=ranks_tail, k=3))
            h_at_5.append(hits_at_k(embeddings, ranks_head=ranks_head, ranks_tail=ranks_tail, k=5))
            h_at_10.append(hits_at_k(embeddings, ranks_head=ranks_head, ranks_tail=ranks_tail, k=10))
        logging.info('Time to get corrupts for validation: {} seconds'.format(torch.tensor(times).sum()))
        batch_sizes = torch.tensor(batch_sizes)
        data_size = torch.sum(batch_sizes)
        mr = (torch.tensor(mr) * batch_sizes).sum() / data_size
        mrr = (torch.tensor(mrr) * batch_sizes).sum() / data_size
        h_at_1 = (torch.tensor(h_at_1) * batch_sizes).sum() / data_size
        h_at_3 = (torch.tensor(h_at_3) * batch_sizes).sum() / data_size
        h_at_5 = (torch.tensor(h_at_5) * batch_sizes).sum() / data_size
        h_at_10 = (torch.tensor(h_at_10) * batch_sizes).sum() / data_size
    return {'mr': mr.item(),
            'mrr': mrr.item(),
            'h@1': h_at_1.item(),
            'h@3': h_at_3.item(),
            'h@5': h_at_5.item(),
            'h@10': h_at_10.item()}


def test_retrieval(kg, testloader, model, loss_fn, binscore_fn, optimizer, options, device='cpu'):
    with torch.no_grad():
        tps, tns, fps, fns = [], [], [], []  # true positives, true negatives, false p's, false n's
        for i_batch, batch in enumerate(testloader):
            batch = torch.stack(batch).to(device).unsqueeze(0)
            head_corrupts, head_f = kg.corrupt_head(batch)
            tail_corrupts, tail_f = kg.corrupt_tail(batch)
            embeddings, head_c_embeddings = model.forward(batch, head_corrupts)
            embeddings, tail_c_embeddings = model.forward(batch, tail_corrupts)
            tp, tn, fp, fn = retrieval_metrics(embeddings, head_c_embeddings, tail_c_embeddings, head_f, tail_f,
                                               binscore_fn)
            logging.log('tp {} tn {} fp {} fn {}'.format(tp, tn, fp, fn))
            tps.append(tp)
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)
    p = precision(np.sum(tps), np.sum(fps))
    r = recall(np.sum(tps), np.sum(fns))
    return p, r


def train_test_val(args, device='cpu', saved_params_dir=None):
    kg = Temp_kg_loader(args.train_path, args.test_path, args.valid_path, truncate=args.truncate_datasets, device=device, entity_subset=args.entity_subset)
    trainloader = kg.get_trainloader(batch_size=args.batch_size, shuffle=True)
    valloader = kg.get_validloader(batch_size=args.batch_size, shuffle=True)
    testloader = kg.get_testloader(batch_size=args.batch_size, shuffle=True)
    if args.extrapolate:
        model = BoxTEmpMLP(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                           args.weight_init, nn_depth=args.nn_depth, nn_width=args.nn_width, lookback=args.lookback,
                           weight_init_args=args.weight_init_args).to(device)
    else:
        model = BoxTEmp(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                        weight_init=args.weight_init, weight_init_args=args.weight_init_args).to(device)
    if args.load_params_path:
        params = torch.load(args.load_params_path, map_location=device)
        model = model.load_state_dict(params)
    optimizer = torch.optim.Adam(model.params(), lr=args.learning_rate)
    loss_fn = BoxELoss(args)

    best_params, best_mrr, progress = train_validate(kg, trainloader, valloader, model, loss_fn, optimizer, args, device=device)
    if best_params is not None:
        model = model.load_state_dict(best_params)
    metrics = test(kg, testloader, model, args, device=device, corrupt_triples_batch_size=args.metrics_batch_size)
    return metrics, progress, copy.deepcopy(model.state_dict())


def run_train_test_binary(args, device='cpu'):
    kg = Temp_kg_loader(args.train_path, args.test_path, args.valid_path, truncate=args.truncate_datasets, device=device)
    trainloader = kg.get_trainloader(batch_size=args.batch_size, shuffle=True)
    testloader = kg.get_combined_loader(datasets=['test', 'valid'], batch_size=args.batch_size, shuffle=True)
    combined_loader = kg.get_combined_loader(datasets=['test', 'valid', 'train'], batch_size=args.batch_size,
                                             shuffle=True)
    model = BoxTEmp(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps()).to(device)
    optimizer = torch.optim.Adam(model.params(), lr=args.learning_rate)
    loss_fn = BoxELoss(args)
    binscore_fn = BoxEBinScore(args)
    return train_test_binary(kg, trainloader, testloader, model, loss_fn, binscore_fn, optimizer, args,
                             combined_loader=combined_loader)


def save_data(args, metrics, model_params, progress, timestamp):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    torch.save(progress, args.log_dir + '/' + timestamp + '-' + args.progress_filename + '.pt')
    torch.save(model_params, args.log_dir + timestamp + '-' + args.params_filename + '.pt')
    with open(args.log_dir + '/' + timestamp + '-' + args.results_filename + '.txt', 'w') as f:
        print(metrics, file=f)
    with open(args.log_dir + '/' + timestamp + '-' + args.info_filename + '.txt', 'w') as f:
        pprint.pprint(vars(args), stream=f)


def run_loop(saved_params_dir=None):
    date_time_now = datetime.now()
    timestamp = '' + str(date_time_now.year) + str(date_time_now.month) + str(date_time_now.day) + str(date_time_now.hour) \
                + str(date_time_now.minute) + str(date_time_now.second)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args(None)
    if args.log_filename:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        complete_filename = args.log_dir + '/' + timestamp + '-' + args.log_filename
        logging.basicConfig(filename=complete_filename, level=logging.INFO)
    else:
        logging.basicConfig(handlers=[logging.StreamHandler()], level=logging.INFO)
    logging.info('Running on {}'.format(device))
    logging.info('%s', args)
    metrics, progress, model_params = train_test_val(args, device=device, saved_params_dir=saved_params_dir)
    logging.info('FINAL TEST METRICS')
    logging.info('%s', metrics)
    save_data(args, metrics, model_params, progress, timestamp)


if __name__ == '__main__':
    print('Execution started')
    start_time = time.time()
    run_loop()
    end_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))