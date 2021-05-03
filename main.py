import torch
import numpy as np
import copy
import sys
import os
import argparse
import pprint
from datetime import datetime
from metrics import mean_rank
from metrics import mean_rec_rank
from metrics import hits_at_k
from metrics import retrieval_metrics
from metrics import precision
from metrics import recall
from model import BoxTEmp
from model import BoxELoss
from model import BoxEBinScore
from data_utils import Temp_kg_loader


def parse_args(args):
    # Hyper-Parameters
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--train_path', default='./train.txt',
                        help='Path to training dataset')
    parser.add_argument('--valid_path', default='./valid.txt',
                        help='Path to validation dataset')
    parser.add_argument('--test_path', default='./test.txt',
                        help='Path to test dataset')
    parser.add_argument('--progress_filename', default='progress',
                        help='Filename given to validation progress data. File extension and timestamp are added automatically.')
    parser.add_argument('--params_filename', default='params',
                        help='Filename given to save model parameters / state dict.  File extension and timestamp are added automatically.')
    parser.add_argument('--results_filename', default='test_results',
                        help='Filename given to final results/metrics. File extension and timestamp are added automatically.')
    parser.add_argument('--info_filename', default='info',
                        help='Filename given to information file. File extension and timestamp are added automatically.')
    parser.add_argument('--results_dir', default='',
                        help='Directory results, params and info will be saved into.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Loss margin.')
    parser.add_argument('--num_epochs', default=10, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training batch size.')
    parser.add_argument('--embedding_dim', default=300, type=int,
                        help='Dimensionality of the embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
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
    parser.add_argument('--adversarial_temp', default=1, type=float,
                        help='Alpha parameter for adversarial negative sampling loss.')
    parser.add_argument('--loss_k', default=1, type=float,
                        help='k parameter for uniform loss.')
    parser.add_argument('--loss_type', default='u', type=str,
                        help="Toggle between uniform ('u') and self-adversarial ('a') loss.")
    parser.add_argument('--num_negative_samples', default=10, type=int,
                        help="Number of negative samples per positive (true) triple.")
    parser.add_argument('--weight_init', default='u', type=str,
                        help="Type of weight initialization for the model.")
    parser.add_argument('--print_loss_step', default=-1, type=int,
                        help="Number of epochs in between printing of current training loss.")
    parser.add_argument('--neg_sampling_type', default='a', type=str,
                        help="Toggle between time agnostic ('a') and time dependent ('d') negative sampling.")
    parser.add_argument('--ignore_time', dest='ignore_time', action='store_true')
    parser.set_defaults(ignore_time=False)
    return parser.parse_args(args)


def train_test_binary(kg, trainloader, testloader, model, loss_fn, binscore_fn, optimizer, options,
                      combined_loader=None, device='cpu'):
    print('training started')
    loss_progress = []
    validation_progress = []
    for i_epoch in range(options.num_epochs):
        epoch_losses = []
        for i_batch, data in enumerate(trainloader):
            data = torch.stack(data).to(device).unsqueeze(0)
            optimizer.zero_grad()
            negatives = kg.sample_negatives(data, options.num_negative_samples, options.neg_sampling_type)
            positive_emb, negative_emb = model(data, negatives)
            loss = loss_fn(positive_emb, negative_emb)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        loss_progress.append(np.mean(epoch_losses))
        if options.print_loss_step > 0 and i_epoch % options.print_loss_step == 0:
            print('MEAN EPOCH LOSS: {}'.format(loss_progress[-1]))
        if i_epoch == 0:
            print('first epoch done')
        if i_epoch % options.validation_step == 0:  # validation step
            print('validation checkpoint reached')
            precision, recall = test_retrieval(kg, testloader, model, loss_fn, binscore_fn, optimizer, options, device=device)
            print('PRECISION: {}, RECALL: {}'.format(precision, recall))
            validation_progress.append((precision, recall))
            if combined_loader is not None:
                uf_precision, uf_recall = test_retrieval(kg, testloader, model, loss_fn, binscore_fn, optimizer,
                                                         options, device=device)
                print('UNFILTERED PRECISION: {}, RECALL: {}'.format(uf_precision, uf_recall))

    precision, recall = test_retrieval(kg, testloader, model, loss_fn, binscore_fn, optimizer, options, device=device)
    print('PRECISION: {}, RECALL: {}'.format(precision, recall))
    validation_progress.append((precision, recall))
    if combined_loader is not None:
        uf_precision, uf_recall = test_retrieval(kg, testloader, model, loss_fn, binscore_fn, optimizer, options, device=device)
        print('UNFILTERED PRECISION: {}, RECALL: {}'.format(uf_precision, uf_recall))
    return precision, recall, validation_progress


def train_validate(kg, trainloader, valloader, model, loss_fn, optimizer, options, device='cpu'):
    print('training started')
    best_mrr = -1
    best_params = None
    loss_progress = []
    validation_progress = []
    for i_epoch in range(options.num_epochs):
        epoch_losses = []
        for i_batch, data in enumerate(trainloader):
            data = torch.stack(data).to(device).unsqueeze(0)
            optimizer.zero_grad()
            negatives = kg.sample_negatives(data, options.num_negative_samples, options.neg_sampling_type)
            positive_emb, negative_emb = model(data, negatives)
            loss = loss_fn(positive_emb, negative_emb)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        loss_progress.append(np.mean(epoch_losses))
        if options.print_loss_step > 0 and i_epoch % options.print_loss_step == 0:
            print('MEAN EPOCH LOSS: {}'.format(loss_progress[-1]))
        if i_epoch == 0:
            print('first epoch done')
        if i_epoch % options.validation_step == 0:  # validation step
            print('validation checkpoint reached')
            metrics = test(kg, valloader, model, loss_fn, optimizer, options, device=device)
            print('METRICS: {}'.format(metrics))
            validation_progress.append(metrics)
            if metrics['mrr'] > best_mrr:
                best_mrr = metrics['mrr']
                best_params = copy.deepcopy(model.state_dict())
    print('final validation')
    metrics = test(kg, valloader, model, loss_fn, optimizer, options, device=device)
    print('METRICS: {}'.format(metrics))
    validation_progress.append(metrics)
    if metrics['mrr'] > best_mrr:
        best_mrr = metrics['mrr']
        best_params = copy.deepcopy(model.state_dict())
    return best_params, best_mrr, {'loss': loss_progress, 'metrics': validation_progress}


def test(kg, dataloader, model, loss_fn, optimizer, options, device='cpu'):
    with torch.no_grad():
        batch_sizes = []
        mr = []
        mrr = []
        h_at_1 = []
        h_at_3 = []
        # h_at_5 = []
        h_at_10 = []
        for i_batch, batch in enumerate(dataloader):
            batch = torch.stack(batch).to(device).unsqueeze(0)
            head_corrupts, head_f = kg.corrupt_head(batch.squeeze())
            tail_corrupts, tail_f = kg.corrupt_tail(batch.squeeze())
            embeddings, head_c_embeddings = model.forward(batch, head_corrupts)
            tail_c_embeddings = model.forward_negatives(tail_corrupts)
            batch_sizes.append(len(batch[0]))
            mr.append(mean_rank(embeddings, head_c_embeddings, tail_c_embeddings, head_f, tail_f))
            mrr.append(mean_rec_rank(embeddings, head_c_embeddings, tail_c_embeddings, head_f, tail_f))
            h_at_1.append(hits_at_k(embeddings, head_c_embeddings, tail_c_embeddings, head_f, tail_f, k=1))
            h_at_3.append(hits_at_k(embeddings, head_c_embeddings, tail_c_embeddings, head_f, tail_f, k=3))
            # h_at_5.append(hits_at_k(embeddings, head_c_embeddings, tail_c_embeddings, head_f, tail_f, k=5))
            h_at_10.append(hits_at_k(embeddings, head_c_embeddings, tail_c_embeddings, head_f, tail_f, k=10))
        batch_sizes = torch.tensor(batch_sizes)
        data_size = torch.sum(batch_sizes)
        mr = (torch.tensor(mr) * batch_sizes).sum() / data_size
        mrr = (torch.tensor(mrr) * batch_sizes).sum() / data_size
        h_at_1 = (torch.tensor(h_at_1) * batch_sizes).sum() / data_size
        h_at_3 = (torch.tensor(h_at_3) * batch_sizes).sum() / data_size
        # h_at_5 = (torch.tensor(h_at_5) * batch_sizes).sum() / data_size
        h_at_10 = (torch.tensor(h_at_10) * batch_sizes).sum() / data_size
    return {'mr': mr.item(),
            'mrr': mrr.item(),
            'h@1': h_at_1.item(),
            'h@3': h_at_3.item(),
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
            print('tp {} tn {} fp {} fn {}'.format(tp, tn, fp, fn))
            tps.append(tp)
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)
    p = precision(np.sum(tps), np.sum(fps))
    r = recall(np.sum(tps), np.sum(fns))
    return p, r


def train_test_val(options, device='cpu', saved_params_dir=None):
    kg = Temp_kg_loader(options.train_path, options.test_path, options.valid_path, truncate=options.truncate_datasets, device=device)
    trainloader = kg.get_trainloader(batch_size=options.batch_size, shuffle=True)
    valloader = kg.get_validloader(batch_size=options.batch_size, shuffle=True)
    testloader = kg.get_testloader(batch_size=options.batch_size, shuffle=True)
    model = BoxTEmp(options.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps()).to(device)
    if saved_params_dir is not None:
        model.load_state_dict(torch.load(saved_params_dir))
    optimizer = torch.optim.Adam(model.params(), lr=options.learning_rate)
    loss_fn = BoxELoss(options)

    best_params, best_mrr, progress = train_validate(kg, trainloader, valloader, model, loss_fn, optimizer, options, device=device)
    if best_params is not None:
        model = model.load_state_dict(best_params)
    optimizer = torch.optim.Adam(model.params(), lr=options.learning_rate)  # this isn't really needed is it?
    metrics = test(kg, testloader, model, loss_fn, optimizer, options, device=device)
    return metrics, progress, copy.deepcopy(model.state_dict())


def run_train_test_binary(options, device='cpu'):
    kg = Temp_kg_loader(options.train_path, options.test_path, options.valid_path, truncate=options.truncate_datasets, device=device)
    trainloader = kg.get_trainloader(batch_size=options.batch_size, shuffle=True)
    testloader = kg.get_combined_loader(datasets=['test', 'valid'], batch_size=options.batch_size, shuffle=True)
    combined_loader = kg.get_combined_loader(datasets=['test', 'valid', 'train'], batch_size=options.batch_size,
                                             shuffle=True)
    model = BoxTEmp(options.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps()).to(device)
    optimizer = torch.optim.Adam(model.params(), lr=options.learning_rate)
    loss_fn = BoxELoss(options)
    binscore_fn = BoxEBinScore(options)
    return train_test_binary(kg, trainloader, testloader, model, loss_fn, binscore_fn, optimizer, options,
                             combined_loader=combined_loader)


def save_data(options, metrics, model_params, progress):
    date_time_now = datetime.now()
    timestamp = '' + str(date_time_now.year) + str(date_time_now.month) + str(date_time_now.day) + str(date_time_now.hour) \
                + str(date_time_now.minute) + str(date_time_now.second)
    if not os.path.exists(options.results_dir):
        os.makedirs(options.results_dir)
    torch.save(progress, options.results_dir + timestamp + '-' + options.progress_filename + '.pt')
    torch.save(model_params, options.results_dir + timestamp + '-' + options.params_filename + '.pt')
    with open(options.results_dir + timestamp + '-' + options.results_filename + '.txt', 'w') as f:
        print(metrics, file=f)
    with open(options.results_dir + timestamp + '-' + options.info_filename + '.txt', 'w') as f:
        pprint.pprint(vars(options), stream=f)


def run_loop(saved_params_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on {}'.format(device))
    options = parse_args(None)
    metrics, progress, model_params = train_test_val(options, device=device, saved_params_dir=saved_params_dir)
    print('FINAL TEST METRICS')
    print(metrics)
    save_data(options, metrics, model_params, progress)


if __name__ == '__main__':
    print('Execution started')
    run_loop()