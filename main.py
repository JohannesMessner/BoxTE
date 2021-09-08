import torch
import numpy as np
import copy
import logging
import os
import pprint
import time
from datetime import datetime
import timing
from metrics import mean_rank, mean_rec_rank, hits_at_k, rank
from model import TBoxE, BoxE, BoxTE, DEBoxE
from boxeloss import BoxELoss
from data_utils import TempKgLoader
from argsparser import parse_args


def instantiate_model(args, kg, device):
    uniform_init_args = [(-0.5/np.sqrt(args.embedding_dim)) * args.weight_init_factor,
                         0.5/np.sqrt(args.embedding_dim) * args.weight_init_factor]
    if args.model_variant in ['StaticBoxE', 'static', 'BoxE', 'boxe']:
        model = BoxE(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                     weight_init_args=uniform_init_args, norm_embeddings=args.norm_embeddings,
                     device=device).to(device)
    elif args.model_variant in ['TBoxE', 'tboxe']:
        model = TBoxE(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                      weight_init_args=uniform_init_args,
                      norm_embeddings=args.norm_embeddings, device=device).to(device)
    elif args.model_variant in ['BoxTE', 'boxte']:
        model = BoxTE(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                      weight_init_args=uniform_init_args, time_weight=args.time_weight,
                      norm_embeddings=args.norm_embeddings, use_r_factor=args.use_r_factor,
                      use_e_factor=args.use_e_factor, device=device, nb_timebumps=args.nb_timebumps,
                      use_r_rotation=args.use_r_rotation, use_e_rotation=args.use_e_rotation,
                      nb_time_basis_vecs=args.nb_time_basis_vecs,
                      norm_time_basis_vecs=args.norm_time_basis_vecs, use_r_t_factor=args.use_r_t_factor,
                      dropout_p=args.timebump_dropout_p, arity_spec_timebumps=args.arity_spec_timebumps).to(device)
    elif args.model_variant in ['DEBoxE', 'DE-Boxe', 'deboxe', 'de-boxe']:
        model = DEBoxE(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                       weight_init_args=uniform_init_args,
                       norm_embeddings=args.norm_embeddings, device=device, time_proportion=args.de_time_prop,
                       activation=args.de_activation).to(device)
    else:
        raise ValueError("Invalid model variant {}. Consult --help for valid model variants.".format(args.model_variant))
    return model


def train_validate(kg, trainloader, valloader, model, loss_fn, optimizer, args, timestamp, device='cpu'):
    logging.info('training started')
    best_mrr = -1
    best_params = None
    loss_progress = []
    validation_progress = []
    timer = timing.ExecutionTimer()
    if args.time_execution:
        timer.activate()
    for i_epoch in range(args.num_epochs):
        model.train()
        timer.log('start_epoch')
        epoch_losses = []
        for i_batch, data in enumerate(trainloader):
            data = torch.stack(data).to(device).unsqueeze(0)
            optimizer.zero_grad()
            timer.log('start_neg_sampling')
            negatives = kg.sample_negatives(data, args.num_negative_samples, args.neg_sampling_type, args.neg_sample_what)
            timer.log('end_neg_sampling')
            timer.log('start_forward')
            positive_emb, negative_emb = model(data, negatives)
            timer.log('end_forward')
            if args.use_time_reg:
                loss = loss_fn(positive_emb, negative_emb, time_bumps=model.compute_combined_timebumps(ignore_dropout=True))
            else:
                loss = loss_fn(positive_emb, negative_emb)
            if not loss.isfinite():
                logging.warning('Loss is {}. Skipping to next mini batch.'.format(loss.item()))
                continue
            epoch_losses.append(loss.item())
            timer.log('start_backward')
            loss.backward()
            timer.log('end_backward')
            if args.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            optimizer.step()
        timer.log('end_epoch')
        loss_progress.append(np.mean(epoch_losses))
        if args.print_loss_step > 0 and i_epoch % args.print_loss_step == 0:
            logging.info('MEAN EPOCH LOSS: {}'.format(loss_progress[-1]))
        if i_epoch == 0:
            logging.info('first epoch done')
        if i_epoch % args.validation_step == 0 and (i_epoch != 0 or (not args.no_initial_validation)):  # validation step
            logging.info('validation checkpoint reached')
            timer.log('start_validation')
            metrics = test(kg, valloader, model, args, device=device,
                               corrupt_triples_batch_size=args.metrics_batch_size)
            timer.log('end_validation')
            logging.info('METRICS: {}'.format(metrics))
            validation_progress.append(metrics)
            if metrics['mrr'] > best_mrr:
                best_mrr = metrics['mrr']
                best_params = copy.deepcopy(model.state_dict())
                save_params(args, best_params, timestamp)
    logging.info('final validation')
    timer.log('start_validation')
    metrics = test(kg, valloader, model, args, device=device, corrupt_triples_batch_size=args.metrics_batch_size)
    timer.log('end_validation')
    logging.info('METRICS: {}'.format(metrics))
    validation_progress.append(metrics)
    if metrics['mrr'] > best_mrr:
        best_mrr = metrics['mrr']
        best_params = copy.deepcopy(model.state_dict())
    return best_params, best_mrr, {'loss': loss_progress, 'metrics': validation_progress}


def test(kg, dataloader, model, args, device='cpu', corrupt_triples_batch_size=1024):
    model.eval()
    with torch.no_grad():
        ranks_head, ranks_tail = [], []
        for i_batch, batch in enumerate(dataloader):
            batch = torch.stack(batch).to(device).unsqueeze(0)
            head_corrupts, head_f = kg.corrupt_tuple(batch, 'h', corrupt_triples_batch_size)
            tail_corrupts, tail_f = kg.corrupt_tuple(batch, 't', corrupt_triples_batch_size)
            embeddings = model.forward_positives(batch)
            batch_ranks_head, batch_ranks_tail = 1, 1
            for i, c_batch_head in enumerate(head_corrupts):
                c_batch_tail, head_f_batch, tail_f_batch = tail_corrupts[i], head_f[i], tail_f[i]
                head_c_embs = model.forward_negatives(c_batch_head)
                tail_c_embs = model.forward_negatives(c_batch_tail)
                batch_ranks_head += rank(embeddings, head_c_embs, head_f_batch) - 1
                batch_ranks_tail += rank(embeddings, tail_c_embs, tail_f_batch) - 1
            ranks_head.append(batch_ranks_head)
            ranks_tail.append(batch_ranks_tail)
        ranks_head, ranks_tail = torch.cat(ranks_head), torch.cat(ranks_tail)
        mr = mean_rank(ranks_head=ranks_head, ranks_tail=ranks_tail)
        mrr = mean_rec_rank(ranks_head=ranks_head, ranks_tail=ranks_tail)
        h_at_1 = hits_at_k(ranks_head=ranks_head, ranks_tail=ranks_tail, k=1)
        h_at_3 = hits_at_k(ranks_head=ranks_head, ranks_tail=ranks_tail, k=3)
        h_at_5 = hits_at_k(ranks_head=ranks_head, ranks_tail=ranks_tail, k=5)
        h_at_10 = hits_at_k(ranks_head=ranks_head, ranks_tail=ranks_tail, k=10)
    return {'mr': mr.item(),
            'mrr': mrr.item(),
            'h@1': h_at_1.item(),
            'h@3': h_at_3.item(),
            'h@5': h_at_5.item(),
            'h@10': h_at_10.item()}


def train_test_val(args, timestamp, device='cpu', saved_params_dir=None):
    kg = TempKgLoader(args.train_path, args.test_path, args.valid_path, truncate=args.truncate_datasets, device=device,
                      entity_subset=args.entity_subset, kg_is_static=args.static, data_format=args.data_format)
    trainloader = kg.get_trainloader(batch_size=args.batch_size, shuffle=True)
    valloader = kg.get_validloader(batch_size=args.batch_size, shuffle=True)
    testloader = kg.get_testloader(batch_size=args.batch_size, shuffle=True)
    model = instantiate_model(args, kg, device)
    if args.load_params_path:
        params = torch.load(args.load_params_path, map_location=device)
        model.load_state_dict(params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.use_time_reg and isinstance(model, BoxTE):
        loss_fn = BoxELoss(args, device=device, timebump_shape=model.compute_combined_timebumps().shape)
    else:
        loss_fn = BoxELoss(args, device=device)

    best_params, best_mrr, progress = train_validate(kg, trainloader, valloader, model, loss_fn, optimizer, args,
                                                     timestamp, device=device)
    if best_params is not None:
        model.load_state_dict(best_params)
    metrics = test(kg, testloader, model, args, device=device, corrupt_triples_batch_size=args.metrics_batch_size)
    return metrics, progress, copy.deepcopy(model.state_dict())


def save_data(args, metrics, model_params, progress, timestamp):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    torch.save(progress, args.log_dir + '/' + timestamp + '-' + args.progress_filename + '.pt')
    torch.save(model_params, args.log_dir + timestamp + '-' + args.params_filename + '.pt')
    with open(args.log_dir + '/' + timestamp + '-' + args.results_filename + '.txt', 'w') as f:
        print(metrics, file=f)
    with open(args.log_dir + '/' + timestamp + '-' + args.info_filename + '.txt', 'w') as f:
        pprint.pprint(vars(args), stream=f)


def save_params(args, model_params, timestamp):
    torch.save(model_params, args.log_dir + timestamp + '-' + args.params_filename + '.pt')


def run_loop(saved_params_dir=None):
    date_time_now = datetime.now()
    timestamp = '' + str(date_time_now.year) + str(date_time_now.month) + str(date_time_now.day) + str(date_time_now.hour) \
                + str(date_time_now.minute) + str(date_time_now.second) + str(date_time_now.microsecond)
    print('timestamp: {}'.format(timestamp))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args(None)
    if args.log_filename:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        complete_filename = args.log_dir + '/' + timestamp + '-' + args.log_filename
        print('saving log under filename {}'.format(complete_filename))
        logging.basicConfig(filename=complete_filename, level=logging.INFO)
    else:
        logging.basicConfig(handlers=[logging.StreamHandler()], level=logging.INFO)
    logging.info('Running on {}'.format(device))
    logging.info('%s', args)
    metrics, progress, model_params = train_test_val(args, timestamp, device=device, saved_params_dir=saved_params_dir)
    logging.info('FINAL TEST METRICS')
    logging.info('%s', metrics)
    save_data(args, metrics, model_params, progress, timestamp)


if __name__ == '__main__':
    print('Execution started')
    start_time = time.time()
    run_loop()
    end_time = time.time()