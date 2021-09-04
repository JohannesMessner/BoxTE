import torch
import numpy as np
import copy
import logging
import os
import argparse
import pprint
import time
from datetime import datetime
import timing
from metrics import mean_rank, mean_rec_rank, hits_at_k, rank
from model import TempBoxE_S, TempBoxE_SMLP, TempBoxE_SMLP_Plus, TempBoxE_SLSTM_Plus
from model import TempBoxE_R, TempBoxE_RMLP, TempBoxE_RMLP_Plus, TempBoxE_RMLP_multi, TempBoxE_RLSTM_Plus
from model import DEBoxE_TimeEntEmb, DEBoxE_EntityEmb, DEBoxE_EntityBump, DEBoxE_EntityBase, DEBoxE_TimeBump
from model import TempBoxE, DEBoxE_TwoBumpsPerTime
from model import DEBoxE_BaseM
from model import TempBoxE_M, TempBoxE_MLSTM_Plus
from model import StaticBoxE
from boxeloss import BoxELoss
from data_utils import TempKgLoader


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
    parser.add_argument('--data_format', default='ICEWS', help="Format the dataset comes in:"
                                                               "All facts have time, time given in 'yyy-mm-dd' -> ICEWS;"
                                                               "Some fact have to time, time given in 'yyyy' -> YAGO.")
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
    parser.add_argument('--validation_step', default=500, type=int,
                        help='Number of epochs in between validations.')
    parser.add_argument('--truncate_datasets', default=-1, type=int,
                        help='Truncate datasets to a subset of entries.')
    parser.add_argument('--entity_subset', default=-1, type=int,
                        help='Truncate datasets to a number of entities. Train/test/val split will be maintained.')
    parser.add_argument('--adversarial_temp', default=1, type=float,
                        help='Alpha parameter for adversarial negative sampling loss.')
    parser.add_argument('--loss_type', default='u', type=str,
                        help="Toggle between uniform ('u'), self-adversarial ('a'), and cross entropy ('ce') loss.")
    parser.add_argument('--gradient_clipping', default=-1, type=float,
                        help="Specify a s.t. gradients will be clipped to (-a,a). Default is no clipping.")
    parser.add_argument('--num_negative_samples', default=10, type=int,
                        help="Number of negative samples per positive (true) triple.")
    parser.add_argument('--weight_init_factor', default=1.0, type=float,
                        help="Can make uniform parameter initialization narrower or broader.")
    parser.add_argument('--print_loss_step', default=-1, type=int,
                        help="Number of epochs in between printing of current training loss.")
    parser.add_argument('--time_weight', default=1, type=float,
                        help="Weight assigned to temporal bump in DE-TimeBump model variant.")
    parser.add_argument('--neg_sampling_type', default='a', type=str,
                        help="Toggle between time agnostic ('a') and time dependent ('d') negative sampling.")
    parser.add_argument('--nn_depth', default=3, type=int,
                        help="Number of hidden layers in the time-approximating MLP. Only relevant in mlp model variants.")
    parser.add_argument('--nn_width', default=300, type=int,
                        help="Width of the time-approximating MLP. Only relevant if '--extrapolate' is set.")
    parser.add_argument('--lookback', default=1, type=int,
                        help="Number of past time steps considered to predict next time. Only relevant for mlp model variants.")
    parser.add_argument('--metrics_batch_size', default=-1, type=int,
                        help="Perform metrics calculation in batches of given size. Default is no batching / a single batch.")
    parser.add_argument('--model_variant', default='base', type=str,
                        help="Choose a model variant from [StaticBoxE, TempBoxE_S, TempBoxE_SMLP, TempBoxE_R,"
                             "TempBoxE_RMLP, TempBoxE_RMLP_multi, TempBoxE_SMLP_Plus, TempBoxE_SLSTM_Plus,"
                             "TempBoxE_RMLP_Plus, TempBoxE_M].")
    parser.add_argument('--de_time_prop', default=0.3, type=float,
                        help="Proportion of features considered temporal in the model varinat DEBoxE_B")
    parser.add_argument('--timebump_dropout_p', default=0.0, type=float,
                        help="Probability of any time bump being dropped out. Default is 0.")
    parser.add_argument('--time_reg_weight', default=0.01, type=float,
                       help="Weight given to the temporal regularizer, if enabled.")
    parser.add_argument('--ball_reg_weight', default=0.01, type=float,
                        help="Weight given to the ball regularizer, if enabled.")
    parser.add_argument('--time_reg_order', default=4, type=int,
                        help="Order ('p') of time regularizer norm.")
    parser.add_argument('--ball_reg_order', default=4, type=int,
                        help="Order ('p') of ball regularizer norm.")
    parser.add_argument('--de_activation', default='sine', type=str,
                        help="Activation function used on temporal features in the model variant DEBoxE_B."
                             "Currently 'sine' and 'sigmoid' are supported.")
    parser.add_argument('--nb_timebumps', default=1, type=int,
                        help="Number of bumps each time step has.")
    parser.add_argument('--nb_time_basis_vecs', default=-1, type=int,
                        help="Number of basis vectors used in time bump factorization. Default is to not use"
                             "factorization, but to learn time bumps directly.")
    parser.add_argument('--ce_reduction', default='mean', type=str,
                        help="Reduction applied to the output of cross entropy loss."
                             "'sum' or 'mean'. Default is 'mean'.")
    parser.add_argument('--neg_sample_what', default='e', type=str,
                        help="Decide if to corrupt entities ('e'), time stamps ('t'), or both ('e+t').")
    parser.add_argument('--extrapolate', dest='extrapolate', action='store_true',
                        help='Enabled temporal extrapolation by approximating time boxes with an MLP.')
    parser.add_argument('--no_initial_validation', dest='no_initial_validation', action='store_true',
                        help='Disable validation after first epoch.')
    parser.add_argument('--time_execution', dest='time_execution', action='store_true',
                        help='Roughly time execution of forward, backward, sampling, and validating.')
    parser.add_argument('--norm_embeddings', dest='norm_embeddings', action='store_true',
                        help='Norm all embeddings using tanh function.')
    parser.add_argument('--eval_per_timestep', dest='eval_per_timestep', action='store_true',
                        help='During validation, show metrics for each time step individually.')
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true',
                        help='Performs normalization after every mlp layer, if an mlp based model is chosen.')
    parser.add_argument('--layer_affine', dest='layer_affine', action='store_true',
                        help='Adds affine parameters to norm layer, if layer_norm is set and and mlp based model is chosen.')
    parser.add_argument('--use_r_factor', dest='use_r_factor', action='store_true',
                        help='For 1bpt model, learn one scalar factor per relation that is multiplied with time bump.')
    parser.add_argument('--use_e_factor', dest='use_e_factor', action='store_true',
                        help='For 1bpt model, learn one scalar factor per entity that is multiplied with time bump.')
    parser.add_argument('--use_r_t_factor', dest='use_r_t_factor', action='store_true',
                        help='For 1bpt model, learn one scalar factor per (relation, time)-pair that is multiplied with time bump.')
    parser.add_argument('--use_r_rotation', dest='use_r_rotation', action='store_true',
                        help='For 1bpt model, learn one scalar angle per relation that rotates time bump.')
    parser.add_argument('--use_e_rotation', dest='use_e_rotation', action='store_true',
                        help='For 1bpt model, learn one scalar angle per entity that rotates time bump.')
    parser.add_argument('--use_time_reg', dest='use_time_reg', action='store_true',
                        help='Regularize over time bumps, favouring smoothness')
    parser.add_argument('--use_ball_reg', dest='use_ball_reg', action='store_true',
                        help='Regularize all embedding vectors, concentrating them into a ball around zero.')
    parser.add_argument('--norm_time_basis_vecs', dest='norm_time_basis_vecs', action='store_true',
                        help='Apply softmax function to first term in time bump factorisation, along time axis.')
    parser.add_argument('--arity_spec_timebumps', dest='arity_spec_timebumps', action='store_true',
                        help='Make timebumps arity-specific, meaning that different bumps are learned'
                             'for heads and tails of a fact.')
    parser.set_defaults(ignore_time=False)
    parser.set_defaults(norm_embeddings=False)
    parser.set_defaults(time_execution=False)
    parser.set_defaults(extrapolate=False)
    parser.set_defaults(no_initial_validation=False)
    parser.set_defaults(eval_per_timestep=False)
    parser.set_defaults(layer_norm=False)
    parser.set_defaults(layer_affine=False)
    parser.set_defaults(use_r_factor=False)
    parser.set_defaults(use_r_t_factor=False)
    parser.set_defaults(use_e_factor=False)
    parser.set_defaults(use_r_rotation=False)
    parser.set_defaults(use_e_rotation=False)
    parser.set_defaults(use_time_reg=False)
    parser.set_defaults(use_ball_reg=False)
    parser.set_defaults(norm_time_basis_vecs=False)
    parser.set_defaults(arity_spec_timebumps=False)

    args = parser.parse_args(args)
    if args.model_variant in ['StaticBoxE', 'static']:
        args.static = True
    else:
        args.static = False
    return args


def instantiate_model(args, kg, device):
    uniform_init_args = [(-0.5/np.sqrt(args.embedding_dim)) * args.weight_init_factor,
                         0.5/np.sqrt(args.embedding_dim) * args.weight_init_factor]
    if args.model_variant in ['TempBoxE_SMLP', 'SMLP', 'smlp']:
        model = TempBoxE_SMLP(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                              nn_depth=args.nn_depth, nn_width=args.nn_width, lookback=args.lookback,
                              weight_init_args=uniform_init_args, norm_embeddings=args.norm_embeddings, device=device).to(device)
    elif args.model_variant in ['TempBoxE_RMLP_multi', 'RMLP_multi', 'rmlp_multi']:
        model = TempBoxE_RMLP_multi(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                                    nn_depth=args.nn_depth, nn_width=args.nn_width, lookback=args.lookback,
                                    weight_init_args=uniform_init_args, norm_embeddings=args.norm_embeddings, device=device).to(device)
    elif args.model_variant in ['TempBoxE_RMLP', 'RMLP', 'rmlp']:
        model = TempBoxE_RMLP(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                              nn_depth=args.nn_depth, nn_width=args.nn_width, lookback=args.lookback,
                              weight_init_args=uniform_init_args, norm_embeddings=args.norm_embeddings, device=device).to(device)
    elif args.model_variant in ['TempBoxE_R', 'R', 'r']:
        model = TempBoxE_R(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                              weight_init_args=uniform_init_args, norm_embeddings=args.norm_embeddings,
                              device=device).to(device)
    elif args.model_variant in ['StaticBoxE', 'static']:
        model = StaticBoxE(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                           weight_init_args=uniform_init_args, norm_embeddings=args.norm_embeddings,
                           device=device).to(device)
    elif args.model_variant in ['TempBoxeS', 'S', 's']:
        model = TempBoxE_S(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                           weight_init_args=uniform_init_args, norm_embeddings=args.norm_embeddings, device=device).to(device)
    elif args.model_variant in ['TempBoxE_SMLP_Plus', 'SMLP+', 'smlp+']:
        model = TempBoxE_SMLP_Plus(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                              nn_depth=args.nn_depth, nn_width=args.nn_width, lookback=args.lookback,
                              weight_init_args=uniform_init_args, norm_embeddings=args.norm_embeddings,
                              device=device).to(device)
    elif args.model_variant in ['TempBoxE_SLSTM_Plus', 'SLSTM+', 'slstm+']:
        model = TempBoxE_SLSTM_Plus(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                              nn_depth=args.nn_depth, nn_width=args.nn_width, lookback=args.lookback,
                              weight_init_args=uniform_init_args, norm_embeddings=args.norm_embeddings,
                              device=device).to(device)
    elif args.model_variant in ['TempBoxE_RMLP_Plus', 'RMLP+', 'rmlp+']:
        model = TempBoxE_RMLP_Plus(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                              nn_depth=args.nn_depth, nn_width=args.nn_width, lookback=args.lookback,
                              weight_init_args=uniform_init_args, norm_embeddings=args.norm_embeddings,
                              device=device, layer_norm=args.layer_norm, layer_affine=args.layer_affine).to(device)
    elif args.model_variant in ['TempBoxE_RLSTM_Plus', 'RLSTM+', 'rlstm+']:
        model = TempBoxE_RLSTM_Plus(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                              nn_depth=args.nn_depth, nn_width=args.nn_width, lookback=args.lookback,
                              weight_init_args=uniform_init_args, norm_embeddings=args.norm_embeddings,
                              device=device).to(device)
    elif args.model_variant in ['TempBoxeM', 'M', 'm']:
        model = TempBoxE_M(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                           weight_init_args=uniform_init_args,
                           norm_embeddings=args.norm_embeddings, device=device).to(device)
    elif args.model_variant in ['TempBoxE_MLSTM_Plus', 'MLSTM+', 'mlstm+']:
        model = TempBoxE_MLSTM_Plus(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                              nn_depth=args.nn_depth, nn_width=args.nn_width, lookback=args.lookback,
                              weight_init_args=uniform_init_args, norm_embeddings=args.norm_embeddings,
                              device=device).to(device)
    elif args.model_variant in ['DEBoxE_TimeEntEmb', 'dea']:
        model = DEBoxE_TimeEntEmb(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                                  weight_init_args=uniform_init_args,
                                  norm_embeddings=args.norm_embeddings, device=device).to(device)
    elif args.model_variant in ['DEBoxE_OneBumpPerTime', 'de-onebumppertime', '1bpt']:
        model = TempBoxE(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                         weight_init_args=uniform_init_args, time_weight=args.time_weight,
                         norm_embeddings=args.norm_embeddings, use_r_factor=args.use_r_factor,
                         use_e_factor=args.use_e_factor, device=device, nb_timebumps=args.nb_timebumps,
                         use_r_rotation=args.use_r_rotation, use_e_rotation=args.use_e_rotation,
                         nb_time_basis_vecs=args.nb_time_basis_vecs,
                         norm_time_basis_vecs=args.norm_time_basis_vecs, use_r_t_factor=args.use_r_t_factor,
                         dropout_p=args.timebump_dropout_p, arity_spec_timebumps=args.arity_spec_timebumps).to(device)
    elif args.model_variant in ['DEBoxE_TwoBumpsPerTime', 'de-twobumpspertime', '2bpt']:
        model = DEBoxE_TwoBumpsPerTime(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                                  weight_init_args=uniform_init_args, time_weight=args.time_weight,
                                  norm_embeddings=args.norm_embeddings, device=device).to(device)
    elif args.model_variant in ['DEBoxE_TimeBump', 'de-timebump']:
        model = DEBoxE_TimeBump(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                                  weight_init_args=uniform_init_args,
                                  norm_embeddings=args.norm_embeddings, device=device, activation=args.de_activation,
                                  time_weight=args.time_weight).to(device)
    elif args.model_variant in ['DEBoxE_EntityEmb', 'de-emb', 'deb']:
        model = DEBoxE_EntityEmb(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                                 weight_init_args=uniform_init_args,
                                 norm_embeddings=args.norm_embeddings, device=device, time_proportion=args.de_time_prop,
                                 activation=args.de_activation).to(device)
    elif args.model_variant in ['DEBoxE_EntityBump', 'de-bump', 'dec']:
        model = DEBoxE_EntityBump(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                                  weight_init_args=uniform_init_args,
                                  norm_embeddings=args.norm_embeddings, device=device, time_proportion=args.de_time_prop,
                                  activation=args.de_activation).to(device)
    elif args.model_variant in ['DEBoxE_EntityBase', 'de-base']:
        model = DEBoxE_EntityBase(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                                  weight_init_args=uniform_init_args,
                                  norm_embeddings=args.norm_embeddings, device=device, time_proportion=args.de_time_prop,
                                  activation=args.de_activation).to(device)
    elif args.model_variant in ['TempBoxE_DEM', 'DEM', 'dem']:
        model = DEBoxE_BaseM(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
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
            #print(loss.item())
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
            if args.eval_per_timestep:
                metrics = test_per_timestep(kg, valloader, model, args, device=device,
                                            corrupt_triples_batch_size=args.metrics_batch_size)
            else:
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
    if args.eval_per_timestep:
        metrics = test_per_timestep(kg, valloader, model, args, device=device, corrupt_triples_batch_size=args.metrics_batch_size)
    else:
        metrics = test(kg, valloader, model, args, device=device, corrupt_triples_batch_size=args.metrics_batch_size)
    timer.log('end_validation')
    logging.info('METRICS: {}'.format(metrics))
    validation_progress.append(metrics)
    if metrics['mrr'] > best_mrr:
        best_mrr = metrics['mrr']
        best_params = copy.deepcopy(model.state_dict())
    return best_params, best_mrr, {'loss': loss_progress, 'metrics': validation_progress}


def test_per_timestep(kg, dataloader, model, args, device='cpu', corrupt_triples_batch_size=1024):
    model.eval()
    with torch.no_grad():
        ranks_head, ranks_tail = [], []
        timestamps = []
        for i_batch, batch in enumerate(dataloader):
            batch = torch.stack(batch).to(device).unsqueeze(0)
            timestamps.append(batch[:, 3, :].squeeze())
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
        timestamps = torch.cat(timestamps)
        ranks_head, ranks_tail = torch.cat(ranks_head), torch.cat(ranks_tail)
        result_dict = dict()
        for t in range(model.max_time):
            ranks_head_t, ranks_tail_t = ranks_head[timestamps == t], ranks_tail[timestamps == t]
            d = dict()
            d['mr'] = mean_rank(ranks_head=ranks_head_t, ranks_tail=ranks_tail_t).item()
            d['mrr'] = mean_rec_rank(ranks_head=ranks_head_t, ranks_tail=ranks_tail_t).item()
            d['h_at_1'] = hits_at_k(ranks_head=ranks_head_t, ranks_tail=ranks_tail_t, k=1).item()
            d['h_at_3'] = hits_at_k(ranks_head=ranks_head_t, ranks_tail=ranks_tail_t, k=3).item()
            d['h_at_5'] = hits_at_k(ranks_head=ranks_head_t, ranks_tail=ranks_tail_t, k=5).item()
            d['h_at_10'] = hits_at_k(ranks_head=ranks_head_t, ranks_tail=ranks_tail_t, k=10).item()
            result_dict['time_' + str(t)] = d
        result_dict['mr'] = mean_rank(ranks_head=ranks_head, ranks_tail=ranks_tail).item()
        result_dict['mrr'] = mean_rec_rank(ranks_head=ranks_head, ranks_tail=ranks_tail).item()
        result_dict['h@1'] = hits_at_k(ranks_head=ranks_head, ranks_tail=ranks_tail, k=1).item()
        result_dict['h@3'] = hits_at_k(ranks_head=ranks_head, ranks_tail=ranks_tail, k=3).item()
        result_dict['h@5'] = hits_at_k(ranks_head=ranks_head, ranks_tail=ranks_tail, k=5).item()
        result_dict['h@10'] = hits_at_k(ranks_head=ranks_head, ranks_tail=ranks_tail, k=10).item()
    return result_dict


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
    if args.use_time_reg:
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
    #torch.autograd.set_detect_anomaly(True)
    print('Execution started')
    start_time = time.time()
    run_loop()
    end_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))