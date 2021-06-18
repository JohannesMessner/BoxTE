import torch
from datetime import datetime
import logging
import os
from main import test, parse_args
from data_utils import TempKgLoader
from model import TempBoxE_S, TempBoxE_SMLP
import pprint
import time


def save_results(args, metrics, timestamp):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(args.log_dir + '/' + timestamp + '-' + args.results_filename + '.txt', 'w') as f:
        print(metrics, file=f)
    with open(args.log_dir + '/' + timestamp + '-' + args.info_filename + '.txt', 'w') as f:
        pprint.pprint(vars(args), stream=f)


def test_model():
    date_time_now = datetime.now()
    timestamp = '' + str(date_time_now.year) + str(date_time_now.month) + str(date_time_now.day) + str(
        date_time_now.hour) \
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
    kg = TempKgLoader(args.train_path, args.test_path, args.valid_path, truncate=args.truncate_datasets,
                      device=device, entity_subset=args.entity_subset)
    loader = kg.get_testloader(batch_size=args.batch_size, shuffle=True)
    if args.extrapolate:
        model = TempBoxE_SMLP(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                              args.weight_init, nn_depth=args.nn_depth, nn_width=args.nn_width, lookback=args.lookback,
                              weight_init_args=args.weight_init_args).to(device)
    else:
        model = TempBoxE_S(args.embedding_dim, kg.relation_ids, kg.entity_ids, kg.get_timestamps(),
                           weight_init=args.weight_init, weight_init_args=args.weight_init_args).to(device)
    params = torch.load(args.load_params_path, map_location=device)
    model = model.load_state_dict(params)

    metrics = test(kg, loader, model, args, device, args.metrics_batch_size)
    logging.info('FINAL TEST METRICS')
    logging.info('%s', metrics)
    save_results(args, metrics, timestamp)


if __name__ == '__main__':
    print('Execution started')
    start_time = time.time()
    test_model()
    end_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))