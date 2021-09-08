import argparse


def parse_args(args):
    # Hyper-Parameters
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--load_params_path', default='',
                        help='Path to saved model parameters that will be loaded. Default initializes a new model.')
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
                        help='Filename given to validation progress data.'
                             'File extension and timestamp are added automatically.')
    parser.add_argument('--params_filename', default='params',
                        help='Filename given to save model parameters / state dict.'
                             'File extension and timestamp are added automatically.')
    parser.add_argument('--results_filename', default='test_results',
                        help='Filename given to final results/metrics.'
                             'File extension and timestamp are added automatically.')
    parser.add_argument('--info_filename', default='info',
                        help='Filename given to information file. File extension and timestamp are added automatically.')
    parser.add_argument('--log_dir', default='',
                        help='Directory results, params, info and log will be saved into.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Loss margin for negative sampling loss.')
    parser.add_argument('--num_epochs', default=10, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training batches.')
    parser.add_argument('--embedding_dim', default=300, type=int,
                        help='Dimensionality of the embedding space.')
    parser.add_argument('--learning_rate', default=.0001, type=float,
                        help='Learning rate for Adam optimiser.')
    parser.add_argument('--validation_step', default=500, type=int,
                        help='Number of epochs in between validations.')
    parser.add_argument('--truncate_datasets', default=-1, type=int,
                        help='For debugging only. Truncate datasets to a subset of entries.')
    parser.add_argument('--entity_subset', default=-1, type=int,
                        help='For debugging only. Truncate datasets to a number of entities.'
                             'Train/test/val split will be maintained.')
    parser.add_argument('--adversarial_temp', default=1, type=float,
                        help='Alpha parameter for adversarial negative sampling loss.')
    parser.add_argument('--loss_type', default='ce', type=str,
                        help="Toggle between uniform ('u'), self-adversarial ('a'), and cross entropy ('ce') loss.")
    parser.add_argument('--gradient_clipping', default=-1, type=float,
                        help="Specify a s.t. gradients will be clipped to (-a,a). Default is no clipping.")
    parser.add_argument('--num_negative_samples', default=75, type=int,
                        help="Number of negative samples per positive (true) triple.")
    parser.add_argument('--weight_init_factor', default=1.0, type=float,
                        help="Can make uniform parameter initialization narrower or broader.")
    parser.add_argument('--print_loss_step', default=-1, type=int,
                        help="Number of epochs in between printing of current training loss.")
    parser.add_argument('--time_weight', default=1, type=float,
                        help="Weight assigned to temporal embeddings.")
    parser.add_argument('--neg_sampling_type', default='a', type=str,
                        help="Toggle between time agnostic ('a') and time dependent ('d') negative sampling.")
    parser.add_argument('--metrics_batch_size', default=-1, type=int,
                        help="Perform metrics calculation in batches of given size. Default is no batching / a single batch.")
    parser.add_argument('--model_variant', default='base', type=str,
                        help="Choose a model variant from [BoxTE, BoxE, DE-BoxE, TBoxE].")
    parser.add_argument('--de_time_prop', default=0.3, type=float,
                        help="Proportion of features considered temporal in the model variant DE-BoxE")
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
                        help="Activation function used on temporal features in the model variant DE-BoxE."
                             "Currently 'sine' and 'sigmoid' are supported.")
    parser.add_argument('--nb_timebumps', default=1, type=int,
                        help="Number of bumps per time step.")
    parser.add_argument('--nb_time_basis_vecs', default=-1, type=int,
                        help="Number of basis vectors used in time bump factorization. Default is to not use"
                             "factorization, but to learn time bumps directly.")
    parser.add_argument('--ce_reduction', default='mean', type=str,
                        help="Reduction applied to the output of cross entropy loss."
                             "'sum' or 'mean'. Default is 'mean'.")
    parser.add_argument('--neg_sample_what', default='e', type=str,
                        help="Decide if to corrupt entities ('e'), time stamps ('t'), or both ('e+t').")
    parser.add_argument('--no_initial_validation', dest='no_initial_validation', action='store_true',
                        help='Disable validation after first epoch.')
    parser.add_argument('--time_execution', dest='time_execution', action='store_true',
                        help='Roughly time execution of forward, backward, sampling, and validating.')
    parser.add_argument('--norm_embeddings', dest='norm_embeddings', action='store_true',
                        help='Norm all embeddings using tanh function.')
    parser.add_argument('--use_r_factor', dest='use_r_factor', action='store_true',
                        help='Learn one scalar factor per relation that is multiplied with time bump.')
    parser.add_argument('--use_e_factor', dest='use_e_factor', action='store_true',
                        help='Learn one scalar factor per entity that is multiplied with time bump.')
    parser.add_argument('--use_r_t_factor', dest='use_r_t_factor', action='store_true',
                        help='Learn one scalar factor per (relation, time)-pair that is multiplied with time bump.')
    parser.add_argument('--use_r_rotation', dest='use_r_rotation', action='store_true',
                        help='Learn one scalar angle per relation that rotates time bump.')
    parser.add_argument('--use_e_rotation', dest='use_e_rotation', action='store_true',
                        help='Learn one scalar angle per entity that rotates time bump.')
    parser.add_argument('--use_time_reg', dest='use_time_reg', action='store_true',
                        help='Regularize over time bumps, favouring smoothness.')
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
    parser.set_defaults(no_initial_validation=False)
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
    if args.model_variant in ['StaticBoxE', 'static', 'BoxE', 'boxe']:
        args.static = True
    else:
        args.static = False
    if (args.use_time_reg or args.use_ball_reg) and not args.model_variant in ['BoxTE', 'boxte']:
        raise ValueError('Regularisers only available with BoxTE model.')
    return args
