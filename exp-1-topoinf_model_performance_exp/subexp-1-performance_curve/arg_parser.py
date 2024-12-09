import argparse
import sys


def add_argument_base(parser):
    
    ### Basic Setting ###
    parser.add_argument('--seed', type=int, default=2024, help='seed to generate SEEDS for all runs.')
    parser.add_argument('--dataset', type=str, choices=['cora', 'citeseer', 'pubmed', 
                                                        'computers', 'photo', 
                                                        'actor', 'texas', 'cornell', 'wisconsin', 
                                                        'amazon-ratings', 'roman-empire'],
                        default='cora')
    parser.add_argument('--model-list', nargs='*', type=str, choices=[ 'SGC', 'APPNP', 'MLP', 'GPRGNN', 'BERNNET','GCN'], 
                        default=['SGC', 'APPNP', 'MLP', 'GPRGNN', 'BERNNET','GCN'])
    parser.add_argument('--device', type=int, default=6, help='GPU device (<0 for CPU).')
    parser.add_argument('--n-runs', type=int, default=5, help='number of runs.')
    ### Splitting Setting ###
    parser.add_argument('--split-mode', type=str, choices=['ratio', 'number'], 
                        default='number', help=r'two common splitting mode for CORA are 60%/20%/20% in ratio and 140/500/1k in number.')
    parser.add_argument('--train-rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val-rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--num-train-per-class', type=int, default=20, help='train set number per class.')
    parser.add_argument('--num-val', type=int, default=500, help='val set number.')
    parser.add_argument('--num-test', type=int, default=1000, help='test set number.')
    ### Model Parameters ###
    parser.add_argument('--num-layers', type=int, default=3, help='number of layers for GNN model.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
    parser.add_argument('--K-appnp', type=int, default=3, help='propagation steps.')
    parser.add_argument('--alpha-appnp', type=float, default=0., help='alpha for APPNP/GPRGNN.')
    parser.add_argument('--dprate', type=float,
                        default=0.5, help='dprate for bernnet')
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR', help='Init for GPRGNN')
    parser.add_argument('--alpha', type=float, default=0.1, help='Init for GPRGNN')
    ### Optimizer Parameters ###
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')       
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay.')  
    ### Training Parameters ###
    parser.add_argument('--n-epochs', type=int, default=200, help='number of epochs.')
    parser.add_argument('--eval-interval', type=int, default=1, help='number of epochs.')
    parser.add_argument('--print-interval', type=int, default=50, help='number of epochs.')
    parser.add_argument('--early_stopping', type=int, default=100, help='early stopping epochs.')
    parser.add_argument('--early_stopping_tolerance', type=int, default=1, help='early stopping tolerance in percentage.')
    ### I/O Parameters ###
    parser.add_argument('--not-save', default=False, action='store_true',
                        help='not save model results, which helps to debug.')
    parser.add_argument('--perf-save-root-dir', type=str, default='./topoinf_model_performance/', 
                        help='directory for model performance to save.')
    parser.add_argument('--save-detailed-perf', default=False, action='store_true',
                        help='save detailed result to help further analysis.')
    parser.add_argument('--save-reduced-perf', default=False, action='store_true',
                        help='save reduced result to help further analysis.')
    ### TopoInf Setting ###
    ## Before Computing ##
    parser.add_argument('--k-order', type=int, 
                        default=3, help='TopoInf filter order, which usually is the same with num_layers of GNN model.')
    parser.add_argument('--without-self-loops', default=False, action='store_true',
                        help='remove self-loop in TopoInf computation.')
    parser.add_argument('--distance-metric', type=str, choices=['euclidean_distance', 'inner_product'], 
                        default='inner_product',
                        help='distance metric for measuring distance.')
    parser.add_argument('--coefficients', nargs='*', type=float, 
                        default=[0.0, 0.0, 0.0, 1.0], 
                        help='graph filter coefficients for TopoInf.')
    parser.add_argument('--lambda-reg', type=float, 
                        default=0.1, help='the coefficient for regularization.')
    parser.add_argument('--topoinf_node_masking', type=str, default=['test_mask'],
                        nargs='*', 
                        choices = ['train_mask', 'val_mask', 'test_mask'],
                        help='node masking during computing TopoInf.')
    ## During Computing
    parser.add_argument('--not_verbose', default=False, action='store_true',
                        help='close verbose mode to not monitor progress during computing.')
    parser.add_argument('-sp', '--single-processing', default=False, action='store_true',
                        help='use single processing (default to multiprocessing to accelerate computation).')
    parser.add_argument('--mp-core', type=int, default=8, 
                        help='number of cores used for multiprocessing.')
    ## After Computing
    parser.add_argument('--save_topoinf', default=False, action='store_true',
                        help='save TopoInf results, which helps to debug.')
    parser.add_argument('--delete-unit', type=str, choices=['mode_ratio', 'number', 'ratio'], 
                        default='mode_ratio', help=r'two common deleting unit, i.e., ratio-based deleting and number-based deleting.'
                                                r'mode ratio means choosing ratio of edges in the specified mode, ratio means choosing ratio of edges in all edges.'
                        )
    parser.add_argument('--delete-mode-list', nargs='*', type=str, choices=['pos', 'neg'], 
                        default=['neg', 'pos'], help=r'two common deleting mode, i.e., deleting positive edges and deleting negative edges.')
    parser.add_argument('--delete-strategy', type=str, choices=['all_random', 'topoinf_random', 'topoinf', 'label'], 
                        default='topoinf', help=r'four common deleting strategy, i.e., all random deleting, random deleting, topoinf-based deleting and label-based deleting.'
                                                r' all random deleting means randomly deleting edges in graph, topoinf random deleting means randomly deleting edges in positive/negative edges'
                        )
    parser.add_argument('--delete-rate-list', nargs='*', type=float, 
                        default= [0.0] + [0.1]*13, 
                        help='deleting rate list.')
    parser.add_argument('--delete-num-list', nargs='*', type=int, 
                        default=[100]*6, 
                        help='deleting number list.')
    parser.add_argument('--topoinf-threshold', type=float, 
                        default=1e-2, help='>= thr as positive edges, <= -thr as negative edges.')
    
    parser.add_argument('--delete-step-length', type=int, 
                        default=50, help='the length of delete iteration')
    parser.add_argument('--output',  type=str, 
                        default="output1/", help='sub_exp-1 or sub_exp-2')
    # args = parser.parse_args()                    # NOTE: used when using command line
    # args = parser.parse_args(args=[])             # NOTE: used when using jupyter notebook
    # args = parser.parse_args(args=sys.argv[1:])   # NOTE: used when import as function

    return parser


def init_args(params=sys.argv[1:]):
    ### Parse Args ###
    parser = argparse.ArgumentParser()
    parser = add_argument_base(parser)
    args = parser.parse_args(params)
    
    return args