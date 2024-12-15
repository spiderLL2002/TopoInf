import time

import torch
import torch.nn.functional as F

from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.utils import one_hot

from torch_sparse import SparseTensor

from .base_training_utils import eval_pred, print_eval_result


def get_pseudo_label_matrix(model, data, args):
    ### Get Init Pseudo Label ###
    with torch.no_grad():
        model.eval()
        logits = model(data)    ## TODO: use MLP trained in GCN?
        if 'pseudo_label_temperature' in args:
            logits = logits / args.pseudo_label_temperature
        y_soft = F.softmax(logits, dim=1).detach()  # NOTE: `y_soft` is the `pseudo_label_matrix`
    
    if len(args.one_hot_mask) > 0:
        one_hot_masking = data[args.one_hot_mask[0]]
        for mask_name in args.one_hot_mask[1:]:
            one_hot_masking = one_hot_masking | data[mask_name]
    
    if args.use_correct_and_smooth:
        # Define C&S model
        adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                sparse_sizes=(data.num_nodes, data.num_nodes)).to(data.edge_index.device).t()
        adj_t = adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj_t
        
        # TODO: add arguments for C&S in `args`
        post = CorrectAndSmooth(num_correction_layers=3, correction_alpha=0.8,
                                num_smoothing_layers=0, smoothing_alpha=0.0,
                                autoscale=False, scale=1.0)
        
        correct_and_smooth_masking = one_hot_masking if len(args.one_hot_mask) > 0 \
                                        else data.train_mask    # NOTE: at least train_mask can be used

        eval_result = eval_pred(y_soft, data, criterion=None)
        print_eval_result(eval_result, prefix='[Before C&S]')

        y_soft = post.correct(y_soft, data.y[correct_and_smooth_masking], correct_and_smooth_masking, DAD)
        eval_result = eval_pred(y_soft, data, criterion=None)
        print_eval_result(eval_result, prefix='[After Correct]')

        y_soft = post.smooth(y_soft, data.y[correct_and_smooth_masking], correct_and_smooth_masking, DA)
        eval_result = eval_pred(y_soft, data, criterion=None)
        print_eval_result(eval_result, prefix='[After Smooth]')

    if len(args.one_hot_mask) > 0:
        y_soft[one_hot_masking] = one_hot(data.y[one_hot_masking], dtype=y_soft.dtype)

    return y_soft


def compute_pseudo_label_topoinf(topoinf_calculator, pseudo_label_matrix, args):
    topoinf_calculator._pre_processing(label_matrix_g = pseudo_label_matrix, 
                                        node_masking = None)

    start_time = time.time()
    topoinf_calculator._pre_processing()
    topoinf_calculator._set_global()
    topoinf_all_e = topoinf_calculator._compute_topoinf_edges_mp(_proc=args.mp_core, verbose=True)
    end_time = time.time()
    print(f"Computation Time for All [{len(topoinf_calculator.G.edges)}] Edges on [{args.dataset.upper()}]: {end_time-start_time:.2f} Seconds.")

    return topoinf_all_e