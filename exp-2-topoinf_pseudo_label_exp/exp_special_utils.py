import sys, os
import time
import copy
import json
from typing import Union

import numpy as np
import networkx as nx

import torch
import torch_geometric
from torch_geometric.utils.convert import to_networkx

UPPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(UPPER_DIR)
sys.path.append(os.path.dirname(UPPER_DIR))
from base_utils.base_general_utils import fix_seed
from base_utils.base_training_utils import train, eval, print_eval_result, get_optimizer

 
def update_edge_index(G_data: Union[torch_geometric.data.Data, nx.graph.Graph], delete_edges: Union[list, tuple]):
    """Cut edges according to delete_edges.
    """
    if isinstance(G_data, nx.graph.Graph):
        G_networkx = G_data.copy()
        _device = 'cpu'
    elif isinstance(G_data, torch_geometric.data.Data):
        G_networkx = to_networkx(G_data, node_attrs=None, to_undirected=True, remove_self_loops=True)
        _device = G_data.edge_index.device
    else:
        raise NotImplementedError

    G_networkx.remove_edges_from(delete_edges)
    # updated edge_index
    edge_index = torch.tensor(np.array(G_networkx.to_directed().edges).T).to(_device)
    # NOTE: remember to turn Graph into directed and move edge_index to _device.

    return G_networkx, edge_index


def topoinf_based_deleting_edges(data, topoinf_all_e: dict, unit_value, args):
    ### Get Deleting Edges ###
    topoinf_all_e_sorted = sorted(topoinf_all_e.items(), key=lambda item: item[1], reverse=True)
    topoinf_all_e_tensor = torch.tensor(list(topoinf_all_e.values()))
    num_pos_edges = (topoinf_all_e_tensor > args.topoinf_threshold).sum().item()
    num_neg_edges = (topoinf_all_e_tensor < -args.topoinf_threshold).sum().item()
    print(f"[Info] [{len(topoinf_all_e_sorted)}] computed edges | [{num_pos_edges}] pos edges | [{num_neg_edges}] neg edges.")

    if args.delete_unit == 'mode_ratio':
        if args.delete_mode == 'pos':
            delete_num = int(num_pos_edges*unit_value)
        elif args.delete_mode == 'neg':
            delete_num = int(num_neg_edges*unit_value)
    elif args.delete_unit == 'number':
        delete_num = unit_value
    elif args.delete_unit == 'ratio':
        num_edges = len(topoinf_all_e_sorted)
        delete_num = int(num_edges*unit_value)

    if args.delete_mode == 'pos':
        if delete_num > num_pos_edges:
            print(f"[Warning] num of del edges [{delete_num}] > num of pos edges [{num_pos_edges}]")
            delete_num = num_pos_edges  # NOTE: delete_num = max(delete_num, num_pos_edges)
        delete_edges = [edge for edge, _ in topoinf_all_e_sorted[:delete_num]]
        print(f"[Info] delete [{delete_num}] edges out of [{num_pos_edges}] pos edges.")
    elif args.delete_mode == 'neg':
        if delete_num > num_neg_edges:
            print(f"[Warning] num of del edges [{delete_num}] > num of neg edges [{num_neg_edges}]")
            delete_num = num_neg_edges  # NOTE: delete_num = max(delete_num, num_neg_edges)
        delete_edges = [edge for edge, _ in topoinf_all_e_sorted[-delete_num:]]
        print(f"[Info] delete [{delete_num}] edges out of [{num_neg_edges}] neg edges.")
    
    print(f"Deleted {delete_num} {args.delete_mode.capitalize()} Edges.")

    ### Delete Edges ###
    _, edge_index = update_edge_index(data, delete_edges)
    
    return edge_index


def RunExp(data, model, args, criterion, run_index=0, seed=2023, 
            save_file_suffix="before_topoinf",
            return_model=False):

    print('#'*30+f' [Run {run_index+1}/{args.n_runs}] '+'#'*30)

    ### Initial model and optimizer ###
    fix_seed(seed)
    model.reset_parameters()
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = get_optimizer(model, args)

    eval_result = eval(model, data, criterion=None, get_detail=False)
    print_eval_result(eval_result, prefix='[Initial]')

    ## Start Training ##
    fix_seed(seed)
    best_val_acc = float('-inf')
    val_acc_history = []

    start_time = time.time()
    for epoch in range(1, 1+args.n_epochs):
        train(model, data, optimizer, criterion)
        
        if epoch % args.eval_interval == 0:
            eval_result = eval(model, data, criterion=None, get_detail=False)

            if epoch % (args.print_interval * args.eval_interval) == 0:
                print_eval_result(eval_result, prefix=f'[Epoch {epoch:3d}/{args.n_epochs:3d}]')
            
            if eval_result['val_mask']['acc'] > best_val_acc:
                best_val_acc = eval_result['val_mask']['acc']
                best_model_param = copy.deepcopy(model.state_dict())  # NOTE: `best_model_param` may be more efficient.

            val_acc_history.append(eval_result['val_mask']['acc'])
            if args.early_stopping > 0 and len(val_acc_history) > args.early_stopping:
                mean_val_acc = torch.tensor(
                    val_acc_history[-(args.early_stopping + 1):-1]).mean().item()
                if (eval_result['val_mask']['acc'] - mean_val_acc) * 100 < - args.early_stopping_tolerance: # NOTE: in percentage
                    print('[Early Stop Info] Stop at Epoch: ', epoch)
                    break
    
    train_time = time.time() - start_time

    ## Eval Best Result ##
    model.load_state_dict(best_model_param)
    best_eval_result_reduced = eval(model, data, criterion=None, get_detail=False)
    best_eval_result_reduced['train_time'] = train_time
    print_eval_result(best_eval_result_reduced, prefix=f'[Final Result] Time: {train_time:.2f}s |')

    ## Save Result ##
    if not args.not_save:
        # save_dir = get_save_dir(args)
        save_dir = args.save_dir

    if args.save_detailed_perf:
        best_eval_result_detailed = eval(model, data, criterion=criterion, get_detail=True)
        best_eval_result_detailed['train_time'] = train_time
        save_path_pt = os.path.join(save_dir,
                                    f"run_[{run_index+1}]_total_[{args.n_runs}]_{save_file_suffix}.pt")
        torch.save(best_eval_result_detailed, save_path_pt)
        print(f"Save PT File: {save_path_pt}")

    if args.save_reduced_perf:
        save_path_json = os.path.join(save_dir, 
                                        f"run_[{run_index+1}]_total_[{args.n_runs}]_{save_file_suffix}.json")
        with open(save_path_json, 'w') as f:
            json.dump(best_eval_result_reduced, f)
        print(f"Save JSON File: {save_path_json}")

    if return_model:
        return model, best_eval_result_reduced
    else:
        return best_eval_result_reduced