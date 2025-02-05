import sys, os
import time
import copy
import json
from typing import Union
from collections import defaultdict
import numpy as np
import networkx as nx
import math

import torch
import torch_geometric
from torch_geometric.utils.convert import to_networkx

UPPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(UPPER_DIR)
sys.path.append(os.path.dirname(UPPER_DIR))
from base_utils.base_general_utils import fix_seed
from base_utils.base_training_utils import train, eval, print_eval_result, get_optimizer
from topoinf_impl import TopoInf

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
 
 
def update_edge_index(G_data: Union[torch_geometric.data.Data, nx.graph.Graph], delete_edges: Union[list, tuple]):
    """Cut edges according to delete_edges.
    """
    if isinstance(G_data, nx.graph.Graph):
        _device = 'cpu'
        existing_edges = set(G_data.edges)
        for edge in delete_edges:
            if edge in existing_edges:
                G_data.remove_edge(*edge)
            else:
                G_data.add_edge(*edge)
        edge_index = torch.tensor(np.array(G_data.to_directed().edges).T).to(_device)
        # NOTE: remember to turn Graph into directed and move edge_index to _device.
        return edge_index
    elif isinstance(G_data, torch_geometric.data.Data):
        edge_index = G_data.edge_index  
        
        delete_edges_set = set(delete_edges)
        edges = edge_index.t().tolist()  
        edge_set = set(tuple(edge) for edge in edges)
        
        edge_set ^= delete_edges_set
        
        edge_index = torch.tensor(list(edge_set)).t()
        
        return edge_index
    else:
        raise NotImplementedError

def get_edges_nums(topoinf_all_e_dict)  : 
    topoinf_all_e_tensor = torch.tensor(list(topoinf_all_e_dict.values()))
    num_pos_edges = (topoinf_all_e_tensor >= 0).sum().item()
    num_neg_edges = (topoinf_all_e_tensor < 0).sum().item() 
    return num_pos_edges ,num_neg_edges 

def get_delete_edges_wrapper(edges_haven_deleted ,topoinf_all_e, args):
    ### Get Deleting Edges ###
    topoinf_all_e_sorted = sorted(topoinf_all_e.items(), key=lambda item: item[1], reverse=True)
    delete_num = args.delete_num + args.delete_num
    if args.delete_mode == 'pos':
        delete_edges = [edge for edge, _ in topoinf_all_e_sorted[:delete_num]]
        
    elif args.delete_mode == 'neg':
        delete_edges = [edge for edge, _ in topoinf_all_e_sorted[-delete_num:]]
        
    #print(f"Deleted {delete_num} {args.delete_mode.capitalize()} Edges.")

    '''delete_info = {
        'delete_num': delete_num,
        'ratio_in_total': delete_num / len(topoinf_all_e_tensor),
    }'''
    
    for e in delete_edges  :
        
        del topoinf_all_e[e]
        edges_haven_deleted.add(e)
        edges_haven_deleted.add((e[1],e[0]))
    return delete_edges

def update_topoinf(edges_haven_deleted ,data,topoinf_all_e,delete_edges,args ,coefficients = None,entropy_dict: dict = {},
                    entropy_coefficient :float = 0.02,
                    degree_delete_edge_dict :dict = {})  :
    topoinf_calculator = TopoInf(data = data, 
            lambda_reg = args.lambda_reg,
            with_self_loops = not args.without_self_loops,
            k_order = args.k_order,
            coefficients = coefficients,
            distance_metric_name = args.distance_metric,
            edges_haven_deleted = edges_haven_deleted,
            entropy_dict = entropy_dict,
            entropy_coefficient = entropy_coefficient,
            degree_delete_edge_dict = degree_delete_edge_dict
            )
    now_topoinf_all_e = copy.deepcopy(topoinf_all_e)
    node_masking = None
    topoinf_calculator._pre_processing(node_masking = node_masking)
    return topoinf_calculator.update_topoinf_edges(delete_edges,verbose = False , topoinf_all = now_topoinf_all_e)

def topoinf_based_deleting_edges(degree_delete_edge_dict,edges_haven_deleted,data,topoinf_all_e , args,coefficients = None,entropy_dict = {},entropy_coefficient  = 0.02):
    now_topoinf_all_e = copy.deepcopy(topoinf_all_e) 
    delete_edges = get_delete_edges_wrapper(edges_haven_deleted,now_topoinf_all_e, args)
    #print(delete_edges ,len(delete_edges))
    data.edge_index =  update_edge_index(data, delete_edges)
    for e in delete_edges : 
        (i,j) = e 
        degree_delete_edge_dict[i] += 1 
        degree_delete_edge_dict[j] += 1 
    now_topoinf_all_e = update_topoinf(edges_haven_deleted ,\
        data,now_topoinf_all_e,delete_edges,args,coefficients ,\
        entropy_dict = entropy_dict,entropy_coefficient  = entropy_coefficient,\
        degree_delete_edge_dict =degree_delete_edge_dict)
    return now_topoinf_all_e ,data