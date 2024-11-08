import sys, os
import time
import copy
import json

import torch

UPPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(UPPER_DIR)
from exp_special_utils import get_edge_sample_prob, guided_dropout_edge

sys.path.append(os.path.dirname(UPPER_DIR))
from base_utils.base_general_utils import fix_seed
from base_utils.base_training_utils import train, eval, print_eval_result, get_optimizer


def get_save_dir(args):
    dataset_model_setting = f"{args.dataset.lower()}_{args.model.lower()}"
    dropedge_setting = f"dropedge_[{args.dropedge_rate}]_dropout_[{args.dropout}]_" \
                        f"temperature_[{args.dropedge_temperature}]_" \
                        f"masking_[{'+'.join(args.one_hot_mask)}]_cs_[{str(args.use_correct_and_smooth).lower()}]_" \
                        f"max_[{args.topoinf_max_v}]"

    save_dir = ''   # NOTE: do not add space in ''!
    for sub_dir in [args.perf_save_root_dir, dataset_model_setting, dropedge_setting]:
        save_dir = os.path.join(save_dir, sub_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    return save_dir



def pseudo_label_topoinf_guided_dropedge(data, topoinf_all_e, args):
    if topoinf_all_e is not None:  # pseudo label topoinf guided DropEdge
        ### DropEdge Prob ###
        # t_discount = 1
        t_discount = 1 - args.epoch_index / args.n_epochs / 2   # simulated annealing
        edge_sample_prob = get_edge_sample_prob(topoinf_all_e, data, 
                                    temperature = args.dropedge_temperature * t_discount, 
                                    thr_v = args.topoinf_max_v)
    else:  # general DropEdge
        edge_sample_prob = None
    
    ### Pseudo Label TopoInf-guided DropEdge ###
    edge_index = guided_dropout_edge(data.edge_index, p=args.dropedge_rate, 
                        edge_sample_prob = edge_sample_prob,
                        force_undirected=True, training=True)
    
    return edge_index


def RunExp(data, model, args, criterion, run_index=0, seed=2023, 
            save_file_suffix='wo_dropedge',
            dropedge=False, topoinf_all_e=None,
            return_model=False):

    ## Training Preparation ##
    print('#'*30+f' [Run {run_index+1}/{args.n_runs}] '+'#'*30)

    ### Initial model and optimizer ###
    fix_seed(seed)
    model.reset_parameters()
    optimizer = get_optimizer(model, args)

    eval_result = eval(model, data, criterion)
    print_eval_result(eval_result, prefix='[Initial]')

    ## Start Training ##
    fix_seed(seed)
    if dropedge:
        data_dropouted = copy.deepcopy(data)
    best_val_acc = float('-inf')
    val_acc_history = []

    start_time = time.time()
    for epoch in range(1, 1+args.n_epochs):
        args.epoch_index = epoch    # NOTE: control temperature to decrease, i.e., simulated annealing
        if not dropedge:    # Not DropEdge
            train(model, data, optimizer, criterion)
        else:   # DropEdge
            data_dropouted.edge_index = \
                pseudo_label_topoinf_guided_dropedge(data, topoinf_all_e, args)
            print(f'[DropEdge Info] {data_dropouted.edge_index.size(1)} / {data.edge_index.size(1)}')
            train(model, data_dropouted, optimizer, criterion)

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
    # save_dir = get_save_dir(args)
    save_dir = args.save_dir

    if args.save_detailed_perf:
        best_eval_result_detailed = eval(model, data, criterion=criterion, get_detail=True)
        best_eval_result_detailed['train_time'] = train_time
        save_path_pt = os.path.join(save_dir,
                                    f"run_[{run_index+1}]_total_[{args.n_runs}]_[{save_file_suffix}].pt")
        torch.save(best_eval_result_detailed, save_path_pt)
        print(f"Save PT File: {save_path_pt}")

    if args.save_reduced_perf:
        save_path_json = os.path.join(save_dir, 
                                        f"run_[{run_index+1}]_total_[{args.n_runs}]_[{save_file_suffix}].json")
        with open(save_path_json, 'w') as f:
            json.dump(best_eval_result_reduced, f)
        print(f"Save JSON File: {save_path_json}")

    if return_model:
        return model, best_eval_result_reduced
    else:
        return best_eval_result_reduced
