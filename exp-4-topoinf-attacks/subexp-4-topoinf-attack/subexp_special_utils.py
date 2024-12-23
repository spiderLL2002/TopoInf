import os
import json
from base_utils.base_io_utils import analyse_one_setting

import sys
UPPER_DIR = os.path.dirname(os.path.abspath(os.getcwd()))    # NOTE: for .ipynb
sys.path.append(UPPER_DIR)

  
def get_save_dir(args):
    dataset_model_setting = f"{args.dataset.lower()}_{args.model.lower()}"
    topoinf_delete_setting = \
        f"delete_[{args.delete_mode}]_by_[{args.delete_unit}]_" \
        f"masking_[{'+'.join(args.one_hot_mask)}]_cs_[{str(args.use_correct_and_smooth).lower()}]_"\
        f"order_[{args.k_order}]_thr_[{args.topoinf_threshold:.1e}]"

    save_dir = ''   # NOTE: do not add space in ''!
    for sub_dir in [args.perf_save_root_dir, dataset_model_setting, topoinf_delete_setting]:
        save_dir = os.path.join(save_dir, sub_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    
    return save_dir


def analyse_and_save_recording( recording: dict = {}, args = None):
    """
    recording[f'run_[{run_index+1}]']
        -> ['before_topoinf']
        -> [f'delete_[{args.delete_unit}]_[{unit_value}]']
            -> ['before_retrain']
            -> ['after_retrain']
    ----- Analysis -----
    -> recording[f'delete_[{args.delete_unit}]_[{unit_value}]']
        -> ['seed_list']
        -> ['before_topoinf']
        -> ['before_retrain']
        -> ['after_retrain']
    """
    analysed_recording = {}
    analysis_attr_list = ['before_topoinf', 'before_retrain', 'after_retrain']
    
    unit_value_list =  args.delete_num_list if args.delete_unit in ['number'] else args.delete_rate_list
    
    for i,unit_value in enumerate(unit_value_list):
        unit_key = f'delete_[{args.delete_unit}]_[{unit_value*(i+1)}]'
        analysed_recording[unit_key] = {}
        for analysis_attr in analysis_attr_list:
            analysed_recording[unit_key][analysis_attr] = {}
            analysed_recording[unit_key][analysis_attr]['test_acc_list'] = []
        
        for run_index in range(args.n_runs):
            run_key = f'run_[{run_index+1}]'
        
            analysed_recording[unit_key]['before_topoinf']['test_acc_list'].append(recording[run_key]['before_topoinf']['test_mask']['acc'])
            analysed_recording[unit_key]['before_retrain']['test_acc_list'].append(recording[run_key][unit_key]['before_retrain']['test_mask']['acc'])
            analysed_recording[unit_key]['after_retrain']['test_acc_list'].append(recording[run_key][unit_key]['after_retrain']['test_mask']['acc'])

        for analysis_attr in analysis_attr_list:
            test_acc_list = analysed_recording[unit_key][analysis_attr]['test_acc_list']
            test_acc_mean, test_acc_uncertainty, test_acc_std = \
                analyse_one_setting(test_acc_list, 
                                    setting_name = f'{unit_key}_[{analysis_attr}]')
            analysed_recording[unit_key][analysis_attr]['test_acc_mean'] = test_acc_mean
            analysed_recording[unit_key][analysis_attr]['test_acc_uncertainty'] = test_acc_uncertainty
            analysed_recording[unit_key][analysis_attr]['test_acc_std'] = test_acc_std

    # save final results
    if not args.not_save:
        save_dir = get_save_dir(args)

        save_path_json = os.path.join(save_dir, "original_recording.json")
        with open(save_path_json, 'w') as f:
            json.dump(recording, f)
        print(f"Save Original Recording in JSON File: [{save_path_json}]")

        save_path_json = os.path.join(save_dir, "analysed_recording.json")
        with open(save_path_json, 'w') as f:
            json.dump(analysed_recording, f)
        print(f"Save Analysed Recording in JSON File: [{save_path_json}]")