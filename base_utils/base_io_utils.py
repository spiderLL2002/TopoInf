import os
import json
import numpy as np
import seaborn as sns


def make_multi_level_dir(dir_list):
    multi_level_dir = ''   # NOTE: do not add space in ''!
    for sub_dir in dir_list:
        multi_level_dir = os.path.join(multi_level_dir, sub_dir)
        if not os.path.exists(multi_level_dir):
            os.mkdir(multi_level_dir)
    return multi_level_dir


def compute_uncertainty(values):
    return np.max(np.abs(sns.utils.ci(sns.algorithms.bootstrap(values,func=np.mean,n_boot=1000),95) - values.mean()))


def analyse_one_setting(test_acc_list, setting_name):
    test_acc_arr = np.array(test_acc_list) * 100

    test_acc_mean = test_acc_arr.mean()
    test_acc_std = test_acc_arr.var().__pow__(0.5)
    test_acc_uncertainty = compute_uncertainty(test_acc_arr)

    print(f' [{setting_name}] '.center(80, '='))
    print('||' + f'All Test Acc: {test_acc_arr}'.center(76, ' ') + '||')
    print('||' + f'Test Acc Mean: {test_acc_mean:.2f} ± {test_acc_uncertainty:.2f} | Std: {test_acc_std:.2f} | Uncertainty: {test_acc_uncertainty:.2f}'.center(76, ' ') + '||')
    print('-'*80)
    
    return float(test_acc_mean), float(test_acc_uncertainty), float(test_acc_std)


def analyse_and_save_recording(recording: dict, analysis_attr_list, save_dir, args):
    analysed_recording = {}

    for analysis_attr in analysis_attr_list:
        test_acc_list = [recording[f'run_[{run_index+1}]'][analysis_attr]['test_mask']['acc'] \
                                             for run_index in range(args.n_runs)]
        test_acc_mean, test_acc_uncertainty, test_acc_std = \
                analyse_one_setting(test_acc_list, 
                                    setting_name = f'{analysis_attr}')
        analysed_recording[analysis_attr] = {}
        analysed_recording[analysis_attr]['test_acc_list'] = test_acc_list
        analysed_recording[analysis_attr]['test_acc_mean'] = test_acc_mean
        analysed_recording[analysis_attr]['test_acc_uncertainty'] = test_acc_uncertainty
        analysed_recording[analysis_attr]['test_acc_std'] = test_acc_std
    
    # save final results
    save_recording(recording=recording, save_dir=save_dir,
                   recording_file = "original_recording.json")
    
    save_recording(recording=analysed_recording, save_dir=save_dir,
                   recording_file = "analysed_recording.json")


def save_recording(recording: dict, save_dir,
                   recording_file = "analysed_recording.json"):
    # save final results
    save_path_json = os.path.join(save_dir, recording_file)
    with open(save_path_json, 'w') as f:
        json.dump(recording, f)
    print(f"Save Recording in JSON File: [{save_path_json}]")