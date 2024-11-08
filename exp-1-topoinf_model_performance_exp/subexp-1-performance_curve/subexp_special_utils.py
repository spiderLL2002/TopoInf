import os

def get_save_dir(args):
    dataset_model_setting = f"{args.dataset.lower()}_{args.model.lower()}"
    if not args.skip_delete:
        topoinf_delete_setting = f"[{args.delete_strategy}]_based_delete_[{args.delete_unit}]_[{args.delete_mode}]_" \
                                    f"[{args.delete_rate if args.delete_unit in ['mode_ratio', 'ratio'] else args.delete_num}]_" \
                                    f"order_[{args.k_order}]_thr_[{args.topoinf_threshold:.1e}]"
    else:
        topoinf_delete_setting = "no_delete"
    
    save_dir = ''   # NOTE: do not add space in ''
    for sub_dir in [args.perf_save_root_dir, dataset_model_setting, topoinf_delete_setting]:
        save_dir = os.path.join(save_dir, sub_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    return save_dir
