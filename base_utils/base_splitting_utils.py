import torch

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask


def set_pyg_data_masking_attr(data):
    """ set train/valid/test masking attributes for pyg data 
        because some datasets (i.e., computers and photo in Amazon) have no train/valid/test masking attribute
    """
    all_masking = ['train_mask', 'val_mask', 'test_mask']
    for masking in all_masking:
        if masking not in data:
            data[masking] = torch.zeros_like(data.y, dtype=torch.bool)

def set_pyg_data_num_classes_attr(data):
    """ set number of classes attributes for pyg data """
    if 'num_classes' not in data:
        # data.num_classes = data.y.max().item()    # BUG: do not forget to plus `1`!
        data.num_classes = data.y.max().item() + 1


def check_pyg_data(data):
    """check validation of pyg data"""
    set_pyg_data_masking_attr(data)
    set_pyg_data_num_classes_attr(data)


def print_pyg_data_split(data):
    print(f"#(train set): {data.train_mask.sum().item()} | "
          f"#(valid set): {data.val_mask.sum().item()} | "
          f"#(test set): {data.test_mask.sum().item()} | "
          )
    

def rand_train_val_test_split(data, num_train_per_class=20, num_val=500, num_test=1000):
    """ randomly splits label into train/valid/test splits """
    check_pyg_data(data)

    data.train_mask.fill_(False)
    for c in range(data.num_classes):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
        data.train_mask[idx] = True

    remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.size(0))]

    data.val_mask.fill_(False)
    data.val_mask[remaining[:num_val]] = True

    data.test_mask.fill_(False)
    if num_test is None or num_test <= 0:   # NOTE: test_set = node_set - train_set - val_set
        data.test_mask[remaining[num_val:]] = True
    else:   # NOTE: #(test_set) = num_test
        data.test_mask[remaining[num_val:num_val + num_test]] = True


def rand_train_val_test_split_wrapper(data, args):
    if args.split_mode == 'ratio':
        num_train_per_class = int(round(args.train_rate*data.num_nodes/data.num_classes))
        num_val = int(round(args.val_rate*data.num_nodes))
        num_test = None
    else:
        num_train_per_class = args.num_train_per_class
        num_val = args.num_val
        num_test = args.num_test
    
    rand_train_val_test_split(data, num_train_per_class=num_train_per_class, 
                              num_val=num_val, num_test=num_test)


if __name__ == '__main__':
    from dataset_loader import DataLoader

    dataset_name = 'cora'

    dataset = DataLoader(dataset_name)
    data = dataset[0]
    data.num_classes = dataset.num_classes

    print_pyg_data_split(data)

    ## number-based splitting
    num_train_per_class = 20
    num_val = 500
    num_test = 1000
    rand_train_val_test_split(data, num_train_per_class=num_train_per_class, num_val=num_val, num_test=num_test)
    print_pyg_data_split(data)

    ## ratio-based splitting
    train_ratio = 0.6
    val_ratio = 0.2
    num_train_per_class = int(data.num_nodes * train_ratio / data.num_classes)
    num_val = int(data.num_nodes * val_ratio)
    num_test = data.num_nodes - num_train_per_class*data.num_classes - num_val
    rand_train_val_test_split(data, num_train_per_class=num_train_per_class, num_val=num_val, num_test=num_test)
    print_pyg_data_split(data)

    print(f'Tested PyG Data Splitting!')

