import os.path as osp
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Actor
from torch_geometric.datasets import WebKB, HeterophilousGraphDataset
from torch_geometric.utils import remove_self_loops, to_undirected
from torch_geometric.utils import subgraph
 
from ogb.nodeproppred import PygNodePropPredDataset
import time

_ALL_DATASETS_ = [ 'computers', 'photo', 'actor' ,'cora', 'citeseer', 'pubmed']

def DataLoader(dataset_name, root_path = '../data/', with_inductive_info = False):
    dataset_name = dataset_name.lower()

    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root_path, name=dataset_name, transform=T.NormalizeFeatures(), split='public')
    elif dataset_name in ['computers', 'photo']:
        dataset = Amazon(root=root_path, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['actor']:
        path = osp.join(root_path, dataset_name)
        dataset = Actor(root=path, transform=T.NormalizeFeatures())
    elif dataset_name in ['texas', 'wisconsin']:
        dataset = WebKB(root=root_path, name=dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['amazon-ratings', 'roman-empire']:
        dataset = HeterophilousGraphDataset(root=root_path, name=dataset_name)      # , transform=T.NormalizeFeatures()
    elif dataset_name in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(dataset_name, root_path)       # , transform=T.NormalizeFeatures()
    else:
        raise ValueError(f'dataset {dataset_name} not supported in dataloader')

    data = dataset[0]
    data.num_classes = dataset.num_classes
    data.name = dataset_name

    # special process
    if dataset_name in ['ogbn-arxiv']:   # NOTE: turn index into mask
        data.y = data.y.flatten()

        split_idx = dataset.get_idx_split()
        num_nodes = data.y.size(0)
        data.train_mask = torch.zeros(num_nodes).bool().scatter_(0, split_idx['train'], True)
        data.val_mask = torch.zeros(num_nodes).bool().scatter_(0, split_idx['valid'], True)
        data.test_mask = torch.zeros(num_nodes).bool().scatter_(0, split_idx['test'], True)

    # turn directed graph to undirected
    directed_datasets = ['actor', 'texas', 'wisconsin', 'amazon-ratings', 'roman-empire', 'ogbn-arxiv']
    if dataset_name in directed_datasets:   # NOTE: these datasets are undirected graphs.
        data.edge_index, _ = remove_self_loops(data.edge_index)
        data.edge_index = to_undirected(data.edge_index)

    # add inductive information
    if with_inductive_info:
        add_inductive_info(data)

    return data


def add_inductive_info(data):
    train_index = data.train_mask.nonzero().flatten()
    data.train_edge_index = subgraph(train_index, 
                                     data.edge_index)[0]
    data.val_edge_index = subgraph(torch.cat([train_index, data.val_mask.nonzero().flatten()], dim=-1), 
                                   data.edge_index)[0]
    data.test_edge_index = data.edge_index
    data.edge_index = data.train_edge_index


def analyse_class_distribution(data):
    num_nodes = data.num_nodes
    for c in range(data.num_classes):
        n_nodes_c = (data.y == c).sum().item()
        print(f"CLASS {c}/{data.num_classes-1}: {n_nodes_c}/{num_nodes}.")


if __name__ == '__main__':
    while True  :
        try:
            for dataset_name in _ALL_DATASETS_:
                data = DataLoader(dataset_name)
                print(f"{dataset_name.upper()}: {data}")
                analyse_class_distribution(data)
                print('-'*50)
        except Exception as e:
            print("Fuck!")
        print(f'Tested All {len(_ALL_DATASETS_)} Datasets: {_ALL_DATASETS_}')
        
        time.sleep(100)