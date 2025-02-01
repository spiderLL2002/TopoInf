# TopoInf

Experiment Code for Paper: ["Characterizing the Influence of Topology on Graph Learning Tasks"](https://arxiv.org/abs/2404.07493), which is available on [arXiv](https://arxiv.org/abs/2404.07493) and has been accepted by DASFAA as a long paper.

## Requirements
- `torch >= 2.0.1`
- `torch-geometric >= 2.3.1`

## Introduction
One of the main contributions of our paper is the development of the **TopoInf** metric, which we validated through extensive experiments.

This repository provides code to compute the TopoInf value for each edge in a graph, implemented in `topoinf_impl.py`. The key features include:
1. Computing the TopoInf value for **any given edge**, regardless of its existence in the graph.
2. Supporting **multi-process computation** for efficient TopoInf calculation.

## Example Usage
We have provided an example usage within `topoinf_impl.py`. You can run it directly using `python topoinf_impl.py`.

```python
from topoinf_impl import TopoInf

# STEP1: Load the graph data and turn it to networkx format.
dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())
G = to_networkx(dataset[0], node_attrs=['y'], to_undirected=True)

# STEP2: Initialize the `TopoInf` class.
topoinf_calculator = TopoInf(data, lambda_reg = 1)
topoinf_calculator._pre_processing(label_matrix_g = label_matrix)   # label_matrix can be ground truth or pseudo.

# STEP3: Specify the edges for which TopoInf values need to be computed.
edge_list = None # None for all edges in graph.

# STEP4: Compute and return the results (single-process/multi-process).
topoinf_all_e = topoinf_calculator._compute_topoinf_edges_mp(_proc=24, verbose=True)    # multi-process
topoinf_all_e = topoinf_calculator._compute_topoinf_edges(verbose=True)    # single-process
```


Additionally, the `TopoInf` class includes `visualize_edge_ego_subgraph()` to aid in visualizing and understanding TopoInf values and `get_graph_wise_topoinf_info()` to compute $\mathcal{C}(A)$.

For more details, please refer to the paper and the code.

## Training Experiments
Due to the variety of experimental setups, training procedures may vary. Each set of experiments is implemented in separate directories. The logic for training GNNs is encapsulated in the `RunExp` class. 

To reproduce the experimental results from the paper, place the dataset in `data` directory and run the `run_exp.sh` script located in the relevant `exp-xxxx` folder, or use `make run_exp` to execute the corresponding Makefile.

## Note
Due to the numerous experimental configurations and settings, the code contains logic for saving model results that may appear complex and verbose. This could impact code readability. A cleaner, more streamlined version may be released in the future.
