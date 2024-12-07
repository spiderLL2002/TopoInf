# TO RUN: 
#   python topoinf_impl.py
#   OMP_NUM_THREADS=1 python topoinf_impl.py

import random
from tqdm import tqdm
import copy
from typing import Union
import warnings
warnings.filterwarnings('ignore')
from collections import deque
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import torch_geometric
from torch_geometric.utils import spmm
from torch_geometric.utils import one_hot, remove_self_loops
from torch_geometric.utils.convert import to_networkx

from torch_sparse import SparseTensor

import torch.multiprocessing as mp


def get_sorted_subgraph_nodes(subgraph_nodes_set, v_i, v_j, is_sort=True):
    """
    subgraph_nodes_set: (K-1)-subgraph nodes, dtype: set
    edge = (v_i, v_j)
    """
    assert v_i != v_j, "source node and target node can not be the same."
    subgraph_nodes_set.remove(v_i)
    subgraph_nodes_set.remove(v_j)
    
    subgraph_nodes_set = list(subgraph_nodes_set)
    if is_sort:
        subgraph_nodes_set.sort()
    subgraph_nodes = [v_i, v_j] + subgraph_nodes_set    # NOTE: force `v_i`, `v_j` first

    return subgraph_nodes


def get_distance_metric_function(metric_name: str = 'euclidean_distance'):
    r"""
    distance metric to define approximation between `f(A) \times L` and `L`
    # TODO: KL divergence
    """
    assert metric_name in ['inner_product', 'euclidean_distance']
    if metric_name == 'inner_product':
        return lambda left_m, right_m: torch.sum(left_m * right_m, dim=1)
    elif metric_name == 'euclidean_distance':
        return lambda left_m, right_m: - torch.sqrt(torch.sum(torch.square(left_m - right_m), dim=1))


def augmented_normalized_adjacent_matrix(adj_matrix: SparseTensor, 
                                            with_self_loops: bool = True):
    r"""
    calculate augmented normalized adjacent matrix \hat{A} = \tilde{D} ^{-1/2} \tilde{A} \tilde{D}^{-1/2}
    adj_matrix: torch_sparse.SparseTensor
    TODO: augmented random walk matrix \hat{A} = D^{-1} A
    """
    if with_self_loops:
        adj_matrix = adj_matrix.set_diag()
    row_sum = adj_matrix.sum(dim=1)
    # d_inv_sqrt = torch.diag(row_sum.pow(-0.5))    # NOTE: this is a dense tensor
    d_inv_sqrt = torch.sparse.spdiags(diagonals=row_sum.pow(-0.5), offsets=torch.tensor([0]), shape=(len(row_sum), len(row_sum)))    # NOTE: this is a sparse tensor
    d_inv_sqrt = SparseTensor.from_torch_sparse_coo_tensor(d_inv_sqrt)
    norm_adj_matrix = d_inv_sqrt.matmul(adj_matrix).matmul(d_inv_sqrt)

    return norm_adj_matrix


def get_polynomial_graph_filter(norm_adj_matrix: Union[SparseTensor, torch.tensor], 
                                k_order=3, coefficients=None):
    """
    calculate filter `f(A) \ times L` instead of `f(A) \ times L`
    """
    is_SparseTensor = isinstance(norm_adj_matrix, SparseTensor)
    if is_SparseTensor:
        norm_adj_matrix = norm_adj_matrix.to_torch_sparse_coo_tensor()
    
    last_graph_filter_matrix = norm_adj_matrix
    graph_filter = coefficients[0] * last_graph_filter_matrix

    for k in range(1, k_order):
        last_graph_filter_matrix = last_graph_filter_matrix.matmul(norm_adj_matrix)
        graph_filter = graph_filter + coefficients[k] * last_graph_filter_matrix

    if is_SparseTensor:
        graph_filter = SparseTensor.from_torch_sparse_coo_tensor(graph_filter)

    return graph_filter


def get_row_square_sum(norm_adj_matrix: SparseTensor):
    """
    calculate filter `f(A) \ times L` instead of `f(A) \ times L`
    NOTE: filter coefficients start from `1`, not `0`
    """
    norm_adj_matrix = norm_adj_matrix.to_torch_sparse_coo_tensor()
    row_square_sum = (norm_adj_matrix * norm_adj_matrix).sum(dim=1).values()

    return row_square_sum


def sparse_sub(left_m: SparseTensor, right_m: SparseTensor):
    left_m = left_m.to_torch_sparse_coo_tensor()
    right_m = right_m.to_torch_sparse_coo_tensor()

    sub_m = SparseTensor.from_torch_sparse_coo_tensor(left_m - right_m)
    return sub_m


def designed_sparse_mul(org_m: SparseTensor, delta_m: SparseTensor):
    '''
    new_m = org_m + delta_m
    ===>
    new_m^2 = org_m^2 + (2*org_m*delta_m + org_m^2)
    '''
    org_m = org_m.to_torch_sparse_coo_tensor()
    delta_m = delta_m.to_torch_sparse_coo_tensor()

    delta_square_m = 2*org_m*delta_m + delta_m*delta_m
    delta_square_v = delta_square_m.sum(dim=1).values()

    return delta_square_v


def k_neighbors(G, source, cutoff=0, merge=True):
    """
    source can be multiple nodes or single node.
        multiple nodes: [iterable: list or tuple] which means `hasattr(source, '__iter__') == True`; or
        single node:    make sure `source in G.nodes`
    NOTE: finding k-order ego-graph by iteration is much faster (10x) than finding k-order ego-graph by shortest path length
    """
    neighbors = {}  # NOTE: dict, i.e., type({}) == dict
    if isinstance(source, list) or isinstance(source, tuple):
        neighbors[0] = set(source)
    else:
        neighbors[0] = {source}     # NOTE: set, i.e., type({0}) == set
    
    for k in range(1, cutoff+1):
        neighbors[k] = set()
        for node in neighbors[k-1]:     # use last level to speed up
            if node in G:
                neighbors[k].update(set(G.neighbors(node)))
    
    if merge:
        _neighbors = set()
        for v in neighbors.values():
            _neighbors.update(v)
        neighbors = _neighbors
    
    return neighbors


def _topoinf_single_edge(edge):
    """
    Compute TopoInf for edge = (v_i, v_j)
    """
    global _G_g
    global _label_matrix_g, _norm_adj_g, _filtered_label_matrix_g
    global _graph_filter_sparse_g, _graph_filter_row_sum_g, _graph_filter_row_square_sum_g
    global _bias_g, _denoise_g, _lambda_reg_g
    global _k_order_g, _coefficients_g, _distance_metric_function_g
    global _node_masking_g, _with_self_loops_g

    (v_i, v_j) = edge
    has_edge = _G_g.has_edge(v_i, v_j)    # NOTE: equivalent to `edge in G.edges`, but `has_edge` is faster

    subgraph_nodes = k_neighbors(_G_g, (v_i, v_j), _k_order_g-1)
    subgraph_nodes = get_sorted_subgraph_nodes(subgraph_nodes, v_i, v_j, is_sort=True)

    norm_adj_original_sub_g = _norm_adj_g[subgraph_nodes, subgraph_nodes]
    # norm_adj_original_sub_g = norm_adj_original_sub_g.to_dense()    # NOTE: turn into a dense matrix, which sacrifices computation complexity

    original_subgraph_filter = get_polynomial_graph_filter(norm_adj_original_sub_g, k_order=_k_order_g, coefficients=_coefficients_g)

    if has_edge:    # edge in G
        edge_value = - norm_adj_original_sub_g[0, 1].coo()[2].cpu().expand(2)   # NOTE: use `.cpu()` in case `norm_adj_original_sub_g` on GPU
        matrix_E_ij_neg = SparseTensor(row=torch.tensor([0, 1]), col=torch.tensor([1, 0]), value=edge_value,
                                sparse_sizes=norm_adj_original_sub_g.sparse_sizes())
        matrix_E_ij_neg = matrix_E_ij_neg.to(norm_adj_original_sub_g.device())  # NOTE: keep device the same
        norm_adj_disturbed_sub_g = matrix_E_ij_neg + norm_adj_original_sub_g    # NOTE: norm_adj_disturbed_sub_g.nnz == norm_adj_original_sub_g.nnz!
    else:           # edge not in G
        edge_value = (_G_g.degree(v_i) + _with_self_loops_g) * (_G_g.degree(v_j) + _with_self_loops_g)
        edge_value = edge_value + bool(edge_value)  # NOTE: in case `edge_value == 0`
        edge_value = torch.tensor([edge_value]).pow(-0.5).expand(2)
        matrix_E_ij_neg = SparseTensor(row=torch.tensor([0, 1]), col=torch.tensor([1, 0]), value=edge_value,
                                sparse_sizes=norm_adj_original_sub_g.sparse_sizes())
        matrix_E_ij_neg = matrix_E_ij_neg.to(norm_adj_original_sub_g.device())  # NOTE: keep device the same
        norm_adj_disturbed_sub_g = matrix_E_ij_neg + norm_adj_original_sub_g

    disturbed_subgraph_filter = get_polynomial_graph_filter(norm_adj_disturbed_sub_g, k_order=_k_order_g, coefficients=_coefficients_g)
    delta_subgraph_filter = sparse_sub(disturbed_subgraph_filter, original_subgraph_filter)

    disturbed_subgraph_filter_row_sum = delta_subgraph_filter.sum(dim=1) + _graph_filter_row_sum_g[subgraph_nodes]
    disturbed_subgraph_filter_row_square_sum = designed_sparse_mul(org_m = _graph_filter_sparse_g[subgraph_nodes, subgraph_nodes],
                        delta_m = delta_subgraph_filter
                        ) + _graph_filter_row_square_sum_g[subgraph_nodes]

    label_matrix_sub_g = _label_matrix_g[subgraph_nodes]

    delta_filtered_label_matrix_sub_g = spmm(delta_subgraph_filter, label_matrix_sub_g)
    original_filtered_label_matrix_sub_g = _filtered_label_matrix_g[subgraph_nodes]
    filtered_label_matrix_sub_g = original_filtered_label_matrix_sub_g + delta_filtered_label_matrix_sub_g
    # normalized_filtered_label_matrix_sub_g = self.normalize(filtered_label_matrix_sub_g)  # NOTE: row_sum has already been calculated
    normalized_filtered_label_matrix_sub_g = filtered_label_matrix_sub_g / disturbed_subgraph_filter_row_sum.reshape(-1, 1)
    
    bias_sub_g = _distance_metric_function_g(normalized_filtered_label_matrix_sub_g, label_matrix_sub_g) \
        - _bias_g[subgraph_nodes]
    denoise_sub_g = disturbed_subgraph_filter_row_square_sum.sqrt() / disturbed_subgraph_filter_row_sum \
        - _denoise_g[subgraph_nodes]

    if _node_masking_g is not None:
        subgraph_node_masking = [True if subg_node in _node_masking_g else False for subg_node in subgraph_nodes]
        # NOTE: this means nodes in `node_masking` will be calculated in TopoInf.
        bias_sub_g = bias_sub_g[subgraph_node_masking]
        denoise_sub_g = denoise_sub_g[subgraph_node_masking]
        
    bias_e_v = torch.sum(bias_sub_g).item()
    denoise_e_v = torch.sum(denoise_sub_g).item()

    topoinf_e_v = bias_e_v - _lambda_reg_g * denoise_e_v
    #C(A) = T(A) - lambda * Ra(Vi) , now is delta C(A)

    return ((v_i, v_j), [topoinf_e_v, bias_e_v, denoise_e_v])


def _compute_topoinf_edge_list(edge_list = None, _proc = int(mp.cpu_count()/2), verbose = False):
    
    num_tasks = len(edge_list)
    chunksize, extra = divmod(num_tasks, _proc * 4)
    if extra:
        chunksize += 1

    org_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)        # NOTE: avoid multiprocessing hanging bug!

    pool = mp.Pool(processes=_proc)
    # topoinf_all_e_list = pool.map(_topoinf_single_edge, edge_list, chunksize=chunksize)
    topoinf_all_e_list = pool.imap_unordered(_topoinf_single_edge, edge_list, chunksize=chunksize)
    
    if verbose:
        topoinf_all_e_list = list(tqdm(topoinf_all_e_list, total=num_tasks, desc=f'Computing TopoInf (MP): '))
    
    pool.close()
    pool.join()     # NOTE: all tasks are done after this command.

    torch.set_num_threads(org_num_threads)
    topoinf_all_e = dict(topoinf_all_e_list)

    return topoinf_all_e



class TopoInf:
    """A class to compute TopoInf for all edges in G.
    """
    def __init__(   self, data: torch_geometric.data.Data, 
                    lambda_reg: float = 1,
                    with_self_loops: bool = True,
                    k_order: int = 3, coefficients: list = None,
                    distance_metric_name: str = 'inner_product',
                    edges_haven_deleted : set = (),
                ):
        """Initialized a container to compute TopoInf.
        """
        self.data = copy.deepcopy(data).cpu()
        self.data.edge_index, _ = remove_self_loops(self.data.edge_index)
        self.with_self_loops = with_self_loops
        
        # turn pgy to networkx
        self.G = to_networkx(self.data, node_attrs=['y'], to_undirected=True, remove_self_loops=True)
        self.k_order = k_order
        if coefficients is None or len(coefficients) < k_order:
            coefficients = [1/k_order] * k_order
        self.coefficients = coefficients

        distance_metric_name = distance_metric_name if distance_metric_name \
            in ['inner_product', 'euclidean_distance'] else 'inner_product'
        self.distance_metric_function = get_distance_metric_function(distance_metric_name)
        self.lambda_reg = lambda_reg
        self.normalize = lambda mat: F.normalize(mat, p=1, dim=1, eps=1e-6)     # NOTE: `1e-6` is indispensable to deal with all zero vector

        self.eps = 1e-6
        self.computed_topoinf = {}
        self.computed_topoinf_detailed = {}
        self.edges_haven_deleted = edges_haven_deleted
    def _pre_processing(self, label_matrix_g: torch.Tensor = None, node_masking: Union[set, list, tuple] = None):
        if (label_matrix_g == None) or (not isinstance(label_matrix_g, torch.Tensor)):
            self.label_matrix_g = one_hot(self.data.y, dtype=torch.float32)
        else:
            self.label_matrix_g = label_matrix_g
        self.num_classes = self.label_matrix_g.size(1)

        if (node_masking is not None) and (not isinstance(node_masking, set)):
            node_masking = set(node_masking)      # NOTE: turn `sequence` to `set` so that we can get `O(1)` performance.
        self.node_masking = node_masking

        self.adj_g = SparseTensor(row=self.data.edge_index[0], col=self.data.edge_index[1],
                                sparse_sizes=(self.data.num_nodes, self.data.num_nodes))
        self.norm_adj_g = augmented_normalized_adjacent_matrix(self.adj_g, self.with_self_loops)
        self.graph_filter_sparse_g = get_polynomial_graph_filter(self.norm_adj_g, k_order=self.k_order, coefficients=self.coefficients)
        
        self.graph_filter_row_sum = self.graph_filter_sparse_g.sum(dim=1)
        self.graph_filter_row_square_sum = get_row_square_sum(self.graph_filter_sparse_g)

        self.filtered_label_matrix_g = spmm(self.graph_filter_sparse_g, self.label_matrix_g)
        # self.normalized_filtered_label_matrix_g = self.normalize(self.filtered_label_matrix_g)    # NOTE: row_sum has already been calculated
        self.normalized_filtered_label_matrix_g = self.filtered_label_matrix_g / self.graph_filter_row_sum.reshape(-1, 1)

        self.bias_g = self.distance_metric_function(self.normalized_filtered_label_matrix_g, self.label_matrix_g)
        self.denoise_g = self.graph_filter_row_square_sum.sqrt() / self.graph_filter_row_sum

        self.node_wise_topoinf = self.bias_g - self.lambda_reg * self.denoise_g

    def _set_global(self):
        global _G_g
        global _label_matrix_g, _norm_adj_g, _filtered_label_matrix_g
        global _graph_filter_sparse_g, _graph_filter_row_sum_g, _graph_filter_row_square_sum_g
        global _bias_g, _denoise_g, _lambda_reg_g
        global _k_order_g, _coefficients_g, _distance_metric_function_g
        global _node_masking_g, _with_self_loops_g

        _G_g = self.G

        _label_matrix_g = self.label_matrix_g
        _norm_adj_g = self.norm_adj_g
        _filtered_label_matrix_g = self.filtered_label_matrix_g

        _graph_filter_sparse_g = self.graph_filter_sparse_g
        _graph_filter_row_sum_g = self.graph_filter_row_sum
        _graph_filter_row_square_sum_g = self.graph_filter_row_square_sum

        _bias_g = self.bias_g
        _denoise_g = self.denoise_g
        _lambda_reg_g = self.lambda_reg

        _k_order_g = self.k_order
        _coefficients_g = self.coefficients
        _distance_metric_function_g = self.distance_metric_function

        _node_masking_g = self.node_masking
        _with_self_loops_g = self.with_self_loops

    def _to_device(self, device: torch.device = torch.device('cpu')):
        self.label_matrix_g = self.label_matrix_g.to(device)
        self.norm_adj_g = self.norm_adj_g.to(device)
        self.filtered_label_matrix_g = self.filtered_label_matrix_g.to(device)
        self.bias_g = self.bias_g.to(device)

    def _check_edge_existence(self, edge):
        v_i, v_j = edge
        return self.G.has_edge(v_i, v_j)

    def random_sample_edges(self, sample_k: int = None, sample_nodes_iter = None, batch_size: int = None):
        sample_k = self.G.number_of_edges() if sample_k is None else sample_k
        sample_nodes_iter = self.G.nodes if sample_nodes_iter is None else sample_nodes_iter
        num_nodes = len(sample_nodes_iter)
        batch_size = int(num_nodes/4) if batch_size is None or 2*batch_size <= num_nodes else batch_size # NOTE: `2*batch_size` shoule not greater than `num_nodes`

        quotient, remainder = divmod(sample_k, batch_size)
        
        sampled_edges = tuple()     # NOTE: equivalent to `sampled_edges = ()`
        for _ in range(quotient):
            sampled_nodes = random.sample(sample_nodes_iter, 2*batch_size)
            batch_sampled_edges = *zip(sampled_nodes[:batch_size], sampled_nodes[batch_size:]),       # NOTE: `,` is necessary for starred expression
            sampled_edges = sampled_edges + batch_sampled_edges

        sampled_nodes = random.sample(sample_nodes_iter, 2*remainder)
        batch_sampled_edges = *zip(sampled_nodes[:remainder], sampled_nodes[remainder:]),       # NOTE: `,` is necessary for starred expression
        sampled_edges = sampled_edges + batch_sampled_edges
        
        return sampled_edges
    
    def update_edge_index(self, update_edges: Union[list, tuple], return_networkx=False):
        """Delete/Add edges according to update_edges.
        """
        G_networkx = self.G.copy()
        delete_edges = []
        add_edges = []
        for edge in update_edges:
            if G_networkx.has_edge(*edge):  # edge in G
                delete_edges.append(edge)
            else:   # edge not in G
                add_edges.append(edge)
        
        G_networkx.remove_edges_from(delete_edges)
        G_networkx.add_edges_from(add_edges)
        # updated edge_index
        edge_index = torch.tensor(np.array(G_networkx.to_directed().edges).T)

        if return_networkx:
            return G_networkx, edge_index
        else:
            return edge_index

    def _topoinf_e(self, edge):
        """
        Compute TopoInf for edge = (v_i, v_j)
        """
        (v_i, v_j) = edge
        has_edge = self.G.has_edge(v_i, v_j)    # NOTE: equivalent to `edge in self.G.edges`, but `has_edge` is faster
    
        subgraph_nodes = k_neighbors(self.G, (v_i, v_j), self.k_order-1)
        subgraph_nodes = get_sorted_subgraph_nodes(subgraph_nodes, v_i, v_j, is_sort=True)

        norm_adj_original_sub_g = self.norm_adj_g[subgraph_nodes, subgraph_nodes]
        # norm_adj_original_sub_g = norm_adj_original_sub_g.to_dense()    # NOTE: turn into a dense matrix, which sacrifices computation complexity

        original_subgraph_filter = get_polynomial_graph_filter(norm_adj_original_sub_g, k_order=self.k_order, coefficients=self.coefficients)
        
        # NOTE: sparse version
        if has_edge:    # edge in G
            
            coo_values = norm_adj_original_sub_g[0, 1].coo()[2]
            if coo_values.numel() > 0:  
                edge_value = - coo_values.cpu().expand(2)
            else:
                edge_value = torch.tensor([0, 0], dtype=torch.float32)
                
            matrix_E_ij_neg = SparseTensor(row=torch.tensor([0, 1]), col=torch.tensor([1, 0]), value=edge_value,
                                    sparse_sizes=norm_adj_original_sub_g.sparse_sizes())
            matrix_E_ij_neg = matrix_E_ij_neg.to(norm_adj_original_sub_g.device())  # NOTE: keep device the same
            norm_adj_disturbed_sub_g = matrix_E_ij_neg + norm_adj_original_sub_g    # NOTE: norm_adj_disturbed_sub_g.nnz == norm_adj_original_sub_g.nnz!
        else:           # edge not in G
            edge_value = (self.G.degree(v_i) + self.with_self_loops) * (self.G.degree(v_j) + self.with_self_loops)
            edge_value = edge_value + bool(edge_value == 0)  # NOTE: in case `edge_value == 0`
            edge_value = torch.tensor([edge_value]).pow(-0.5).expand(2)
            matrix_E_ij_pos = SparseTensor(row=torch.tensor([0, 1]), col=torch.tensor([1, 0]), value=edge_value,
                                    sparse_sizes=norm_adj_original_sub_g.sparse_sizes())
            matrix_E_ij_pos = matrix_E_ij_neg.to(norm_adj_original_sub_g.device())  # NOTE: keep device the same
            norm_adj_disturbed_sub_g = matrix_E_ij_pos + norm_adj_original_sub_g
        
        disturbed_subgraph_filter = get_polynomial_graph_filter(norm_adj_disturbed_sub_g, k_order=self.k_order, coefficients=self.coefficients)
        delta_subgraph_filter = sparse_sub(disturbed_subgraph_filter, original_subgraph_filter)

        disturbed_subgraph_filter_row_sum = delta_subgraph_filter.sum(dim=1) + self.graph_filter_row_sum[subgraph_nodes]
        disturbed_subgraph_filter_row_square_sum = designed_sparse_mul(org_m = self.graph_filter_sparse_g[subgraph_nodes, subgraph_nodes],
                            delta_m = delta_subgraph_filter
                            ) + self.graph_filter_row_square_sum[subgraph_nodes]

        label_matrix_sub_g = self.label_matrix_g[subgraph_nodes]

        delta_filtered_label_matrix_sub_g = spmm(delta_subgraph_filter, label_matrix_sub_g)
        original_filtered_label_matrix_sub_g = self.filtered_label_matrix_g[subgraph_nodes]
        filtered_label_matrix_sub_g = original_filtered_label_matrix_sub_g + delta_filtered_label_matrix_sub_g
        # normalized_filtered_label_matrix_sub_g = self.normalize(filtered_label_matrix_sub_g)  # NOTE: row_sum has already been calculated
        normalized_filtered_label_matrix_sub_g = filtered_label_matrix_sub_g / disturbed_subgraph_filter_row_sum.reshape(-1, 1)
        
        bias_sub_g = self.distance_metric_function(normalized_filtered_label_matrix_sub_g, label_matrix_sub_g) \
            - self.bias_g[subgraph_nodes]
        denoise_sub_g = disturbed_subgraph_filter_row_square_sum.sqrt() / disturbed_subgraph_filter_row_sum \
            - self.denoise_g[subgraph_nodes]

        if self.node_masking is not None:
            subgraph_node_masking = [True if subg_node in self.node_masking else False for subg_node in subgraph_nodes]
            # NOTE: this means nodes in `node_masking` will be calculated in TopoInf! Do not get the opposite understanding.
            bias_sub_g = bias_sub_g[subgraph_node_masking]
            denoise_sub_g = denoise_sub_g[subgraph_node_masking]
            
        bias_e_v = torch.sum(bias_sub_g).item()
        denoise_e_v = torch.sum(denoise_sub_g).item()

        topoinf_e_v = bias_e_v - self.lambda_reg * denoise_e_v
        
        return topoinf_e_v, bias_e_v, denoise_e_v
    
    def visualize_edge_ego_subgraph(self, analysis_edge):
        analysis_topoinf = self._topoinf_e(analysis_edge)
        subgraph_nodes = k_neighbors(self.G, analysis_edge, cutoff=self.k_order-1)
        subgraph = self.G.subgraph(subgraph_nodes)

        v_i, v_j = analysis_edge
        has_edge = self.G.has_edge(v_i, v_j)
        if not has_edge:
            subgraph = nx.Graph(subgraph)   # NOTE: this is used to unfreeze the graph!
            subgraph.add_edge(v_i, v_j)

        node_dist_to_color = {
            0: "tab:purple",
            1: "tab:red",
            2: "tab:orange",
            3: "tab:olive",
            4: "tab:green",
            5: "tab:blue",
            6: "violet",
            7: "limegreen",
            8: "darkorange",
            9: "gold",
        }
        _node_colors = [node_dist_to_color[c] for _, c in subgraph.nodes(data="y")]

        _pos = nx.spring_layout(subgraph)
        plt.figure(figsize=(8, 8))

        edgelist_except_analysis = subgraph.edges - [(v_i, v_j), (v_j, v_i)]
        nx.draw_networkx_edges(subgraph, _pos, edgelist=edgelist_except_analysis, width=2, alpha=0.7, edge_color="k")
        if has_edge:    # solid line if edge in G
            nx.draw_networkx_edges(subgraph, _pos, edgelist=[analysis_edge], style='solid', width=2, alpha=0.7, edge_color="b")
        else:    # dashed line if edge in G
            nx.draw_networkx_edges(subgraph, _pos, edgelist=[analysis_edge], style='dashed', width=2, alpha=0.7, edge_color="r")
        nx.draw_networkx_nodes(subgraph, _pos, node_color=_node_colors)
        nx.draw_networkx_labels(subgraph, _pos, font_size=8)
        plt.axis("off")
        plt.title(f"{self.k_order-1}-Subgraph of Edge ${analysis_edge}$\nTopoInf = {analysis_topoinf:.2f}")
        plt.show()
    
    def visualize_topoinf_distribution(self):
        topoinf_all_e_tensor = torch.tensor(list(self.computed_topoinf.values()))
        n, bins, patches = plt.hist(topoinf_all_e_tensor, 50, density=False, facecolor='C0', alpha=0.75)
        plt.title(f"Distribution of TopoInf")
        plt.show()

    def get_edges_within_range(self, low, high=None, eps=1e-6):
        high = low + eps if high == None else high
        edges_within_range = []
        for key, value in self.computed_topoinf.items():
            if value >= low and value <= high:
                edges_within_range.append(key)

        return edges_within_range
    
    def get_graph_wise_topoinf_info(self):
        if self.node_masking is not None:
            node_masking_indices = torch.tensor(list(self.node_masking))
            node_wise_topoinf = self.bias_g[node_masking_indices]
        else:
            node_wise_topoinf = self.bias_g
        
        info = {
            'graph_wise_topoinf': torch.sum(node_wise_topoinf).item(),
            'graph_wise_topoinf_log': torch.sum(torch.log10(node_wise_topoinf)).item(),
            'number_of_target_nodes': len(node_wise_topoinf),
        }

        return info

    def neighbors_of_delete_edges(self, delete_edges: list = None) : 
        neighbors_edge = set()
        haven_edges = set(self.edges_haven_deleted)
        haven_point = set()
        q = deque() 
        for (i,j) in delete_edges  :
            haven_edges.add((i,j))
            haven_edges.add((j,i))
            q.append(i)
            q.append(j)
        while q : 
            now_node = q.popleft() 
            if now_node not in haven_point and now_node in self.G  :
                haven_point.add(now_node)
                for neighbor in self.G.neighbors(now_node):
                    if (now_node,neighbor) not in haven_edges : 
                        neighbors_edge.add((now_node,neighbor))
                        haven_edges.add((now_node,neighbor))
                        haven_edges.add((neighbor,now_node))
                        if now_node not in haven_point :
                            q.append(neighbor)
        return list(neighbors_edge)
        
    def _compute_topoinf_edges(self, edge_list: list = None, verbose: bool = False):
        """Compute TopoInf for edges in given edge lists.
        Parameters
        ----------
        edge_list : list of edges
            The list of edges to compute TopoInf, set to `None` to run for all edges in G. (Default value = `None`)
        
        Returns
        -------
        output : 
            A dictionary of edge TopoInf. E.g.: {(v_i, v_j): TopoInf}.
        """

        if (edge_list == None) or (not isinstance(edge_list, (tuple, list))):
            edge_list = self.G.edges()  # list(self.G.edges())
        
        if verbose:
            edge_list = tqdm(edge_list, desc=f'Computing TopoInf (SP): ')

        topoinf_all_e = {}
        topoinf_detail_all_e = {}
        for edge in edge_list:
            topoinf_e_v, bias_e_v, denoise_e_v = self._topoinf_e(edge=edge)
            topoinf_all_e[edge] = topoinf_e_v
            topoinf_detail_all_e[edge] = [topoinf_e_v, bias_e_v, denoise_e_v]

        self.computed_topoinf.update(topoinf_all_e)
        self.computed_topoinf_detailed.update(topoinf_detail_all_e)
        
        return topoinf_all_e
    
    def update_topoinf_edges(self, delete_list: list = None, verbose: bool = False  ,topoinf_all = None ):
        if (delete_list == None) or (not isinstance(delete_list, (tuple, list))):
            delete_list = self.G.edges()  # list(self.G.edges())
        edge_list = self.neighbors_of_delete_edges( delete_list)
        #print("本次删除的边" , delete_list)
        #print("需要update的边的数量", len(edge_list))
        #print("update了哪些边",print(edge_list))
        if verbose:
            edge_list = tqdm(edge_list, desc=f'Computing TopoInf (SP): ')
        topoinf_all_e = copy.deepcopy(topoinf_all)
        topoinf_detail_all_e = {}
        for edge in edge_list:
            topoinf_e_v, bias_e_v, denoise_e_v = self._topoinf_e(edge=edge)
            topoinf_all_e[edge] = topoinf_e_v
            topoinf_detail_all_e[edge] = [topoinf_e_v, bias_e_v, denoise_e_v]

        self.computed_topoinf.update(topoinf_all_e)
        self.computed_topoinf_detailed.update(topoinf_detail_all_e)
        
        return topoinf_all_e
    
    def update_topoinf_edges_mp(self, delete_list: list = None, _proc: int = int(mp.cpu_count()/2), 
                                  verbose: bool = False ,topoinf_all = None ):
        if (delete_list == None) or (not isinstance(delete_list, (tuple, list))):
            delete_list = self.G.edges()  # list(self.G.edges())
            
        edge_list = self.neighbors_of_delete_edges( delete_list)
        #print("本次删除的边" , delete_list)
        #print("需要update的边的数量", len(edge_list))
        #print("update了哪些边",print(edge_list))
        
        topoinf_all_e = copy.deepcopy(topoinf_all)
        topoinf_detail_all_e = _compute_topoinf_edge_list(edge_list = edge_list, _proc = _proc, verbose = verbose)
        self.computed_topoinf_detailed.update(topoinf_detail_all_e)

        for edge in edge_list  : 
            topoinf_all_e[edge] = topoinf_detail_all_e[edge][0]
        
        return topoinf_all_e
        
    
    def _compute_topoinf_edges_mp(self, edge_list: list = None, _proc: int = int(mp.cpu_count()/2), 
                                  verbose: bool = False):
        """Compute TopoInf for edges in given edge lists **IN A MULTIPROCESSING WAY**.
        Parameters
        ----------
        edge_list : list of edges
            The list of edges to compute TopoInf, set to `None` to run for all edges in G. (Default value = `None`)
        _proc : int
            Number of processes. (Default value = `int(mp.cpu_count()/2)`)
        
        Returns
        -------
        output : 
            A dictionary of edge TopoInf. E.g.: {(v_i, v_j): TopoInf}.
        """
        if edge_list is None:
            edge_list = self.G.edges
        self._set_global()
        
        topoinf_detail_all_e = _compute_topoinf_edge_list(edge_list = edge_list, _proc = _proc, verbose = verbose)
        self.computed_topoinf_detailed.update(topoinf_detail_all_e)

        topoinf_all_e = {key: value[0] for key, value in topoinf_detail_all_e.items()}
        
        return topoinf_all_e
    

if __name__ == '__main__':
    import os.path as osp
    import time
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T

    print(torch.__config__.parallel_info())

    dataset_name = 'cora'
    root_path = '../'
    path = osp.join(root_path, 'data', dataset_name)
    dataset = Planetoid(path, dataset_name, transform=T.NormalizeFeatures())

    data = dataset[0]
    # turn pgy to networkx
    G = to_networkx(data, node_attrs=['y'], to_undirected=True)

    ## Single Processing
    topoinf_calculator = TopoInf(data, lambda_reg = 1)
    topoinf_calculator._pre_processing()

    start_time = time.time()
    topoinf_all_e_sp_reg = topoinf_calculator._compute_topoinf_edges(verbose=True)
    end_time = time.time()
    print(f"Single Core Time for All {len(topoinf_calculator.G.edges)} Edges: {end_time-start_time:.2f} Seconds.")

    ## Multi-Processing
    topoinf_calculator = TopoInf(data, lambda_reg = 1)
    topoinf_calculator._pre_processing()

    start_time = time.time()
    topoinf_all_e_mp_reg = topoinf_calculator._compute_topoinf_edges_mp(_proc=24, verbose=True)
    end_time = time.time()
    print(f"Multi Core Time for All {len(topoinf_calculator.G.edges)} Edges: {end_time-start_time:.2f} Seconds.")

    assert topoinf_all_e_mp_reg == topoinf_all_e_sp_reg