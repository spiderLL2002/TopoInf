import time

import torch
import torch.nn.functional as F
import random
from math import exp
from torch_geometric.nn import CorrectAndSmooth
from torch_geometric.utils import one_hot ,degree
from collections import defaultdict
from torch_sparse import SparseTensor

from .base_training_utils import eval_pred, print_eval_result
from collections import defaultdict

def get_pseudo_label_matrix(model, data, args):
    ### Get Init Pseudo Label ###
    with torch.no_grad():
        model.eval()
        logits = model(data)    ## TODO: use MLP trained in GCN?
        if 'pseudo_label_temperature' in args:
            logits = logits / args.pseudo_label_temperature
        y_soft = F.softmax(logits, dim=1).detach()  # NOTE: `y_soft` is the `pseudo_label_matrix`
    
    if len(args.one_hot_mask) > 0:
        one_hot_masking = data[args.one_hot_mask[0]]
        for mask_name in args.one_hot_mask[1:]:
            one_hot_masking = one_hot_masking | data[mask_name]
    
    if args.use_correct_and_smooth:
        # Define C&S model
        adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                sparse_sizes=(data.num_nodes, data.num_nodes)).to(data.edge_index.device).t()
        adj_t = adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        DAD = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        DA = deg_inv_sqrt.view(-1, 1) * deg_inv_sqrt.view(-1, 1) * adj_t
        
        # TODO: add arguments for C&S in `args`
        post = CorrectAndSmooth(num_correction_layers=3, correction_alpha=0.8,
                                num_smoothing_layers=0, smoothing_alpha=0.0,
                                autoscale=False, scale=1.0)
        
        correct_and_smooth_masking = one_hot_masking if len(args.one_hot_mask) > 0 \
                                        else data.train_mask    # NOTE: at least train_mask can be used

        eval_result = eval_pred(y_soft, data, criterion=None)
        print_eval_result(eval_result, prefix='[Before C&S]')

        y_soft = post.correct(y_soft, data.y[correct_and_smooth_masking], correct_and_smooth_masking, DAD)
        eval_result = eval_pred(y_soft, data, criterion=None)
        print_eval_result(eval_result, prefix='[After Correct]')

        y_soft = post.smooth(y_soft, data.y[correct_and_smooth_masking], correct_and_smooth_masking, DA)
        eval_result = eval_pred(y_soft, data, criterion=None)
        print_eval_result(eval_result, prefix='[After Smooth]')

    if len(args.one_hot_mask) > 0:
        y_soft[one_hot_masking] = one_hot(data.y[one_hot_masking], dtype=y_soft.dtype)

    return y_soft

def get_v2t_t2v (pseudo_label_matrix):
    predicted_classes = torch.argmax(pseudo_label_matrix, dim=1)
    v2t = predicted_classes  
    num_classes = pseudo_label_matrix.size(1)
    t2v = [torch.where(predicted_classes == c)[0] for c in range(num_classes)]

    return v2t, t2v

def select_vertex(vertex_type_degree_ratio,degree_vertex, num_samples = 1500, alpha_degree = 0.1,alpha_ratio = 0.8):
   
    probabilities = torch.exp(-(alpha_ratio * vertex_type_degree_ratio +alpha_degree  * degree_vertex))

    probabilities = torch.nn.functional.softmax(probabilities, dim=0)
    selected_indices = torch.multinomial(probabilities, num_samples=num_samples)

    return selected_indices
def get_add_edge(v2t,add_vertex ,vertex_type_degree_ratio,degree_vertex, num_samples = 1000 ,alpha_degree = 0.1,alpha_ratio = 1.3):
    t2v = defaultdict(list) 
    add_edge = list()
    prob = []
    for node in add_vertex:
        label = v2t[node].item()
        for prenode in t2v[label]:
            prob.append(exp(-alpha_ratio * (vertex_type_degree_ratio[node] + vertex_type_degree_ratio[prenode]) 
                            - alpha_degree * (degree_vertex[node] + degree_vertex[prenode])))
            add_edge.append((int(node), int(prenode)))
        t2v[label].append(node)
        
    prob = torch.tensor(prob)
    prob = torch.nn.functional.softmax(prob, dim=0)
    selected_indices = list(torch.multinomial(prob, num_samples=num_samples))
    
    selected_add_edge = [add_edge[idx] for idx in selected_indices]
    return selected_add_edge

def get_vertex_type_degree_ratio(v2t,data) :
    """ param data: torch_geometric.data.Data 
        return: a tensor
    """
    edge_index = data.edge_index.to(v2t.device) 
    y = v2t

    num_nodes = data.num_nodes
    
    # 计算节点的度，并确保在同一设备上
    degrees = degree(edge_index[0], num_nodes=num_nodes).to(v2t.device)

    # 初始化同一 label 的度计数
    same_label_degrees = torch.zeros(num_nodes, dtype=torch.float, device=v2t.device)
 
    # 遍历所有边，统计同一 label 邻居的度
    for src, dst in edge_index.t():  # edge_index.t() 将边索引变为 [(src, dst), ...]
        if y[src] == y[dst]:  # 同一 label
            same_label_degrees[src] += 1
            same_label_degrees[dst] += 1  # 无向图，两边都需要统计

    # 计算聚合程度比例（避免度为 0 的节点，设置比例为 0）
    aggregation_ratio = (same_label_degrees /2) / degrees
    aggregation_ratio[degrees == 0] = 0  # 度为 0 的节点，设置为 0
    #torch.set_printoptions(profile="full")
    #print(degrees)
    return aggregation_ratio ,degrees

def compute_pseudo_label_topoinf(topoinf_calculator, pseudo_label_matrix, args):
    topoinf_calculator._pre_processing(label_matrix_g = pseudo_label_matrix, 
                                        node_masking = None)
    start_time = time.time()
    topoinf_calculator._pre_processing()
    topoinf_calculator._set_global()
    topoinf_all_e = topoinf_calculator._compute_topoinf_edges_mp(_proc = 24, verbose=True)
    end_time = time.time()
    print(f"Computation Time for All [{len(topoinf_calculator.G.edges)}] Edges on [{args.dataset.upper()}]: {end_time-start_time:.2f} Seconds.")

    return topoinf_all_e

def compute_add_edge_topoinf(topoinf_calculator, pseudo_label_matrix ,add_edge ):
    topoinf_calculator._pre_processing(label_matrix_g = pseudo_label_matrix, 
                                        node_masking = None)
    start_time = time.time()
    topoinf_calculator._pre_processing()
    topoinf_calculator._set_global()
    topoinf_all_e = topoinf_calculator._compute_topoinf_edges_mp(_proc = 24 ,edge_list = add_edge , verbose = True)
    end_time = time.time()
    print(f"Computation add_edge Time  : {end_time-start_time:.2f} Seconds.")

    return topoinf_all_e