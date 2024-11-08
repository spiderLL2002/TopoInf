import torch
import torch.nn.functional as F

import time

def get_edge_sample_prob(topoinf_all_e, data, 
                            temperature = 0.1, 
                            thr_v = None, 
                            return_prob = True):
    """
    temperature > 0 for dropping positive edges (this usually gets better performance)
    temperature < 0 for dropping negative edges
    """
    edge_sample_prob_list = []
    for v_i, v_j in data.edge_index.cpu().T.numpy():
        if v_i == v_j:      # self loop
            # topoinf_e_v = 0
            raise ValueError('No TopoInf for self loop')
        elif (v_i, v_j) in topoinf_all_e:
            topoinf_e_v = topoinf_all_e[(v_i, v_j)]
        else:
            topoinf_e_v = topoinf_all_e[(v_j, v_i)]
        if thr_v is not None and thr_v > 0:
            topoinf_e_v = min(max(topoinf_e_v, -thr_v), thr_v)
        edge_sample_prob_list.append(topoinf_e_v)

    edge_sample_prob = torch.tensor(edge_sample_prob_list)
    if return_prob:
        edge_sample_prob = F.softmax(edge_sample_prob / temperature)

    return edge_sample_prob


def guided_dropout_edge(edge_index: torch.Tensor, p: float = 0.5, edge_sample_prob: torch.Tensor = None,
                 force_undirected: bool = False,
                 training: bool = True):

    if not training or p == 0.0:
        return edge_index

    row, col = edge_index

    if edge_sample_prob is None:    # uniform
        edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p
    else:   # sample according to prob
        num_edges = edge_index.size(1)
        edge_sample_prob = edge_sample_prob.to(edge_index.device)
        prob_to_comp = num_edges * p * edge_sample_prob
        # NOTE: if prob in prob_to_comp great than 1, then divide extra prob to all other edges
        ## calculate extra prob
        prob_great_than_one_mask = prob_to_comp > 1
        num_mask_indices = prob_great_than_one_mask.sum()
        extra_prob = prob_to_comp[prob_great_than_one_mask].sum() - num_mask_indices
        ## share extra prob
        prob_to_comp[prob_great_than_one_mask] = 1
        prob_to_comp[~prob_great_than_one_mask] += extra_prob / (num_edges - num_mask_indices)

        edge_mask = torch.rand(row.size(0), device=edge_index.device) >= prob_to_comp

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # print(f"#(DropEdge): {edge_mask.size(0)} - {edge_index.size(1)} = {edge_mask.size(0) - edge_index.size(1)}")

    return edge_index



def compute_pseudo_label_topoinf_wrapper(topoinf_calculator, pseudo_label_matrix, data, args):

    if 'topoinf_node_masking' in args and args.topoinf_node_masking is not None and len(args.topoinf_node_masking) > 0:
        masking_nodes_indices = data[args.topoinf_node_masking[0]]
        for mask_name in args.topoinf_node_masking[1:]:
            masking_nodes_indices = masking_nodes_indices | data[mask_name]
        node_masking = set(torch.where(masking_nodes_indices)[0].tolist())
        print(f'[TOPOING MASKING INFO] #(node masking [{args.topoinf_node_masking}]): [{len(node_masking)}]')
    else:
        node_masking = None
        print('[TOPOING MASKING INFO] #(node masking): NONE')
    
    topoinf_calculator._pre_processing(label_matrix_g = pseudo_label_matrix, 
                                        node_masking = node_masking)

    start_time = time.time()
    topoinf_all_e = topoinf_calculator._compute_topoinf_edges_mp(_proc=args.mp_core, verbose=True)
    end_time = time.time()
    print(f"Computation Time for All [{len(topoinf_calculator.G.edges)}] Edges on [{args.dataset.upper()}]: {end_time-start_time:.2f} Seconds.")

    return topoinf_all_e