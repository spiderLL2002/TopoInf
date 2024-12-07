import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import APPNP
from torch_geometric.utils import remove_self_loops, add_self_loops, degree


def get_gnn_model(gnn_name):
    gnn_name = gnn_name.upper()
    
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'SGC':
        Net = SGC_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name =='MLP':
        Net = MLP
    else:   
        from spectral_models import ChebNet, GPRGNN, BernNet, TAGCN    # spectral_models
        if gnn_name == 'CHEBNET':   # ChebNet
            Net = ChebNet
        elif gnn_name == 'GPRGNN':
            Net = GPRGNN
        elif gnn_name == 'BERNNET':     # BernNet
            Net = BernNet
        elif gnn_name == 'TAGCN':     # BernNet
            Net = TAGCN
        else:
            from spatial_models import GraphSAGE, GIN, GCNII    # spatial_models
            if gnn_name == 'GRAPHSAGE':
                Net = GraphSAGE
            elif gnn_name == 'GIN':
                Net = GIN
            elif gnn_name == 'GCNII':
                Net = GCNII
            else:
                raise ValueError(f'model {gnn_name} not supported in models')

    return Net


class GCN_Net(torch.nn.Module):
    def __init__(self, data, args):
        super(GCN_Net, self).__init__()

        self.num_layers = args.num_layers
        self.conv_list = torch.nn.ModuleList([])
        self.conv_list.append(GCNConv(data.num_features, args.hidden))
        for _ in range(self.num_layers - 2):
            self.conv_list.append(GCNConv(args.hidden, args.hidden))
        self.conv_list.append(GCNConv(args.hidden, data.num_classes))
        
        self.dropout = args.dropout
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.conv_list:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)    # NOTE: there is a dropout layer.
        for i in range(self.num_layers - 1):
            x = self.conv_list[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_list[-1](x, edge_index)
        
        return F.log_softmax(x, dim=1)


class GAT_Net(torch.nn.Module):
    def __init__(self, data, args):
        super(GAT_Net, self).__init__()
        self.num_layers = args.num_layers
        self.conv_list = torch.nn.ModuleList([])
        self.conv_list.append(GATConv(
            data.num_features,
            args.hidden,
            heads=args.heads_gat,
            dropout=args.dropout))
        for _ in range(self.num_layers - 2):
            self.conv_list.append(GATConv(
                args.hidden * args.heads_gat,
                args.hidden,
                heads=args.heads_gat,
                dropout=args.dropout))
        self.conv_list.append(GATConv(
            args.hidden * args.heads_gat,
            data.num_classes,
            heads=args.output_heads_gat,
            concat=False,
            dropout=args.dropout))
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.conv_list)):
            self.conv_list[i].reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(len(self.conv_list) - 1):
            x = F.elu(self.conv_list[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_list[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNP_Net(torch.nn.Module):
    def __init__(self, data, args):
        super(APPNP_Net, self).__init__()
        self.num_layers = args.num_layers
        self.alpha = args.alpha_appnp
        self.num_lins = args.num_layers
        self.lins = torch.nn.ModuleList([Linear(data.num_features, args.hidden)] + [Linear(args.hidden, args.hidden) for _ in range(self.num_lins-2)] + [Linear(args.hidden, data.num_classes)])
        self.prop1 = APPNP(self.num_layers, self.alpha)
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_lins):
            self.lins[i].reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_lins): 
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[i](x)
            if i != self.num_lins - 1:
                x = F.relu(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)



def get_normalized_adj_with_renormalization(edge_index, num_nodes):
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    src, dst = edge_index
    deg = degree(dst, num_nodes=num_nodes)

    deg_src = deg[src].pow(-0.5)
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-0.5)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a_hat = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([num_nodes, num_nodes])).t()

    return a_hat


class SGC_Net(torch.nn.Module): 
    def __init__(self, data, args):
        super(SGC_Net, self).__init__()
        self.num_layers = args.num_layers
        self.num_lins = args.num_layers
        self.lins = torch.nn.ModuleList([Linear(data.num_features, args.hidden)] + [Linear(args.hidden, args.hidden) for _ in range(self.num_lins-2)] + [Linear(args.hidden, data.num_classes)])
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_lins):
            self.lins[i].reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)
        a_hat = get_normalized_adj_with_renormalization(edge_index, num_nodes)
        
        for _ in range(self.num_layers):
            
            a_hat = a_hat.to(x.device)
            x = a_hat @ x

        for i in range(self.num_lins): 
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[i](x)
            if i != self.num_lins - 1:
                x = F.relu(x)
        return F.log_softmax(x, dim=1)



class MLP(torch.nn.Module):
    def __init__(self, data, args):
        super(MLP, self).__init__()
        self.num_layers = args.num_layers
        conv_list = [Linear(data.num_features, args.hidden)]
        conv_list = conv_list + [Linear(args.hidden, args.hidden) for i in range(self.num_layers-2)] + [Linear(args.hidden, data.num_classes)]
        self.conv_list = torch.nn.ModuleList(conv_list)
        self.dropout = args.dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.conv_list:
            conv.reset_parameters()

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers-1):
            x = F.relu(self.conv_list[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_list[-1](x)

        return F.log_softmax(x, dim=1)


