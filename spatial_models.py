import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch.nn import BatchNorm1d as BN

from torch_geometric.nn import GCN2Conv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GINConv

from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_sparse import SparseTensor


class GIN(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(data.num_features, args.hidden),
                ReLU(),
                # BN(args.hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(args.num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(args.hidden, args.hidden),
                        ReLU(),
                        # BN(args.hidden),
                    ), train_eps=True))
        self.lin1 = Linear(args.hidden, args.hidden)
        self.lin2 = Linear(args.hidden, data.num_classes)

        self.dropout = args.dropout

        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=-1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.conv1 = SAGEConv(data.num_features, args.hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(args.num_layers - 1):
            self.convs.append(SAGEConv(args.hidden, args.hidden))
        self.lin1 = Linear(args.hidden, args.hidden)
        self.lin2 = Linear(args.hidden, data.num_classes)

        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=-1)


def get_normalized_adj_with_renormalization(edge_index, num_nodes):
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], 
                    sparse_sizes=(num_nodes, num_nodes))
    adj = adj.set_diag()    # Add diagonal entries
    adj_t = adj.t()         # Transpose
    adj_t = gcn_norm(adj_t) # GCN normalization

    return adj_t


class GCNII(torch.nn.Module):
    def __init__(self, data, args, hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5,
                 shared_weights=True, dropout=0.6):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(data.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, data.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)
        adj_t = get_normalized_adj_with_renormalization(edge_index, num_nodes)

        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


