
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import torch.nn.functional as F
from torch import nn
import torch_scatter
import sys
import torch
import random
from torch_geometric.utils import softmax


class LightGConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
        
    # def forward(self,x,edge_index):
    #     row, col = edge_index
    #     deg = degree(col, x.size(0), dtype=x.dtype)
    #     deg_inv_sqrt = deg.pow(-0.5)
    #     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    #     norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    #     out = self.propagate(edge_index, x=x, norm=norm)
    #     return out

    def forward(self, x, edge_index, norm):
        out = self.propagate(edge_index, x=x, norm=norm)
        return out
    
    def message(self,x_j,norm):
        return norm.view(-1,1) * x_j
    def update(self,inputs: Tensor) -> Tensor:
        return inputs


class LightGINConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
        self.eps = torch.nn.Parameter(torch.empty(1))
        self.eps.data.fill_(0.0)
        
    def forward(self,x,edge_index, norm, norm_self):
        # row, col = edge_index
        # deg = degree(col, x.size(0), dtype=x.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, norm=norm)
        # norm_self = deg_inv_sqrt[range(x.size(0))] * deg_inv_sqrt[range(x.size(0))]
        # norm_self = norm_self.unsqueeze(dim=1).repeat(1,x.size(1))
        out = out + (1 + self.eps) * norm_self * x  # GIN
        return out
    
    def message(self,x_j,norm):
        return norm.view(-1,1) * x_j
    def update(self,inputs: Tensor) -> Tensor:
        return inputs


# class GINConv(MessagePassing): # from torch_geometric
#     r"""The graph isomorphism operator from the `"How Powerful are
#     Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper"""
#     def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
#                  **kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(**kwargs)
#         self.nn = nn
#         self.initial_eps = eps
#         if train_eps:
#             self.eps = torch.nn.Parameter(torch.empty(1))
#         else:
#             self.register_buffer('eps', torch.empty(1))
#         self.reset_parameters()

#     def reset_parameters(self):
#         super().reset_parameters()
#         reset(self.nn)
#         self.eps.data.fill_(self.initial_eps)


#     def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
#                 size: Size = None) -> Tensor:

#         if isinstance(x, Tensor):
#             x: OptPairTensor = (x, x)

#         # propagate_type: (x: OptPairTensor)
#         out = self.propagate(edge_index, x=x, size=size)

#         x_r = x[1]
#         if x_r is not None:
#             out = out + (1 + self.eps) * x_r

#         return self.nn(out)


#     def message(self, x_j: Tensor) -> Tensor:
#         return x_j

#     def message_and_aggregate(self, adj_t: SparseTensor,
#                               x: OptPairTensor) -> Tensor:
#         if isinstance(adj_t, SparseTensor):
#             adj_t = adj_t.set_value(None, layout=None)
#         return spmm(adj_t, x[0], reduce=self.aggr)

#     def __repr__(self) -> str:
#         return f'{self.__class__.__name__}(nn={self.nn})'


class GAT(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, negative_slope=0.2, dropout=0., **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = nn.Linear(in_features=self.in_channels, out_features=self.out_channels * self.heads, bias=False)
        self.lin_r = self.lin_l
        self.att_l = nn.Parameter(Tensor(1, self.heads, self.out_channels))
        self.att_r = nn.Parameter(Tensor(1, self.heads, self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size=None):
        H, C = self.heads, self.out_channels
        out = None
        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)
        alpha_l = x_l * self.att_l
        alpha_r = x_r * self.att_r
        out = self.propagate(edge_index,size=size,x=(x_l, x_r),alpha=(alpha_l, alpha_r))

        # concat
        out = out.view(-1, H * C)
        # mean
        # out = out.mean(dim=1)

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout)
        out = x_j * alpha

        return out

    def aggregate(self, inputs, index, dim_size=None):
        node_dim = self.node_dim
        out = torch_scatter.scatter(src=inputs,index=index,dim=node_dim,dim_size=dim_size,reduce='sum')
        return out


class LightGAT(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, negative_slope=0.2, dropout=0., **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.att_l = nn.Parameter(Tensor(1, self.heads, self.out_channels))
        self.att_r = nn.Parameter(Tensor(1, self.heads, self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size=None):
        H, C = self.heads, self.out_channels
        out = None
        x_l = x.view(-1, H, C)
        x_r = x.view(-1, H, C)
        alpha_l = x_l * self.att_l
        alpha_r = x_r * self.att_r
        out = self.propagate(edge_index,size=size,x=(x_l, x_r),alpha=(alpha_l, alpha_r))

        # concat
        out = out.view(-1, H * C)
        # mean
        # out = out.mean(dim=1)

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout)
        out = x_j * alpha

        return out

    def aggregate(self, inputs, index, dim_size=None):
        node_dim = self.node_dim
        out = torch_scatter.scatter(src=inputs,index=index,dim=node_dim,dim_size=dim_size,reduce='sum')
        return out