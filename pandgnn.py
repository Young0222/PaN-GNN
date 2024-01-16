import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

import torch
from torch import nn
from torch_geometric.data import Data
from convols import LightGConv, LightGINConv
import torch.nn.functional as F
import sys
import random
import numpy as np
from torch_geometric.utils import degree
from time import perf_counter as t


class PandGNN(nn.Module):
    def __init__(self,train,num_u,num_v,offset,num_layers = 2,MLP_layers=2,dim = 64,reg=1e-4,aggregate='siren',
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(PandGNN,self).__init__()
        self.M = num_u
        self.N = num_v
        self.num_layers = num_layers
        self.MLP_layers = MLP_layers
        self.device = device
        self.reg = reg
        self.embed_dim = dim
        self.agg = aggregate

        # For the graph with positive edges
        self.E_pos = nn.Parameter(torch.empty(self.M, dim))
        self.E_neg = nn.Parameter(torch.empty(self.M, dim))
        self.E_item = nn.Parameter(torch.empty(self.N, dim))
        self.E_item_n = nn.Parameter(torch.empty(self.N, dim))
        nn.init.xavier_normal_(self.E_pos.data)
        nn.init.xavier_normal_(self.E_neg.data)
        nn.init.xavier_normal_(self.E_item.data)
        nn.init.xavier_normal_(self.E_item_n.data)

        self.attn = nn.Linear(dim,dim,bias=True)
        self.q = nn.Linear(dim,1,bias=False)
        self.attn_softmax = nn.Softmax(dim=1)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # self.convs.append(LightGConv())
            self.convs.append(LightGINConv())


    def generate_embedding_multi(self, data, E_pos, E_item):
        # Generate embeddings z_p
        B=[]
        B.append(torch.cat([E_pos, E_item], dim=0))
        x = torch.cat([E_pos, E_item], dim=0).to(self.device)
        edge_index = data.edge_index.to(self.device)

        # Calculte norm & norm_self
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        norm_self = deg_inv_sqrt[range(x.size(0))] * deg_inv_sqrt[range(x.size(0))]
        norm_self = norm_self.unsqueeze(dim=1).repeat(1,x.size(1))

        x = self.convs[0](x, edge_index, norm, norm_self)
        B.append(x)
        for i in range(1,self.num_layers):
            x = self.convs[i](x.to(self.device), edge_index, norm, norm_self)
            B.append(x)
        z = sum(B)/len(B)

        return z


    def generate_embedding_one(self, data, E_pos, E_item):
        # Generate embeddings z_p
        B=[]
        B.append(torch.cat([E_pos, E_item], dim=0))
        x = torch.cat([E_pos, E_item], dim=0).to(self.device)
        edge_index = data.edge_index.to(self.device)

        # Calculte norm & norm_self
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.convs[0](x, edge_index, norm)
        B.append(x)
        for i in range(1,self.num_layers):
            x = self.convs[i](x.to(self.device), edge_index, norm)
            B.append(x)
        z = sum(B)/len(B)

        return z


    def aggregate_u_two_embeddings(self, data_p, data_n, data_p_1, data_p_2, data_n_1, data_n_2):   # user has two embeddings

        # Generate six embeddings
        z_p = self.generate_embedding_multi(data_p, self.E_pos,self.E_item)
        z_n = self.generate_embedding_multi(data_n, self.E_neg,self.E_item_n)
        z_p_1 = self.generate_embedding_multi(data_p_1, self.E_pos,self.E_item)
        z_p_2 = self.generate_embedding_multi(data_p_2, self.E_pos,self.E_item)
        z_n_1 = self.generate_embedding_multi(data_n_1, self.E_neg,self.E_item_n)
        z_n_2 = self.generate_embedding_multi(data_n_2, self.E_neg,self.E_item_n)

        # z_p = self.generate_embedding_one(data_p, self.E_pos,self.E_item)
        # z_n = self.generate_embedding_one(data_n, self.E_neg,self.E_item_n)
        # z_p_1 = self.generate_embedding_one(data_p_1, self.E_pos,self.E_item)
        # z_p_2 = self.generate_embedding_one(data_p_2, self.E_pos,self.E_item)
        # z_n_1 = self.generate_embedding_one(data_n_1, self.E_neg,self.E_item_n)
        # z_n_2 = self.generate_embedding_one(data_n_2, self.E_neg,self.E_item_n)

        return z_p, z_n, z_p_1, z_p_2, z_n_1, z_n_2


    def aggregate_simple(self,data_p):   # only positive embeddings

        # Generate one embedding
        z_p = self.generate_embedding_multi(data_p, self.E_pos,self.E_item)
        # z_p = self.generate_embedding_one(data_p, self.E_pos,self.E_item)

        return z_p


    def forward(self,u,v,w,n,data_p,data_n,data_p_1,data_p_2,data_n_1,data_n_2,EPOCH,device):   # user has two embeddings, Recall@10=0.1990

        if EPOCH % 10 != 1:
            emb_p = self.aggregate_simple(data_p)
            w_ = w.to(device)
            u_1 = emb_p[u].to(device)  # 1024*64
            v_1 = emb_p[v].to(device)   # 1024*64
            n_1 = emb_p[n].to(device)    # 1024*40*64
            positivebatch = torch.mul(u_1 , v_1) #1024*64
            negativebatch = torch.mul(u_1.view(len(u_1),1,self.embed_dim), n_1)
            sBPR_loss_1 =  F.logsigmoid(  (-torch.sign(w_)+2).view(len(u_1),1) * positivebatch.sum(dim=1).view(len(u_1),1) - negativebatch.sum(dim=2)  ).sum(dim=1) # weight
            sBPR_loss_1 = torch.sum(sBPR_loss_1)
            reg_loss_1 = u_1.norm(dim=1).pow(2).sum() + v_1.norm(dim=1).pow(2).sum() + n_1.norm(dim=2).pow(2).sum() 
            

            return -sBPR_loss_1 + self.reg * reg_loss_1

        else:

            emb_p, emb_n, emb_p_1, emb_p_2, emb_n_1, emb_n_2 = self.aggregate_u_two_embeddings(data_p,data_n,data_p_1,data_p_2,data_n_1,data_n_2)
            w_ = w.to(device)
            u_1 = emb_p[u].to(device)  # 1024*64
            v_1 = emb_p[v].to(device)   # 1024*64
            n_1 = emb_p[n].to(device)    # 1024*40*64
            u_2 = emb_n[u].to(device)   # 1024*64
            v_2 = emb_n[v].to(device)   # 1024*64
            n_2 = emb_n[n].to(device)    # 1024*40*64
            u_3 = emb_p_1[u].to(device)   # 1024*64
            v_3 = emb_p_1[v].to(device)   # 1024*64
            n_3 = emb_p_1[n].to(device)    # 1024*40*64
            u_4 = emb_p_2[u].to(device)   # 1024*64
            v_4 = emb_p_2[v].to(device)   # 1024*64
            n_4 = emb_p_2[n].to(device)    # 1024*40*64
            u_5 = emb_n_1[u].to(device)   # 1024*64
            v_5 = emb_n_1[v].to(device)   # 1024*64
            n_5 = emb_n_1[n].to(device)    # 1024*40*64
            u_6 = emb_n_2[u].to(device)   # 1024*64
            v_6 = emb_n_2[v].to(device)   # 1024*64
            n_6 = emb_n_2[n].to(device)    # 1024*40*64

            # 正反馈网络的loss: (u_1, v_1, n_1)
            positivebatch = torch.mul(u_1 , v_1) #1024*64
            negativebatch = torch.mul(u_1.view(len(u_1),1,self.embed_dim), n_1)
            sBPR_loss_1 =  F.logsigmoid(  (-torch.sign(w_)+2).view(len(u_1),1) * positivebatch.sum(dim=1).view(len(u_1),1) - negativebatch.sum(dim=2)  ).sum(dim=1) # weight
            sBPR_loss_1 = torch.sum(sBPR_loss_1)
            reg_loss_1 = u_1.norm(dim=1).pow(2).sum() + v_1.norm(dim=1).pow(2).sum() + n_1.norm(dim=2).pow(2).sum() 

            # 负反馈网络的loss: (u_2, v_2, n_2)
            positivebatch = torch.mul(u_2 , v_2) #1024*64
            negativebatch = torch.mul(u_2.view(len(u_2),1,self.embed_dim), n_2)
            sBPR_loss_2 =  F.logsigmoid(  negativebatch.sum(dim=2) - (torch.sign(w_)+2).view(len(u_2),1) * positivebatch.sum(dim=1).view(len(u_2),1)  ).sum(dim=1) # weight
            sBPR_loss_2 = torch.sum(sBPR_loss_2)
            reg_loss_2 = u_2.norm(dim=1).pow(2).sum() + v_2.norm(dim=1).pow(2).sum() + n_2.norm(dim=2).pow(2).sum() 

            # 正反馈上对比学习的loss: (u_3, v_3) and (u_4, v_4)
            tau = 0.8
            f = lambda x: torch.exp(x / tau)
            u_3 = F.normalize(u_3)
            v_3 = F.normalize(v_3)
            n_3 = F.normalize(n_3)[:,0,:]
            u_4 = F.normalize(u_4)
            v_4 = F.normalize(v_4)
            n_4 = F.normalize(n_4)[:,0,:]

            sim_mat_pos_u = torch.matmul(u_3, torch.transpose(u_4, dim0=1, dim1=0))
            f_u = f(sim_mat_pos_u)
            pos_pair = f_u.diag()
            sim_mat_neg_n_3 = torch.matmul(u_3, torch.transpose(n_3, dim0=1, dim1=0))
            sim_mat_neg_n_4 = torch.matmul(u_4, torch.transpose(n_4, dim0=1, dim1=0))
            neg_pair = f(sim_mat_neg_n_3).diag() + f(sim_mat_neg_n_4).diag()
            pos_contrastive_loss = - torch.log( pos_pair / (pos_pair + neg_pair) )
            pos_contrastive_loss = torch.sum(pos_contrastive_loss)

            # 负反馈上对比学习的loss: (u_5, v_5) and (u_6, v_6)
            u_5 = F.normalize(u_5)
            v_5 = F.normalize(v_5)
            n_5 = F.normalize(n_5)[:,0,:]
            u_6 = F.normalize(u_6)
            v_6 = F.normalize(v_6)
            n_6 = F.normalize(n_6)[:,0,:]

            sim_mat_neg_u = torch.matmul(u_5, torch.transpose(u_6, dim0=1, dim1=0))       
            f_u = f(sim_mat_neg_u)
            pos_pair = f_u.diag()
            sim_mat_neg_u_5 = torch.matmul(u_5, torch.transpose(n_5, dim0=1, dim1=0))
            sim_mat_neg_u_6 = torch.matmul(u_6, torch.transpose(n_6, dim0=1, dim1=0))
            neg_pair = f(sim_mat_neg_u_5).diag() + f(sim_mat_neg_u_6).diag()
            neg_contrastive_loss = - torch.log( pos_pair / (pos_pair + neg_pair) )
            neg_contrastive_loss = torch.sum(neg_contrastive_loss)

            return (-sBPR_loss_1 + self.reg * reg_loss_1) + (-sBPR_loss_2 + self.reg * reg_loss_2) + pos_contrastive_loss + neg_contrastive_loss
