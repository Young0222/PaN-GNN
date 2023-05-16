
import torch
from torch import nn
from torch_geometric.data import Data
from convols import LightGConv
import torch.nn.functional as F
import sys

class PNeGNN(nn.Module):
    def __init__(self,train,num_u,num_v,offset, num_layers=2, MLP_layers=2, dim=64, reg=1e-4,
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(PNeGNN,self).__init__()
        self.M = num_u
        self.N = num_v
        self.num_layers = num_layers
        self.MLP_layers = MLP_layers
        self.device = device
        self.reg = reg
        self.embed_dim = dim

        # For the graph with positive edges (LightGCN)
        self.E_pos = nn.Parameter(torch.empty(self.M, dim))
        self.E_neg = nn.Parameter(torch.empty(self.M, dim))
        self.E_item = nn.Parameter(torch.empty(self.N, dim))
        self.E_item_n = nn.Parameter(torch.empty(self.N, dim))
        nn.init.xavier_normal_(self.E_pos.data)
        nn.init.xavier_normal_(self.E_neg.data)
        nn.init.xavier_normal_(self.E_item.data)
        nn.init.xavier_normal_(self.E_item_n.data)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(LightGConv())

        # For the graph with negative edges
        self.E2 = nn.Parameter(torch.empty(self.M + self.N, dim))
        nn.init.xavier_normal_(self.E2.data)
        self.mlps = nn.ModuleList()

        for _ in range(MLP_layers):
            self.mlps.append(nn.Linear(dim,dim,bias=True))
            nn.init.xavier_normal_(self.mlps[-1].weight.data)
        
        # Attention model
        self.attn = nn.Linear(dim,dim,bias=True)
        self.q = nn.Linear(dim,1,bias=False)
        self.attn_softmax = nn.Softmax(dim=1)


    def aggregate_u_two_embeddings(self, data_p, data_n, data_n_1):   # user has two embeddings

        # Generate embeddings z_p
        B=[]
        B.append(torch.cat([self.E_pos,self.E_item], dim=0))
        x = self.convs[0](torch.cat([self.E_pos,self.E_item], dim=0).to(self.device),data_p.edge_index.to(self.device))
        B.append(x)
        for i in range(1,self.num_layers):
            x = self.convs[i](x.to(self.device),data_p.edge_index.to(self.device))
            B.append(x)

        z_p = sum(B)/len(B) 

        # Generate embeddings z_ng
        C = []; C.append(self.E2)
        x = F.dropout(F.relu(self.mlps[0](self.E2)),p=0.5,training=self.training)
        for i in range(1,self.MLP_layers):
            x = self.mlps[i](x)
            x = F.relu(x)
            x = F.dropout(x,p=0.5,training=self.training)
            C.append(x)
        z_ng = C[-1]
        
        # Attntion for z_p
        w_p = self.q(F.dropout(torch.tanh((self.attn(z_p))),p=0.5,training=self.training))
        w_n = self.q(F.dropout(torch.tanh((self.attn(z_ng))),p=0.5,training=self.training))
        alpha_ = self.attn_softmax(torch.cat([w_p,w_n],dim=1))
        z_p = alpha_[:,0].view(len(z_p),1) * z_p + alpha_[:,1].view(len(z_p),1) * z_ng


        # Generate embeddings z_n
        B=[]
        B.append(torch.cat([self.E_neg,self.E_item_n], dim=0))
        x = self.convs[0](torch.cat([self.E_neg,self.E_item_n], dim=0).to(self.device),data_n.edge_index.to(self.device))
        B.append(x)
        for i in range(1,self.num_layers):
            x = self.convs[i](x.to(self.device),data_n.edge_index.to(self.device))
            B.append(x)

        z_n = sum(B)/len(B)


        # Generate embeddings z_n_1
        B=[]
        B.append(torch.cat([self.E_neg,self.E_item_n], dim=0))
        x = self.convs[0](torch.cat([self.E_neg,self.E_item_n], dim=0).to(self.device),data_n_1.edge_index.to(self.device))
        B.append(x)
        for i in range(1,self.num_layers):
            x = self.convs[i](x.to(self.device),data_n_1.edge_index.to(self.device))
            B.append(x)

        z_n_1 = sum(B)/len(B)

        return z_p, z_n, z_n_1


    def forward(self,u,v,w,n,data_p,data_n,data_n_1,device):
        emb_p, emb_n, emb_n_1 = self.aggregate_u_two_embeddings(data_p,data_n,data_n_1)
        u_1 = emb_p[u].to(device)  # 1024*64
        u_2 = emb_n[u].to(device)   # 1024*64
        u_3 = emb_n_1[u].to(device)   # 1024*64
        w_ = w.to(device)

        v_1 = emb_p[v].to(device)   # 1024*64
        n_1 = emb_p[n].to(device)    # 1024*1*64
        v_2 = emb_n[v].to(device)   # 1024*64
        n_2 = emb_n[n].to(device)    # 1024*1*64
        v_3 = emb_n_1[v].to(device)   # 1024*64

        # loss of the positive graph: (u_1, v_1, n_1)
        positivebatch = torch.mul(u_1 , v_1) #1024*64
        negativebatch = torch.mul(u_1.view(len(u_1),1,self.embed_dim), n_1)
        sBPR_loss_1 =  F.logsigmoid(  (-torch.sign(w_)+2).view(len(u_1),1) * positivebatch.sum(dim=1).view(len(u_1),1) - negativebatch.sum(dim=2)  ).sum(dim=1) # weight
        reg_loss_1 = u_1.norm(dim=1).pow(2).sum() + v_1.norm(dim=1).pow(2).sum() + n_1.norm(dim=2).pow(2).sum() 

        # loss of the positive graph: (u_2, v_2, n_2)
        positivebatch = torch.mul(u_2 , v_2) #1024*64
        negativebatch = torch.mul(u_2.view(len(u_2),1,self.embed_dim), n_2)
        sBPR_loss_2 =  F.logsigmoid(  negativebatch.sum(dim=2) - (torch.sign(w_)+2).view(len(u_2),1) * positivebatch.sum(dim=1).view(len(u_2),1)  ).sum(dim=1) # weight
        reg_loss_2 = u_2.norm(dim=1).pow(2).sum() + v_2.norm(dim=1).pow(2).sum() + n_2.norm(dim=2).pow(2).sum() 

        # contrastive learning loss: (u_2, v_2) and (u_3, v_3)
        u_2 = F.normalize(u_2)
        u_3 = F.normalize(u_3)
        v_2 = F.normalize(v_2)
        v_3 = F.normalize(v_3)
        sim_mat_u = torch.matmul(u_2, torch.transpose(u_3, dim0=1, dim1=0))
        sim_mat_v = torch.matmul(v_2, torch.transpose(v_3, dim0=1, dim1=0))

        tau = 0.8
        f = lambda x: torch.exp(x / tau)
        f_u = f(sim_mat_u)
        f_v = f(sim_mat_v)
        pos_pair = f_u.diag() + f_v.diag()
        neg_pair = f_u.sum() + f_v.sum() - f_u.diag() - f_v.diag()
        contrastive_loss = - torch.log( pos_pair / (pos_pair + neg_pair) )

        return (- torch.sum(sBPR_loss_1) + self.reg * reg_loss_1) + 1.0 * (- torch.mean(sBPR_loss_2) + self.reg * reg_loss_2) + 1.0 * torch.mean(contrastive_loss) 