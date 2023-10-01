import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from torch.utils.data import DataLoader
from torch import optim

from tqdm import tqdm
from evaluator import evaluator as ev
from util import *
from data_loader import Data_loader
from pandgnn import PandGNN
import argparse
import sys
import numpy as np
import random
import math
import io
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj, add_random_edge


def mean(lst):
    s = sum(lst)
    n = len(lst)
    return s/n


def sd(lst):
    mean = sum(lst)/len(lst)
    bias_mean = [(x - mean)**2 for x in lst]
    s2 = sum(bias_mean)/len(bias_mean)
    return math.sqrt(s2)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


def main(args):
    all_r = []
    all_p = []
    all_n = []
    all_h = []
    for all_i in range(1,6):
        print("Current index: ", all_i)
        data_class=Data_loader(args.dataset,args.version)
        print('data loading...');st=time.time()
        train,test = data_class.data_load()
        train = train.astype({'userId':'int64', 'movieId':'int64'})
        data_class.train = train; data_class.test = test
        print('loading complete! time :: %s'%(time.time()-st))
        
        
        print('generate negative candidates...'); st=time.time()
        if args.dataset == 'ML-1M':
            neg_dist = deg_dist(train,data_class.num_v)
        else:
            neg_dist = deg_dist_2(train,data_class.num_v)
        print('complete ! time : %s'%(time.time()-st))    
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model= PandGNN(train, data_class.num_u,data_class.num_v,offset=args.offset,num_layers = args.num_layers,MLP_layers=args.MLP_layers,dim=args.dim,device=device,reg=args.reg,aggregate=args.aggregate)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr = args.lr)
        
        
        print("\nTraining on {}...\n".format(device))
        model.train()
        print("data: ", train)
        user_pos = train[train['userId']==1][train['rating']>3.5]['movieId'].values-1
        print("user_pos: ", user_pos, "\n")

        training_dataset=bipartite_dataset(train,neg_dist,args.offset,data_class.num_u,data_class.num_v,args.K)
        
        res_r, res_p, res_n, res_h = 0,0,0,0

        # 构建正边邻接矩阵
        edge_user = torch.tensor(train[train['rating']>args.offset]['userId'].values-1)     # 从0开始编号！
        edge_item = torch.tensor(train[train['rating']>args.offset]['movieId'].values-1)+data_class.num_u
        edge_p = torch.stack((torch.cat((edge_user,edge_item),0),torch.cat((edge_item,edge_user),0)),0)

        # edge_rating = torch.zeros([data_class.num_u+data_class.num_v, data_class.num_u+data_class.num_v])

        # for index, row in train.iterrows():
        #     ind_user = row['userId'] - 1
        #     ind_item = row['movieId'] - 1 + data_class.num_u
        #     edge_rating[ind_user][ind_item] = row['rating']

        # print("edge_rating sum, max, min: ", torch.sum(edge_rating), torch.max(edge_rating), torch.min(edge_rating))
        # edge_rating = edge_rating.to(device)
        data_p=Data(edge_index=edge_p)
        data_p.to(device)

        # 构建负边邻接矩阵
        offset_n = 2.5
        edge_user_n = torch.tensor(train[train['rating']<offset_n]['userId'].values-1)
        edge_item_n = torch.tensor(train[train['rating']<offset_n]['movieId'].values-1)+data_class.num_u
        edge_n = torch.stack((torch.cat((edge_user_n,edge_item_n),0),torch.cat((edge_item_n,edge_user_n),0)),0)
        data_n=Data(edge_index=edge_n)
        data_n.to(device)

        # 正反馈: 随机增加边
        add_edge_rate = 0.5
        edge_p_1 = add_random_edge(edge_p, p=add_edge_rate)[0]
        data_p_1=Data(edge_index=edge_p_1)
        data_p_1.to(device)
        edge_p_2 = add_random_edge(edge_p, p=add_edge_rate)[0]
        data_p_2=Data(edge_index=edge_p_2)
        data_p_2.to(device)

        # 负反馈: 随机删除边
        drop_edge_rate = 0.5
        edge_n_1 = dropout_adj(edge_n, p=drop_edge_rate)[0]
        data_n_1 = Data(edge_index=edge_n_1)
        data_n_1.to(device)
        edge_n_2 = dropout_adj(edge_n, p=drop_edge_rate)[0]
        data_n_2 = Data(edge_index=edge_n_2)
        data_n_2.to(device)


        for EPOCH in range(1,args.epoch+1):
            if EPOCH%20-1==0:
                training_dataset.negs_gen_EP(20)
                
            LOSS=0
            training_dataset.edge_4 = training_dataset.edge_4_tot[:,:,EPOCH%20-1]

            ds = DataLoader(training_dataset,batch_size=args.batch_size,shuffle=True)
            q=0
            pbar = tqdm(desc = 'Version : {} Epoch {}/{}'.format(args.version,EPOCH,args.epoch),total=len(ds),position=0)

            for u,v,w,negs in ds:   
                q+=len(u)
                st=time.time()
                optimizer.zero_grad()
                loss = model(u,v,w,negs,data_p,data_n,data_p_1,data_p_2,data_n_1,data_n_2,device) # original
                loss.backward()                
                optimizer.step()
                LOSS+=loss.item() * len(ds)
                
                pbar.update(1)
                pbar.set_postfix({'loss':loss.item()})

            pbar.close()

            if EPOCH%20 ==1:

                model.eval()
                if args.aggregate == 'siren':
                    emb = model.aggregate()
                    emb_u, emb_v = torch.split(emb,[data_class.num_u,data_class.num_v])
                    emb_u = emb_u.cpu().detach()
                    emb_v = emb_v.cpu().detach()
                    r_hat = emb_u.mm(emb_v.t())
                    reco = gen_top_k(data_class,r_hat)
                else:
                    emb_p, emb_n, _, _, _, _  = model.aggregate_u_two_embeddings(data_p,data_n,data_p_1,data_p_2,data_n_1,data_n_2)
                    emb_u, emb_v = torch.split(emb_p,[data_class.num_u,data_class.num_v])
                    emb_n_u, emb_n_v = torch.split(emb_n,[data_class.num_u,data_class.num_v])
                    emb_u = emb_u.cpu().detach(); emb_v = emb_v.cpu().detach(); emb_n_u = emb_n_u.cpu().detach(); emb_n_v = emb_n_v.cpu().detach()
                    # np.save('emb_u.npy', emb_u)
                    # np.save('emb_v.npy', emb_v)
                    r_hat = emb_u.mm(emb_v.t())
                    r_hat_n = emb_n_u.mm(emb_n_v.t())
                    reco = gen_top_k_new3(data_class, r_hat, r_hat_n)
                

                eval_ = ev(data_class,reco,args)
                eval_.precision_and_recall()
                eval_.normalized_DCG()
                print("\n***************************************************************************************")
                print(" /* Recommendation Accuracy */")
                print('N :: %s'%(eval_.N))
                print('Precision at :: %s'%(eval_.N), eval_.p['total'][eval_.N-1])
                print('Recall at :: %s'%(eval_.N), eval_.r['total'][eval_.N-1])
                print('nDCG at :: %s'%(eval_.N), eval_.nDCG['total'][eval_.N-1])
                print('Hit at :: %s'%(eval_.N), eval_.h['total'][eval_.N-1])
                print('TP at :: %s'%(eval_.N), eval_.tp['total'][eval_.N-1])
                print('FP at :: %s'%(eval_.N), eval_.fp['total'][eval_.N-1])
                print('FN at :: %s'%(eval_.N), eval_.fn['total'][eval_.N-1])
                # print('neg mean: ', eval_.neg_mean)
                # print('neg >0.5 ratio: ', eval_.total_num_n, eval_.cal_num_1_n, eval_.cal_num_1_n/eval_.total_num_n)
                # print('neg >0.4 ratio: ', eval_.total_num_n, eval_.cal_num_2_n, eval_.cal_num_2_n/eval_.total_num_n)
                # print('neg >0.3 ratio: ', eval_.total_num_n, eval_.cal_num_3_n, eval_.cal_num_3_n/eval_.total_num_n)
                # print('neg >0.2 ratio: ', eval_.total_num_n, eval_.cal_num_4_n, eval_.cal_num_4_n/eval_.total_num_n)
                # print('neg >0.1 ratio: ', eval_.total_num_n, eval_.cal_num_5_n, eval_.cal_num_5_n/eval_.total_num_n)
                # print('neg >0.0 ratio: ', eval_.total_num_n, eval_.cal_num_6_n, eval_.cal_num_6_n/eval_.total_num_n)
                # print('pos mean: ', eval_.pos_mean)
                # print('pos >0.5 ratio: ', eval_.total_num_p, eval_.cal_num_1_p, eval_.cal_num_1_p/eval_.total_num_p)
                # print('pos >0.4 ratio: ', eval_.total_num_p, eval_.cal_num_2_p, eval_.cal_num_2_p/eval_.total_num_p)
                # print('pos >0.3 ratio: ', eval_.total_num_p, eval_.cal_num_3_p, eval_.cal_num_3_p/eval_.total_num_p)
                # print('pos >0.2 ratio: ', eval_.total_num_p, eval_.cal_num_4_p, eval_.cal_num_4_p/eval_.total_num_p)
                # print('pos >0.1 ratio: ', eval_.total_num_p, eval_.cal_num_5_p, eval_.cal_num_5_p/eval_.total_num_p)
                # print('pos >0.0 ratio: ', eval_.total_num_p, eval_.cal_num_6_p, eval_.cal_num_6_p/eval_.total_num_p)
                print("***************************************************************************************")
                if eval_.r['total'][eval_.N-1][2] > res_r:
                    res_r = eval_.r['total'][eval_.N-1][2]
                    res_p = eval_.p['total'][eval_.N-1][2]
                    res_n = eval_.nDCG['total'][eval_.N-1][2]
                    res_h = eval_.h['total'][eval_.N-1][2]
                model.train()

        if EPOCH == args.epoch:
            print("Final results (R,P,N,H) are: ", res_r, res_p, res_n, res_h)

        all_r.append(res_r)
        all_p.append(res_p)
        all_n.append(res_n)
        all_h.append(res_h)

    print("")
    print("mean: ", mean(all_r), mean(all_p), mean(all_n), mean(all_h))
    print("sd: ", sd(all_r), sd(all_p), sd(all_n), sd(all_h))



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type = str,
                        default = 'ML-1M',
                        help = "Dataset"
                        )
    parser.add_argument('--version',
                        type = int,
                        default =1,
                        help = "Dataset version"
                        )
    parser.add_argument('--batch_size',
                        type = int,
                        default = 1024,
                        help = "Batch size"
                        )

    parser.add_argument('--dim',
                        type = int,
                        default = 64,
                        help = "Dimension"
                        )
    parser.add_argument('--lr',
                        type = float,
                        default = 5e-3,
                        help = "Learning rate"
                        )
    parser.add_argument('--offset',
                        type = float,
                        default = 3.5,
                        help = "Criterion of likes/dislikes"
                        )
    parser.add_argument('--K',
                        type = int,
                        default = 40,
                        help = "The number of negative samples"
                        )
    parser.add_argument('--num_layers',
                        type = int,
                        default = 4,
                        help = "The number of layers of a GNN model for the graph with positive edges"
                        )
    parser.add_argument('--MLP_layers',
                        type = int,
                        default = 2,
                        help = "The number of layers of MLP for the graph with negative edges"
                        )
    parser.add_argument('--epoch',
                        type = int,
                        default = 1000,
                        help = "The number of epochs"
                        )
    parser.add_argument('--reg',
                        type = float,
                        default = 0.05,
                        help = "Regularization coefficient"
                        )
    parser.add_argument('--aggregate',
                        type = str,
                        default = 'pandgnn',
                        help = "aggregate method"
                        )
    args = parser.parse_args()
    main(args)