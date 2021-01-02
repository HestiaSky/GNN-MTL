import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from layers import GraphConvolution
from utils import glorot, truncated_normal
import scipy.spatial

class GCN(Module):
    # Graph Convolution Networks

    def __init__(self, in_features, out_features, dropout, act, sparse_feature=True, use_bias=False):
        super(GCN, self).__init__()
        self.layers = []
        self.layers.append(GraphConvolution(in_features = in_features,
                                       out_features = out_features,
                                       dropout = dropout,
                                       act = act,
                                       init = truncated_normal,
                                       sparse_feature= sparse_feature,
                                       use_bias=use_bias,
                                       transform=True
                                       ))
        self.layers.append(GraphConvolution(in_features = out_features,
                                       out_features = out_features,
                                       dropout = dropout,
                                       act = lambda x: x,
                                       init = glorot,
                                       sparse_feature= False,
                                       use_bias=use_bias,
                                       transform=False
                                       ))
        self.seq = nn.Sequential(*self.layers)
    
    def forward(self, x, adj):
        input = (x, adj)
        # check, _ = self.layers[0](input)
        output, _  = self.seq.forward(input)
        return output

class GCN_ALIGN(Module):
    # Entity Alignment with graph convolution networks

    def __init__(self, neg_k, in_features, out_features, dropout, act, sparse_feature=True, use_bias=False):
        super(GCN_ALIGN, self).__init__()
        self.gcn = GCN(in_features, out_features, dropout, act, sparse_feature, use_bias)
        self.embeddings = None
        self.neg_k = neg_k

    def sample_neg(self, train, e):
        """Sample negative paris for left->right and right->left resp."""
        k = self.neg_k
        t = len(train)
        L = np.ones((t, k)) * (train[:, 0].reshape((t, 1)))
        self.neg_left = L.reshape((t * k,))
        self.neg_right = np.random.choice(e, t * k)
        L = np.ones((t, k)) * (train[:, 1].reshape((t, 1)))
        self.neg2_left = np.random.choice(e, t * k)
        self.neg2_right = L.reshape((t * k,))

    def get_loss(self, x, adj, train, gamma):
        self.embeddings = self.gcn(x, adj)
        left = train[:, 0]
        right = train[:, 1]
        t = len(train)
        k = self.neg_k
        left_x = self.embeddings[left]
        right_x = self.embeddings[right]
        A = torch.sum(torch.abs(left_x - right_x), 1)
        neg_l_x = self.embeddings[self.neg_left]
        neg_r_x = self.embeddings[self.neg_right]
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
        C = - torch.reshape(B, [t, k])
        D = A + gamma
        L1 = F.relu(torch.add(C, torch.reshape(D, [t, 1])))
        neg_l_x = self.embeddings[self.neg2_left]
        neg_r_x = self.embeddings[self.neg2_right]
        B = torch.sum(torch.abs(neg_l_x - neg_r_x), 1)
        C = - torch.reshape(B, [t, k])
        L2 = F.relu(torch.add(C, torch.reshape(D, [t, 1])))
        return (torch.sum(L1) + torch.sum(L2)) / (2.0 * t * k)
        
    def eval_at_1(self, test):
        """Evaluation the embeddings on the Hits@1 metric"""
        if self.embeddings is None:
            return -1.0
        L = np.array([e1 for e1, e2 in test])
        R = np.array([e2 for e1, e2 in test])
        M = torch.cdist(self.embeddings[L], self.embeddings[R], p=1)
        cnt = torch.zeros(len(L))
        cnt[torch.argmin(M, dim=1) == torch.arange(0, len(L)).to(self.embeddings.device)] = 1
        return torch.sum(cnt) / len(cnt) * 100