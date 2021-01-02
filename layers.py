import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class SparseDropout(torch.nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob=1-dprob

    def forward(self, x):
        mask=((torch.rand(x._values().size())+(self.kprob)).floor()).type(torch.bool)
        rc=x._indices()[:,mask]
        val=x._values()[mask]*(1.0/self.kprob)
        return torch.sparse.FloatTensor(rc, val)

class GraphConvolution(Module):
    # Simple GCN layer.
    def __init__(self, in_features, out_features, dropout, act, init, sparse_feature=True, use_bias=False, transform=True):
        super(GraphConvolution, self).__init__()
        self.W = Parameter(init((in_features, out_features)))
        self.act = act
        self.in_features = in_features
        self.out_features = out_features
        self.sparse_feature = sparse_feature
        self.use_bias = use_bias
        self.transform = transform
        if self.sparse_feature:
            self.sparse_dropout = SparseDropout(dropout)
        else:
            self.dropout = dropout
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(1, out_features))

    def forward(self, input):
        x, adj = input
        if self.sparse_feature:
            x = self.sparse_dropout(x)
        else:
            x = F.dropout(x, self.dropout)
        if self.transform:
            if self.sparse_feature:
                hidden = torch.sparse.mm(x, self.W)
            else:
                hidden = torch.mm(x, self.W)
        else:
            hidden = x
        if adj.is_sparse:
            support = torch.sparse.mm(adj, hidden)
        else:
            support = torch.mm(adj, hidden)
        if self.use_bias:
            support += self.bias
        output = self.act(support), adj
        return output