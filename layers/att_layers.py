"""Attention layers (some modules are copied from https://github.com/Diego999/pyGAT."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, activation):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.act = activation

    def forward(self, input, adj):
        N = input.size()[0]
        edge = adj.coalesce().indices()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        ones = torch.ones(size=(N, 1))
        if h.is_cuda:
            ones = ones.cuda()
        e_rowsum = torch.sparse.mm(torch.sparse_coo_tensor(edge, edge_e), ones)
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = torch.sparse.mm(torch.sparse_coo_tensor(edge, edge_e), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        return self.act(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, activation, alpha, nheads, concat):
        """Sparse version of GAT."""
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.output_dim = output_dim
        self.attentions = [SpGraphAttentionLayer(input_dim,
                                                 output_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 activation=activation) for _ in range(nheads)]
        self.concat = concat
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, input):
        x, adj = input
        x = F.dropout(x, self.dropout, training=self.training)
        if self.concat:
            h = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        else:
            h_cat = torch.cat([att(x, adj).view((-1, self.output_dim, 1)) for att in self.attentions], dim=2)
            h = torch.mean(h_cat, dim=2)
        h = F.dropout(h, self.dropout, training=self.training)
        return (h, adj)
