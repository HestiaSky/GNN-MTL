"""Graph decoders."""
import torch.nn as nn
import torch.nn.functional as F

from layers.att_layers import GraphAttentionLayer
from layers.layers import GraphConvolution, Linear


class Decoder(nn.Module):
    # Decoder abstract class for node classification tasks.

    def __init__(self):
        super(Decoder, self).__init__()

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class GCNDecoder(Decoder):
    # Graph Convolution Decoder.

    def __init__(self, args):
        super(GCNDecoder, self).__init__()
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True


class GATDecoder(Decoder):
    # Graph Attention Decoder.

    def __init__(self, args):
        super(GATDecoder, self).__init__()
        self.cls = GraphAttentionLayer(args.dim, args.n_classes, args.dropout, F.elu, args.alpha, 1, True)
        self.decode_adj = True


class MLPDecoder(Decoder):
    # Multi-layer perceptron Decoder.

    def __init__(self, args):
        super(MLPDecoder, self).__init__()
        dims = [args.dim, args.dim, args.dim, args.n_classes]
        acts = ['relu'] * 3
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            layers.append(Linear(in_dim, out_dim, args.dropout, act, args.bias))
        self.cls = nn.Sequential(*layers)
        self.decode_adj = False


model2decoder = {
    'GCN': GCNDecoder,
    'GAT': MLPDecoder,
    'HGCN': GCNDecoder,
}