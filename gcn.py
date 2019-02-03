"""
Semi-Supervised Classification with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1609.02907
Code: https://github.com/tkipf/gcn
GCN with SPMV specialization.
"""
import torch
import torch.nn as nn
import dgl.function as fn


class GCNLayer(nn.Module):
    """GCNLayer adapted from the released official implementation
    https://github.com/tkipf/gcn/blob/master/gcn/layers.py, with
    modifications for the use of dgl.

    We also modify the code so that we can perform a mutli-head
    graph convolution with the update rule:

    ReLU((1/K)\sum_k AXW_k), where K is the number of heads and
    k indexes heads.
    """
    def __init__(self,
                 g,
                 in_dim,
                 out_dim,
                 activation=nn.ReLU(),
                 dropout=0.,
                 bias=True,
                 num_heads=1):
        super(GCNLayer, self).__init__()
        self.g = g
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.

        self.num_heads = num_heads
        self.weights = nn.Parameter(torch.Tensor(num_heads, in_dim, out_dim))
        self._init()
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

    def _init(self):
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, h):
        if self.dropout:
            h = self.dropout(h)
        # Average over multi heads
        h = torch.bmm(h.expand(self.num_heads, *h.size()), self.weights).mean(0)
        # normalization by square root of src degree
        self.g.ndata['h'] = h
        self.g.update_all(fn.copy_src(src='h', out='m'),
                          fn.sum(msg='m', out='h'))
        h = self.g.ndata.pop('h')
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return h

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 hidden_dim,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 bias,
                 num_heads):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(
            g, in_dim, hidden_dim, activation, dropout,
                                    bias, num_heads))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(g, hidden_dim, hidden_dim, activation, dropout,
                                        bias, num_heads))
        # output layer
        self.layers.append(GCNLayer(g, hidden_dim, n_classes, None, dropout,
                                    bias, num_heads))

    def forward(self, features):
        h = features
        for layer in self.layers:
            h = layer(h)
        return h

def test_multihead_gcn(dataset='cora'):
    from train import train
    from utils import load_data

    data = load_data(dataset)
    # Create model
    # You can change the num_heads below for testing.
    train(dataset=(data.graph, data.features, data.labels, data.train_mask,
                   data.val_mask, data.test_mask, data.num_labels),
          args={
              'n_epochs': 300,
              'lr': 1e-2,
              'weight_decay': 5e-4,
              'model': 'gcn',
              'hidden_dim': 16,
              'n_layers': 1,
              'activation': nn.ReLU(),
              'dropout': 0.5,
              'bias': True,
              'num_heads': 1,
              'device': 'cpu'})

if __name__ == '__main__':
    test_multihead_gcn('pubmed')