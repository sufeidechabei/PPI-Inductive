"""
Graph Attention Networks
Paper: https://arxiv.org/abs/1710.10903
Code: https://github.com/PetarV-/GAT
GAT with batch processing
"""

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.init as init


class GraphAttention(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 out_dim,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 alpha,
                 residual=False,):
        super(GraphAttention, self).__init__()
        self.g = g
        self.num_heads = num_heads
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = None
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.attn_l = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        self.attn_r = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = None
        # As mentioned in https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/
        # ops/variable_scope.py#L292, the default initializer is `glorot_uniform_initializer`.
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.residual_fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
            else:
                self.residual_fc = None
        self._init()

    def _init(self):
        init.xavier_uniform_(self.fc.weight.data, gain=1.414)
        init.xavier_uniform_(self.attn_l.data, gain=1.414)
        init.xavier_uniform_(self.attn_r.data, gain=1.414)
        if self.residual and self.residual_fc is not None:
            init.xavier_uniform_(self.residual_fc.weight.data, gain=1.414)

    def forward(self, inputs):
        # prepare, inputs are of shape V x F, V the number of nodes, F the size of input features
        h = inputs
        if self.feat_drop:
            h = self.feat_drop(h)
        # V x K x F', K number of heads, F' size of transformed features
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))
        head_ft = ft.transpose(0, 1)                                  # K x V x F'
        a1 = torch.bmm(head_ft, self.attn_l).transpose(0, 1)          # V x K x 1
        a2 = torch.bmm(head_ft, self.attn_r).transpose(0, 1)          # V x K x 1
        if self.feat_drop:
            ft = self.feat_drop(ft)
        self.g.set_n_repr({'ft' : ft, 'a1' : a1, 'a2' : a2})

        # 1. compute softmax without normalization for edge attention
        self.compute_edge_attention()
        # 2. compute two results, one is the node features scaled by the dropped,
        # unnormalized attention values. Another is the normalizer of the attention values.
        self.g.update_all([fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.copy_edge('a', 'a')],
                          [fn.sum('ft', 'ft'), fn.sum('a', 'z')])
        # 3. apply normalizer
        ret = self.g.ndata.pop('ft') / self.g.ndata['z']
        # 4. residual
        if self.residual:
            # Note that a broadcasting addition will be employed.
            if self.residual_fc:
                resval = self.residual_fc(h).reshape((h.shape[0], self.num_heads, -1))
            else:
                resval = h.unsqueeze(1)
            ret = resval + ret
        return ret

    def compute_edge_attention(self):
        # 1. compute edge attention logits
        self.g.apply_edges(self.edge_attn_logits)
        # 2. fetch max logits from dests
        self.g.update_all(fn.copy_edge('a', 'a'), self.fetch_max_logits)
        # 3. normalize edge attention logits
        self.g.apply_edges(self.edge_attn_exp)

    def edge_attn_logits(self, edges):
        a = self.leaky_relu(edges.src['a1'] + edges.dst['a2'])
        return {'a': a}

    def fetch_max_logits(self, nodes):
        return {'max_attn_logits': nodes.mailbox['a'].max(1)[0]}

    def edge_attn_exp(self, edges):
        a = torch.exp(edges.data['a'] - edges.dst['max_attn_logits'])
        if self.attn_drop:
            a_drop = self.attn_drop(a)
            return {'a' : a, 'a_drop' : a_drop}
        return {'a': a, 'a_drop': a}

class GAT(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 hidden_dim,
                 n_classes,
                 num_heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 alpha,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        n_layers = len(num_heads) - 1
        self.num_layers = n_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GraphAttention(
            g, in_dim, hidden_dim, num_heads[0], feat_drop, attn_drop, alpha, False))
        # hidden layers
        for l in range(1, n_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GraphAttention(
                g, hidden_dim * num_heads[l-1], hidden_dim, num_heads[l],
                feat_drop, attn_drop, alpha, residual))
        # output projection
        self.gat_layers.append(GraphAttention(
            g, hidden_dim * num_heads[-2], n_classes, num_heads[-1],
            feat_drop, attn_drop, alpha, residual))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](h).flatten(1)
            h = self.activation(h)
        # output projection
        logits = self.gat_layers[-1](h).mean(1)
        return logits
