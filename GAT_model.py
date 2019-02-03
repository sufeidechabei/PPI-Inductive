

import torch
import torch.nn as nn
import dgl.function as fn


class GraphAttention(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 out_dim,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 alpha,
                 residual=False):
        super(GraphAttention, self).__init__()
        self.g = g
        self.num_heads = num_heads
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = None
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = None
        self.attn_l = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        self.attn_r = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_r.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
                nn.init.xavier_normal_(self.res_fc.weight.data, gain=1.414)
            else:
                self.res_fc = None

    def forward(self, inputs):
        # prepare
        h = inputs  # NxD
        if self.feat_drop:
            h = self.feat_drop(h)
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
        head_ft = ft.transpose(0, 1)  # HxNxD'
        a1 = torch.bmm(head_ft, self.attn_l).transpose(0, 1)  # NxHx1
        a2 = torch.bmm(head_ft, self.attn_r).transpose(0, 1)  # NxHx1
        if self.feat_drop:
            ft = self.feat_drop(ft)
        self.g.ndata.update({'ft' : ft, 'a1' : a1, 'a2' : a2})
        # 1. compute edge attention
        self.g.apply_edges(self.edge_attention)
        # 2. compute two results: one is the node features scaled by the dropped,
        # unnormalized attention values; another is the normalizer of the attention values.
        self.g.update_all([fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.copy_edge('a', 'a')],
                          [fn.sum('ft', 'ft'), fn.sum('a', 'z')])
        # 3. apply normalizer
        ret = self.g.ndata['ft'] / self.g.ndata['z']  # NxHxD'
        # 4. residual
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(h).reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
            else:
                resval = torch.unsqueeze(h, 1)  # Nx1xD'
            ret = resval + ret
        return ret

    def edge_attention(self, edges):
        # an edge UDF to compute unnormalized attention values from src and dst
        a = self.leaky_relu(edges.src['a1'] + edges.dst['a2'])
        a = torch.exp(a).clamp(-10, 10)  # use clamp to avoid overflow
        if self.attn_drop:
            a_drop = self.attn_drop(a)
            return {'a' : a, 'a_drop' : a_drop}
        return {'a' : a, 'a_drop': a}

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 alpha,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GraphAttention(
            g, in_dim, num_hidden, heads[0], feat_drop, attn_drop, alpha, False))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GraphAttention(
                g, num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, alpha, residual))
        # output projection
        self.gat_layers.append(GraphAttention(
            g, num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, alpha, residual))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](h).flatten(1)
            h = self.activation(h)
        # output projection
        logits = self.gat_layers[-1](h).mean(1)
        return logits
