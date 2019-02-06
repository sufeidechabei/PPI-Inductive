"""
Graph Attention Networks
Paper: https://arxiv.org/abs/1710.10903
Code: https://github.com/PetarV-/GAT
GAT with batch processing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gat_message(edges):
    return {'ft': edges.src['ft'], 'a2': edges.src['a2']}


class GATReduce(nn.Module):
    def __init__(self, attn_drop):
        super(GATReduce, self).__init__()
        if attn_drop:
            self.attn_drop = nn.Dropout(p=attn_drop)
        else:
            self.attn_drop = 0

    def forward(self, nodes):
        a1 = torch.unsqueeze(nodes.data['a1'], 1)  # shape (B, 1, 1)
        a2 = nodes.mailbox['a2']  # shape (B, deg, 1)
        ft = nodes.mailbox['ft']  # shape (B, deg, D)
        # attention
        a = a1 + a2  # shape (B, deg, 1)
        e = F.softmax(F.leaky_relu(a), dim=1)
        if self.attn_drop:
            e = self.attn_drop(e)
        return {'accum': torch.sum(e * ft, dim=1)}  # shape (B, D)


class GATFinalize(nn.Module):
    def __init__(self, headid, indim, hiddendim, activation, residual):
        super(GATFinalize, self).__init__()
        self.headid = headid
        self.activation = activation
        self.residual = residual
        self.residual_fc = None
        if residual:
            if indim != hiddendim:
                self.residual_fc = nn.Linear(indim, hiddendim, bias=False)
                nn.init.xavier_normal_(self.residual_fc.weight.data, gain=1.414)

    def forward(self, nodes):
        ret = nodes.data['accum']
        if self.residual:
            if self.residual_fc is not None:
                ret = self.residual_fc(nodes.data['h']) + ret
            else:
                ret = nodes.data['h'] + ret
        return {'head%d' % self.headid: self.activation(ret)}


class GATPrepare(nn.Module):
    def __init__(self, indim, hiddendim, drop):
        super(GATPrepare, self).__init__()
        self.fc = nn.Linear(indim, hiddendim, bias=False)
        if drop:
            self.drop = nn.Dropout(drop)
        else:
            self.drop = 0
        self.attn_l = nn.Linear(hiddendim, 1, bias=False)
        self.attn_r = nn.Linear(hiddendim, 1, bias=False)
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_r.weight.data, gain=1.414)

    def forward(self, feats):
        h = feats
        if self.drop:
            h = self.drop(h)
        ft = self.fc(h)
        a1 = self.attn_l(ft)
        a2 = self.attn_r(ft)
        return {'h': h, 'ft': ft, 'a1': a1, 'a2': a2}


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_heads,
                 activation,
                 in_drop,
                 attn_drop,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.prp = nn.ModuleList()
        self.red = nn.ModuleList()
        self.fnl = nn.ModuleList()
        # input projection (no residual)
        for hid in range(num_heads[0]):
            self.prp.append(GATPrepare(in_dim, num_hidden, in_drop))
            self.red.append(GATReduce(attn_drop))
            self.fnl.append(GATFinalize(hid, in_dim, num_hidden, activation, False))
        # hidden layers
        for l in range(num_layers - 1):
            for hid in range(num_heads[1+l]):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.prp.append(GATPrepare(num_hidden * num_heads[l], num_hidden, in_drop))
                self.red.append(GATReduce(attn_drop))
                self.fnl.append(GATFinalize(hid, num_hidden * num_heads[l],
                                            num_hidden, activation, residual))
        # output projection
        for final_head in range(num_heads[-1]):
            self.prp.append(GATPrepare(num_hidden * num_heads[-2], num_classes, in_drop))
            self.red.append(GATReduce(attn_drop))
            self.fnl.append(GATFinalize(final_head, num_hidden * num_heads[-2],
                                        num_classes, activation, residual))

    def forward(self, features):
        last = features
        for l in range(self.num_layers):
            for hid in range(self.num_heads[l]):
                i = l * self.num_heads[l] + hid
                # prepare
                self.g.ndata.update(self.prp[i](last))
                # message passing
                self.g.update_all(gat_message, self.red[i], self.fnl[i])
            # merge all the heads
            last = torch.cat(
                    [self.g.pop_n_repr('head%d' % hid) for hid in range(self.num_heads[l])],
                    dim=1)
        # output projection
        output = 0
        for final_head in range(self.num_heads[-1]):
            head_index = final_head-self.num_heads[-1]
            self.g.ndata.update(self.prp[head_index](last))
            self.g.update_all(gat_message, self.red[head_index], self.fnl[head_index])
            output = output + self.g.pop_n_repr("head" + str(final_head))
        output/=self.num_heads[-1]
        return output