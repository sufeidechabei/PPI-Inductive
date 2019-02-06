import numpy as np
import json
import networkx as nx
from networkx.readwrite import json_graph
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from gat import GAT
device = torch.device('cuda:0')
def process_p2p():
    print('Loading G...')
    with open('ppi/ppi-G.json') as jsonfile:
        g_data = json.load(jsonfile)

    with open('ppi/ppi-class_map.json') as jsonfile:
        class_map = json.load(jsonfile)
    features = np.load('ppi/ppi-feats.npy')
    label_list = []
    for i in range(len(class_map)):
        label_list.append(np.expand_dims(np.array(class_map[str(i)]), axis=0))
    labels = np.concatenate(label_list)
    G = nx.DiGraph(json_graph.node_link_graph(g_data))
    ### train_mask = [node['id'] for node in g_data['nodes']
                  ### if node['test'] is False and node['val'] is False]
    test_mask = [node['id'] for node in g_data['nodes']
                  if node['test'] is True]
    valid_mask = [node['id'] for node in g_data['nodes']
                  if node['val'] is True]
    graph_id = np.load('./ppi/graph_id.npy')
    train_mask_list =  []
    for train_graph_id in range(1, 21):
        train_mask_list.append(np.where(graph_id==train_graph_id)[0])

    return G, features, labels, train_mask_list, test_mask, valid_mask


graph, features, labels, train_mask_list, test_mask, valid_mask = process_p2p()
train_feats = features[np.concatenate(train_mask_list)]
scaler = StandardScaler()
scaler.fit(train_feats)
features = scaler.transform(features)

features = torch.from_numpy(features).float().to(device)
labels = torch.from_numpy(labels).to(device)
g = DGLGraph(graph)
n_classes = labels.size()[1]
num_feats = features.size()[1]
batch_size = 2
cur_step = 0
patience = 10
batch_list = []
for batch_index in range(len(train_mask_list)//batch_size):
    begin_index = batch_index * batch_size
    end_index = (batch_index+1) * batch_size
    batch_list.append(np.concatenate(train_mask_list[begin_index:end_index]))

def evaluate(mask, model):
    with torch.no_grad():
        model.eval()
        model.g = g.subgraph(mask)
        output = model(features[mask].float())
        loss_data = loss_fcn(output, labels[mask].float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels[mask].data.cpu().numpy(),
                         predict, average='micro')
        print("F1-score: {:.4f} ".format(score))
        return score, loss_data.item()

best_score = -1
best_loss = 10000
best_model = None
best_loss_curve = []
val_early_loss = 10000
val_early_score = -1
model = GAT(g,
            2,
            num_feats,
            256,
            n_classes,
            [4, 4, 6],
            F.elu,
            0,
            0,
            True)

loss_fcn = torch.nn.BCEWithLogitsLoss()

# use optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
model = model.to(device)
for epoch in range(400):
    model.train()
    loss_list = []
    for train_batch in batch_list:
        model.g = g.subgraph(train_batch)
        input_feature = features[train_batch]
        logits = model(input_feature)
        loss = loss_fcn(logits, labels[train_batch].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    loss_data = np.array(loss_list).mean()
    print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))
    if epoch % 5 == 0:
        score, val_loss = evaluate(valid_mask, model)
        if score > best_score or best_loss>val_loss:
            if score > best_score and best_loss > val_loss:
                val_early_loss = val_loss
                val_early_score = score
            best_score = np.max((score, best_score))
            best_loss = np.min((best_loss, val_loss))
            cur_step = 0
        else:
            cur_step+=1
            if cur_step==patience:
                break

evaluate(test_mask, model)
torch.save(best_model.state_dict(), './gat_best_model.ckpt')
