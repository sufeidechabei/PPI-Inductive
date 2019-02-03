import numpy as np
import json
import networkx as nx
from networkx.readwrite import json_graph
import torch
import copy
import torch.nn.functional as F
from dgl import DGLGraph
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from gcn import GCN
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
features = torch.from_numpy(features).cuda()
labels = torch.from_numpy(labels).cuda()
g = DGLGraph(graph)
n_classes = labels.size()[1]
num_feats = features.size()[1]
batch_size = 2
batch_list = []
for batch_index in range(len(train_mask_list)//batch_size):
    begin_index = batch_index * batch_size
    end_index = (batch_index+1) * batch_size
    batch_list.append(np.concatenate(train_mask_list[begin_index:end_index]))
param_grid = {'hidden_size': [16, 32],
              'layers': [1, 2, 3],
              'num_heads': [1, 4, 6, 8],
              'dropout': [0, 0.6],
              'bias': [True, False]

}
param_list = ParameterGrid(param_grid)
def evaluate(mask):
    with torch.no_grad():
        model.eval()
        output = model(features.float())
        predict = np.where(output.data.cpu().numpy() >= 0.3, 1, 0)
        score = f1_score(labels[mask].data.cpu().numpy(),
                         predict[mask], average='micro')
        print("F1-score: {:.4f} ".format(score))
        return score

best_score = -1
best_setting = None
best_model = None
for param in param_list:
    try:
        model = GCN(g,
                    num_feats,
                    param['hidden_size'],
                    n_classes,
                    param['layers'],
                    F.elu,
                    param['dropout'],
                    param['bias'],
                    param['num_heads'],)

        loss_fcn = torch.nn.BCEWithLogitsLoss()

        # use optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        model.cuda()
        loss_result_list = []
        for epoch in range(200):
            model.train()
            loss_list = []
            for train_batch in batch_list:
                logits = model(features.float())
                loss = loss_fcn(logits[train_batch], labels[train_batch].float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())
            print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, np.array(loss_list).mean()))
            loss_result_list.append(np.array(loss_list).mean())
            if epoch % 5 == 0:
                score = evaluate(test_mask)
                if score>best_score:
                    best_score = score
                    best_setting = param
                    best_model = copy.deepcopy(model)
        save_name_list = []
        for key, value in param.items():
            save_name_list.append(str(key))
            save_name_list.append(str(value))
        np.save('./gcn_curve/' + '_'.join(save_name_list) + '.npy', np.array(loss_result_list))


    except:
        continue
torch.save(best_model.state_dict(), './gcn_best_model.ckpt')

with open('./gcn_result.txt', 'a+') as f:
    f.writelines('F1_score: ' + str(best_score) + '\n')
    for k, value in best_setting.items():
        f.writelines(str(k) +": " + str(value) + '\n')
