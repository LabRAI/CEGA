# -*- coding: utf-8 -*-


from dgl.data import citation_graph as citegrh
import networkx as nx
import numpy as np
import torch as th
import math
import random
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time
from dgl.nn.pytorch import GraphConv
from attacks.utils import *
from sklearn.metrics import f1_score
import json
from tqdm import tqdm

from dgl.data import AmazonCoBuyComputerDataset
from dgl.data import AmazonCoBuyPhotoDataset
from dgl.data import CoauthorCSDataset
from dgl.data import CoauthorPhysicsDataset
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

time_limit = 300


def attack0(dataset_name, seed, cuda, 
            attack_node_arg = 0.25, file_path = '', 
            LR = 1e-3,
            TGT_LR = 1e-2,
            EVAL_EPOCH = 1000,
            TGT_EPOCH = 1000, 
            dropout = True,
            model_performance = True,
            **kwargs):
    
    device = th.device(cuda)
    set_seed(seed)
    metrics_df = pd.DataFrame(columns=['Num Attack Nodes', 'Method' ,'Test Accuracy', 'Test Fidelity'])

    g, features, labels, node_number, train_mask, test_mask = load_data(dataset_name)
    train_mask = train_mask.type(torch.bool)
    test_mask = test_mask.type(torch.bool)
    
    attack_node_number = int(node_number * attack_node_arg)
    feature_number = features.shape[1]
    label_number = len(labels.unique())
    
    # normalization
    g = g.to(device)
    degs = g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    if cuda != None:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)
    if dropout == True:
        gcn_Net = GCN_drop(feature_number, label_number)
    else: 
        gcn_Net = GcnNet(feature_number, label_number)
    optimizer = th.optim.Adam(gcn_Net.parameters(), lr=TGT_LR, weight_decay=5e-4)
    dur = []

    features = features.to(device)
    gcn_Net = gcn_Net.to(device)
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    labels = labels.to(device)
    target_performance = {
                        'acc': 0,
                        'f1score': 0
                        } 
    
    print("=========Target Model Generating==========================")
    for epoch in range(TGT_EPOCH):
        if epoch >= 3:
            t0 = time.time()

        gcn_Net.train()
        logits = gcn_Net(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)
        
        acc, f1score = evaluate(gcn_Net, g, features, labels, test_mask)
        if acc > target_performance['acc']:
            target_performance['acc'] = acc
        if f1score > target_performance['f1score']:
            target_performance['f1score'] = f1score 
            
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test F1 macro {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, f1score, np.mean(dur)))

    g = g.cpu()
    features = features.cpu()
    gcn_Net = gcn_Net.cpu()
    train_mask = train_mask.cpu()
    test_mask = test_mask.cpu()
    labels = labels.cpu()

    # Generate sub-graph index
    alpha = 0.8
    sub_graph_node_index = []
    for i in range(attack_node_number):
        sub_graph_node_index.append(random.randint(0, node_number - 1))

    sub_labels = labels[sub_graph_node_index]

    syn_nodes = []
    g_matrix = np.asmatrix(g.adjacency_matrix().to_dense())

    for node_index in sub_graph_node_index:
        # get nodes
        one_step_node_index = g_matrix[node_index, :].nonzero()[1].tolist()
        two_step_node_index = []
        for first_order_node_index in one_step_node_index:
            syn_nodes.append(first_order_node_index)
            two_step_node_index = g_matrix[first_order_node_index, :].nonzero()[1].tolist()

    sub_graph_syn_node_index = list(set(syn_nodes) - set(sub_graph_node_index))
    total_sub_nodes = list(set(sub_graph_syn_node_index + sub_graph_node_index))

    # Generate features for SubGraph attack
    np_features_query = features.clone()

    for node_index in sub_graph_syn_node_index:
        # initialized as zero
        np_features_query[node_index] = np_features_query[node_index] * 0
        # get one step and two steps nodes
        one_step_node_index = g_matrix[node_index, :].nonzero()[1].tolist()
        one_step_node_index = list(set(one_step_node_index).intersection(set(sub_graph_node_index)))

        total_two_step_node_index = []
        num_one_step = len(one_step_node_index)
        for first_order_node_index in one_step_node_index:
            # caculate the feature: features =  0.8 * average_one + 0.8^2 * average_two
            # new_array = features[first_order_node_index]*0.8/num_one_step
            this_node_degree = len(g_matrix[first_order_node_index, :].nonzero()[1].tolist())
            np_features_query[node_index] = torch.from_numpy(np.sum(
                [np_features_query[node_index],
                 features[first_order_node_index] * alpha / math.sqrt(num_one_step * this_node_degree)],
                axis=0))

            two_step_node_index = g_matrix[first_order_node_index, :].nonzero()[1].tolist()
            total_two_step_node_index = list(
                set(total_two_step_node_index + two_step_node_index) - set(one_step_node_index))
        total_two_step_node_index = list(set(total_two_step_node_index).intersection(set(sub_graph_node_index)))

        num_two_step = len(total_two_step_node_index)
        for second_order_node_index in total_two_step_node_index:

            # caculate the feature: features =  0.8 * average_one + 0.8^2 * average_two
            this_node_second_step_nodes = []
            this_node_first_step_nodes = g_matrix[second_order_node_index, :].nonzero()[1].tolist()
            for nodes_in_this_node in this_node_first_step_nodes:
                this_node_second_step_nodes = list(
                    set(this_node_second_step_nodes + g_matrix[nodes_in_this_node, :].nonzero()[1].tolist()))
            this_node_second_step_nodes = list(set(this_node_second_step_nodes) - set(this_node_first_step_nodes))

            this_node_second_degree = len(this_node_second_step_nodes)
            np_features_query[node_index] = torch.from_numpy(np.sum(
                [np_features_query[node_index],
                 features[second_order_node_index] * (1 - alpha) / math.sqrt(num_two_step * this_node_second_degree)],
                axis=0))

    features_query = th.FloatTensor(np_features_query)

    #  Generate sub-graph adj-matrix, features, labels
    total_sub_nodes = list(set(sub_graph_syn_node_index + sub_graph_node_index))
    sub_g = np.zeros((len(total_sub_nodes), len(total_sub_nodes)))
    for sub_index in range(len(total_sub_nodes)):
        sub_g[sub_index] = g_matrix[total_sub_nodes[sub_index], total_sub_nodes]

    for i in range(node_number):
        if i in sub_graph_node_index:
            test_mask[i] = False
            train_mask[i] = True
            continue
        if i in sub_graph_syn_node_index:
            test_mask[i] = True
            train_mask[i] = False
        else:
            test_mask[i] = True
            train_mask[i] = False

    sub_train_mask = train_mask[total_sub_nodes]

    sub_features = features_query[total_sub_nodes]
    sub_labels = labels[total_sub_nodes]

    sub_features = th.FloatTensor(sub_features)
    sub_labels = th.LongTensor(sub_labels)
    sub_train_mask = sub_train_mask

    gcn_Net.eval()

    # =================Generate Label===================================================
    logits_query = gcn_Net(g, features)
    _, labels_query = th.max(logits_query, dim=1)

    sub_labels_query = labels_query[total_sub_nodes]
    sub_g = nx.from_numpy_array(sub_g)

    sub_g.remove_edges_from(nx.selfloop_edges(sub_g))
    sub_g.add_edges_from(zip(sub_g.nodes(), sub_g.nodes()))

    sub_g = dgl.from_networkx(sub_g) # DGLGraph(sub_g)
    n_edges = sub_g.number_of_edges()
    # normalization
    degs = sub_g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0

    sub_g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    dur = []

    print("=========Model Extracting==========================")
        
    # warmup
    idx_train = []
    initial_set = []
    for label in range(label_number):
        label_nodes = []
        for i, l in enumerate(sub_labels):
            if sub_train_mask[i] == True and l == label:
                label_nodes.append(i)
        selected_nodes = random.sample(label_nodes, k=2)  # Select 4 nodes per class
        initial_set.extend(selected_nodes)
    idx_train.extend(initial_set)

    idx_avaliable = [i for i,j in enumerate(sub_train_mask) if j==True and i not in idx_train] # graph nodes in training set
    idx_avaliable_temp = copy.deepcopy(idx_avaliable)
    count = 0
    while True:
        random_node = random.choice(idx_avaliable)                                       
        idx_train.append(random_node) 
        idx_avaliable.remove(random_node)
        idx_avaliable_temp.remove(random_node)
        count += 1

        if count >= (20*label_number):
            break

    output_data = {
        'total_sub_nodes': total_sub_nodes,
        'idx_train': idx_train
    }
    
    print('node selection finished')
    with open(f'./node_selection/Random_{dataset_name}_select_nodes_{20*label_number}_{seed}.json', 'w') as f:
        json.dump(output_data, f)
    
    sub_g = sub_g.to(device)
    sub_features = sub_features.to(device)
    sub_labels_query = sub_labels_query.to(device)
    labels_query = labels_query.to(device)
    g = g.to(device)
    features = features.to(device)
    test_mask = test_mask.to(device)
    labels = labels.to(device)

    # print('========================== Model Evaluation ==========================')
    if model_performance:
        for num_node in tqdm(range(2*label_number, 21*label_number, label_number), desc='Model Eval'):

            set_seed(seed)
            if dropout == True:
                net = GCN_drop(feature_number, label_number)
            else: 
                net = GcnNet(feature_number, label_number)
            optimizer = th.optim.Adam(net.parameters(), lr=LR, weight_decay=5e-4)

            # recover sub_train_mask
            attack_nodes = th.zeros_like(sub_train_mask)
            attack_nodes[idx_train[:num_node]] = True
            attack_nodes = attack_nodes.to(device)
            net = net.to(device)
            
            max_acc1 = 0
            max_acc2 = 0
            max_f1 = 0
            for epoch in range(EVAL_EPOCH):

                net.train()
                logits = net(sub_g, sub_features)

                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp[attack_nodes], sub_labels_query[attack_nodes])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                acc1, _ = evaluate(net, g, features, labels_query, test_mask)
                acc2, f1score = evaluate(net, g, features, labels, test_mask)
                if acc1 > max_acc1:
                    max_acc1 = acc1
                if acc2 > max_acc2:
                    max_acc2 = acc2
                if f1score > max_f1:
                    max_f1 = f1score
                
            epoch_metrics = pd.DataFrame({
                'Num Attack Nodes': [num_node],
                'Method': ['random'],
                'Test Accuracy': [max_acc2],
                'Test Fidelity': [max_acc1],
                'Test F1score': [max_f1],
            })         
            metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)
            print("Test Acc {:.4f} | Test Fid {:.4f} | Test F1score {:.4f} | Time(s) {:.4f}".format(
                acc2, acc1, max_f1, np.mean(dur)))

        epoch_metrics = pd.DataFrame({
            'Num Attack Nodes': [int(th.sum(train_mask))],
            'Method': ['random'],
            'Test Accuracy': [target_performance['acc']],
            'Test Fidelity': [1],
            'Test F1score': [target_performance['f1score']],
        })         
        metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)

        log_file_path = f"{file_path}/{dataset_name}/log_random_{seed}.csv"
        metrics_df.to_csv(log_file_path, mode='w', header=False, index=False)

    if True:
        set_seed(seed)
        log_file_path = f"{file_path}/{dataset_name}/log_random_{seed}.csv"
        if dropout == True:
            net_full = GCN_drop(feature_number, label_number)
        else: 
            net_full = GcnNet(feature_number, label_number)
        optimizer_full = th.optim.Adam(net_full.parameters(), lr=LR, weight_decay=5e-4)

        net_full = net_full.to(device)

        perfm_attack = {
            'acc': 0,
            'fid': 0,
            'f1score': 0
            } 
        
        print('========================== Model Evaluation ==========================')
        progress_bar = tqdm(range(EVAL_EPOCH), desc="Generating model with ALL attack nodes", ncols=100)
        for epoch in progress_bar:
            if epoch >= 3:
                t0 = time.time()

            net_full.train()
            logits = net_full(sub_g, sub_features)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, sub_labels_query) #[sub_train_mask]

            optimizer_full.zero_grad()
            loss.backward()
            optimizer_full.step()

            if epoch >= 3:
                dur.append(time.time() - t0)
            
            acc, f1score = evaluate(net_full, g, features, labels, test_mask)
            fid, _ = evaluate(net_full, g, features, labels_query, test_mask)
            if acc > perfm_attack['acc']:
                perfm_attack['acc'] = acc
            if fid > perfm_attack['fid']:
                perfm_attack['fid'] = fid
            if f1score > perfm_attack['f1score']:
                perfm_attack['f1score'] = f1score 
            
        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}",
            "Test Acc": f"{acc:.4f}",
            "Test F1": f"{f1score:.4f}",
            # "Processed %": f"{(epoch + 1) / TGT_EPOCH * 100:.2f}",
            # "Time(s)": f"{np.mean(dur) if dur else 0:.4f}"
        })
        sub_train_mask.cpu()
        epoch_metrics = pd.DataFrame({
            'Num Attack Nodes': [sub_train_mask.sum().item()],
            'Method': ['random'],
            'Test Accuracy': [perfm_attack['acc']],
            'Test Fidelity': [perfm_attack['fid']],
            'Test F1score': [perfm_attack['f1score']],
        }) 
        metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)
        metrics_df.to_csv(log_file_path, mode='w', header=False, index=False)
        

    
    if False:
        metrics_df = pd.DataFrame()

        for num_node in tqdm(range(2 * label_number, 21 * label_number, 2), desc='Model Eval'):

            set_seed(seed)
            if dropout:
                net = GCN_drop(feature_number, label_number)
            else:
                net = GcnNet(feature_number, label_number)

            optimizer = th.optim.Adam(net.parameters(), lr=LR, weight_decay=5e-4)

            # Recover sub_train_mask
            attack_nodes = th.zeros_like(sub_train_mask)
            attack_nodes[idx_train[:num_node]] = True
            attack_nodes = attack_nodes.to(device)
            net = net.to(device)

            max_acc2 = 0

            for epoch in range(EVAL_EPOCH):

                net.train()
                logits = net(sub_g, sub_features)

                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp[attack_nodes], sub_labels_query[attack_nodes])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc2, _ = evaluate(net, g, features, labels, test_mask)
                if acc2 > max_acc2:
                    max_acc2 = acc2

            diff = abs(1 - max_acc2)

            epoch_metrics = pd.DataFrame({
                'Num Attack Nodes': [num_node],
                'Accuracy': [max_acc2],
                'Accuracymax': [perfm_attack['acc']],
                'Method': ['random'],
                'Attack node Num': [attack_node_number],
            })
            metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)

        log_file_path = f"{file_path}/gaps/{dataset_name}/log_random_{seed}.csv"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        metrics_df.to_csv(log_file_path, mode='w', header=True, index=False)
