import networkx as nx
import numpy as np
import torch as th
import pandas as pd
import math
import random
import time
from sklearn.metrics.pairwise import euclidean_distances
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv
from sklearn.cluster import KMeans
import dgl.function as fn
from attacks.utils import *
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

time_limit = 300

def calculate_entropy(probs):
    entropy = -np.sum(probs * np.log(np.clip(probs, a_min=1e-7, a_max=None)), axis=1)
    return entropy

def rank_entropy(net, sub_g, sub_features, sub_train_mask, sub_train_mask_new,
                 num_each, return_rank = True):
    logits = net(sub_g, sub_features)
    prob = F.softmax(logits, dim=-1)
    nodes_interest = th.nonzero(sub_train_mask & ~sub_train_mask_new).squeeze()
    probs_interest = prob[nodes_interest]
    entropy_interest = calculate_entropy(probs_interest)
    nodes_rank = nodes_interest[th.argsort(entropy_interest, descending = True)]
    if len(nodes_rank) >= num_each:
        if return_rank:
            return nodes_rank
        else:
            print(f"=========Update Entropy Querying Label with {num_each} Nodes==========================")
            # selected_indices = random.sample(list(missing_indices), num_each)
            selected_indices = nodes_rank[:num_each]
            sub_train_mask_new[selected_indices] = True
            return sub_train_mask_new

def multiclassentropy_numpy(tens,dim=1):
    reverse = 1 - tens
    ent_1 = -np.log(np.clip(tens, a_min=1e-7,a_max=None)) * tens
    ent_2 = -np.log(np.clip(reverse, a_min=1e-7,a_max=None)) * reverse
    ent = ent_1 + ent_2
    entropy = np.mean(ent, axis=1)

    return entropy

def rank_density(net, sub_g, sub_features, sub_train_mask, sub_train_mask_new,
                 num_each, num_clusters, return_rank = True):
    full_true_indices = th.nonzero(sub_train_mask).squeeze()
    current_true_indices = th.nonzero(sub_train_mask_new).squeeze()
    missing_indices = set(full_true_indices.tolist()) - set(current_true_indices.tolist())
    ## Get the embeddings that we need
    ## Under numpy formality
    embedding_all = net(sub_g, sub_features, return_hidden=True).detach().numpy()
    kmeans = KMeans(n_clusters = num_clusters)
    kmeans.fit(embedding_all)
    ## Set up cluster_centers
    cluster_centers = kmeans.cluster_centers_
    # Calculate the Euclidean distance
    dist = np.linalg.norm(embedding_all - cluster_centers[kmeans.labels_], axis = 1)
    density_scores = th.from_numpy(1 / (1 + dist))
    # pull back to the node coefficients
    list_missing_indices = torch.tensor(list(missing_indices))
    shuffle_order = th.argsort(density_scores, descending = True)
    positions = [th.where(shuffle_order == temp)[0].item() for temp in list_missing_indices]
    sorted_positions = th.argsort(th.tensor(positions))
    list_output = list_missing_indices[sorted_positions]
    if len(list_output) >= num_each:
        if return_rank:
            return list_output
        else:
            print(f"=========Update Entropy Querying Label with {num_each} Nodes==========================")
            # selected_indices = random.sample(list(missing_indices), num_each)
            selected_indices = list_output[:num_each]
            sub_train_mask_new[selected_indices] = True
            return sub_train_mask_new

def rank_centrality(sub_g, sub_train_mask,
                    sub_train_mask_new, num_each, return_rank = True):
    nodes_interest = th.nonzero(sub_train_mask & ~sub_train_mask_new).squeeze()
    page_rank_score = page_rank(sub_g)[nodes_interest]
    nodes_centrality = nodes_interest[th.argsort(page_rank_score, descending=True)]
    if len(nodes_centrality) >= num_each:
        if return_rank:
            return nodes_centrality
        else:
            print(f"=========Update Entropy Querying Label with {num_each} Nodes==========================")
            # selected_indices = random.sample(list(missing_indices), num_each)
            selected_indices = nodes_centrality[:num_each]
            sub_train_mask_new[selected_indices] = True
            return sub_train_mask_new

def centralissimo(graph):
    centralities = []
    graph = graph.to_networkx()
    centralities.append(nx.pagerank(graph))
    L = len(centralities[0])
    Nc = len(centralities)
    cenarray = np.zeros((Nc, L))
    for i in range(Nc):
        cenarray[i][list(centralities[i].keys())]=list(centralities[i].values())
    normcen = (cenarray.astype(float)-np.min(cenarray,axis=1)[:,None])/(np.max(cenarray,axis=1)-np.min(cenarray,axis=1))[:,None]
    return normcen[0]

def page_rank(graph, damping_factor=0.85, max_iter=100, tol=1e-8):
    num_nodes = graph.number_of_nodes()
    
    pagerank_scores = torch.ones(num_nodes) / num_nodes
    graph.ndata['pagerank'] = pagerank_scores
    
    graph.ndata['deg'] = graph.out_degrees().float().clamp(min=1) # Avoid dividing by 0
    
    for _ in range(max_iter):
        prev_scores = pagerank_scores.clone()
        graph.ndata['h'] = pagerank_scores / graph.ndata['deg']
        graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_new'))

        pagerank_scores = damping_factor * graph.ndata['h_new'] + (1 - damping_factor) / num_nodes

        delta = torch.abs(pagerank_scores - prev_scores).sum().item()
        if delta < tol:
            break

        graph.ndata['pagerank'] = pagerank_scores
    return graph.ndata['pagerank']


def find_short_dist(embedding_pool, cluster_centers):
    distances = torch.cdist(embedding_pool, cluster_centers)
    min_distances, _ = torch.min(distances, dim=1)
    return min_distances

def quantile_selection(A, B, C, alpha, beta, gamma, sub_train_mask, sub_train_mask_new, num_each):
    elements = th.nonzero(sub_train_mask & ~sub_train_mask_new).squeeze()
    ranks_A = [compute_rank(A, el) for el in elements]
    ranks_B = [compute_rank(B, el) for el in elements]
    ranks_C = [compute_rank(C, el) for el in elements]
    weighted_ranks = []
    for i in range(len(elements)):
        weighted_rank = alpha * ranks_A[i] + beta * ranks_B[i] + gamma * ranks_C[i]
        weighted_ranks.append(weighted_rank)
    # Sort elements based on weighted ranks
    sorted_indices = np.argsort(weighted_ranks)
    sorted_elements = th.stack([elements[i] for i in sorted_indices])
    # sorted_weighted_ranks = [weighted_ranks[i] for i in sorted_indices]
    print(f"=========Update Entropy Querying Label with {num_each} Nodes==========================")
    # selected_indices = random.sample(list(missing_indices), num_each)
    selected_indices = sorted_elements[:num_each]
    sub_train_mask_new[selected_indices] = True
    return sub_train_mask_new
def compute_rank(tensor, element):
    return np.where(tensor == element)[0][0]
def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        f1score = f1_score(labels.cpu().numpy(), indices.cpu().numpy(), average='macro')
        return correct.item() * 1.0 / len(labels), f1score


def getPool(train_mask, reduce=True):
    pool = th.where(train_mask > 0.9)[0]        
    return pool

def trainOnce():
    return 1

# calculate the percentage of elements smaller than the k-th element
def perc(input):
    return 1-np.argsort(np.argsort(-input,kind='stable'),kind='stable')/len(input)

# calculate the percentage of elements larger than the k-th element
def percd(input):
    return 1-np.argsort(np.argsort(input,kind='stable'),kind='stable')/len(input)


def attack0(dataset_name, seed, cuda, 
            attack_node_arg=0.25, file_path = '',
            LR = 1e-3,
            TGT_LR = 1e-2,
            EVAL_EPOCH = 10,
            TGT_EPOCH = 10,
            WARMUP_EPOCH = 400,
            dropout = False,
            model_performance = True):
    
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
            
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test F1 macro {:.4f}".format(
            epoch, loss.item(), acc, f1score))

    g = g.cpu()
    features = features.cpu()
    gcn_Net = gcn_Net.cpu()
    train_mask = train_mask.cpu()
    test_mask = test_mask.cpu()
    labels = labels.cpu()
    
    # Generate sub-graph index
    alpha = 0.8
    g_matrix = np.asmatrix(g.adjacency_matrix().to_dense())

    sub_graph_node_index = []
    for i in range(attack_node_number):
        sub_graph_node_index.append(random.randint(0, node_number - 1))
    sub_labels = labels[sub_graph_node_index]

    syn_nodes = []
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
    
    # Generate sub-graph adj-matrix, features, labels
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
    
    sub_g = DGLGraph(sub_g)
    # normalization
    degs = sub_g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    
    sub_g.ndata['norm'] = norm.unsqueeze(1)
    dur = []
    
    print("=========Model Extracting==========================")
    idx_train = []
    max_acc1 = 0 
    max_acc2 = 0
    max_f1 = 0
    if dropout == True:
        net = GCN_drop(feature_number, label_number)
    else: 
        net = GcnNet(feature_number, label_number)
    optimizer = th.optim.Adam(net.parameters(), lr=LR, weight_decay=5e-4)

    # warmup
    initial_set = []
    for label in range(label_number):
        label_nodes = []
        for i, l in enumerate(sub_labels):
            if sub_train_mask[i] == True and l == label:
                label_nodes.append(i)
        selected_nodes = random.sample(label_nodes, k=2)  # initial pool for each class
        initial_set.extend(selected_nodes)

    attack_warmup = th.zeros(len(sub_train_mask), dtype=th.bool)
    attack_warmup[initial_set] = True
    net.train()

    for epoch in range(WARMUP_EPOCH):
        logits = net(sub_g, sub_features)
        logp = F.log_softmax(logits, dim=1)

        loss = F.nll_loss(logp[attack_warmup], sub_labels_query[attack_warmup])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc, f1score = evaluate(net, g, features, labels, test_mask)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test F1 score {:.4f}".format(
            epoch + 1, loss.item(), acc, f1score))


    attack_nodes = attack_warmup
    idx_train.extend(initial_set)

    start_time = time.time()
    log_dir = f"{file_path}/timelogs/{dataset_name}/logtime_age_{seed}"
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    
    # select One node using AGE
    for epoch in range((20*label_number)):
        net.eval()
        logits = net(sub_g, sub_features)
        probs = F.softmax(logits, dim=-1).detach().numpy()

        entropy = calculate_entropy(probs)
        kmeans = KMeans(n_clusters=label_number, random_state=0).fit(probs)
        ed = euclidean_distances(probs, kmeans.cluster_centers_)
        ed_score = np.min(ed, axis=1)
        edprec = percd(ed_score)

        normcen = centralissimo(sub_g)
        cenperc = perc(normcen)

        entrperc = perc(entropy)

        # calc final weights
        # gamma = np.random.beta(1, 1.005 - 0.95 ** epoch)
        if dataset_name == 'cora':
            gamma = 0.7
        elif dataset_name == 'citeseer':
            gamma = 0.3
        elif dataset_name == 'pubmed':
            gamma = 0.9
        else:
            gamma = np.random.beta(1, 1.005 - 0.95 ** epoch)
        alpha = beta = (1- gamma) / 2
        final_weight = entrperc + beta * edprec + gamma * cenperc
        mask = (sub_train_mask == True) & (attack_warmup == False)
        final_weight = final_weight[mask]
        temp = np.argmax(final_weight)
        select = np.where(mask)[0][temp]
        
        idx_train.append(int(select))
        attack_warmup[select] = True

        # train net Once
        net.train()
        logits = net(sub_g, sub_features)
        logp = F.log_softmax(logits, dim=1)

        loss = F.nll_loss(logp[attack_warmup], sub_labels_query[attack_warmup])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, _ = evaluate(net, g, features, labels_query, test_mask)
        acc2, f1score = evaluate(net, g, features, labels, test_mask)
        if acc1 > max_acc1:
            max_acc1 = acc1
        if acc2 > max_acc2:
            max_acc2 = acc2
        if f1score > max_f1:
            max_f1 = f1score
        print("Test Acc {:.4f} | Test Fid {:.4f} | Test F1score {:.4f} ".format(
            acc2, acc1, max_f1))

    node_selection_time = time.time() - start_time
    with open(log_dir, 'w') as log_file:
        log_file.write(f"AGE {dataset_name} {seed} ")
        log_file.write(f"{node_selection_time:.4f}s\n")

    output_data = {
        'total_sub_nodes': total_sub_nodes,
        'idx_train': idx_train
    }
    
    print('node selection finished')
    with open(f'./node_selection/AGE_{dataset_name}_selected_nodes_{(20*label_number)}_{seed}.json','w')as f:
        json.dump(output_data, f)
    
    sub_g = sub_g.to(device)
    sub_features = sub_features.to(device)
    sub_labels_query = sub_labels_query.to(device)
    labels_query = labels_query.to(device)
    g = g.to(device)
    features = features.to(device)
    test_mask = test_mask.to(device)
    labels = labels.to(device)

    print('========================== Model Evaluation ==========================')
    if model_performance:
        for num_node in range(2*label_number, 21*label_number, label_number):

            # init model
            set_seed(seed)
            if dropout == True:
                net = GCN_drop(feature_number, label_number)
            else: 
                net = GcnNet(feature_number, label_number)
            optimizer = th.optim.Adam(net.parameters(), lr=LR, weight_decay=5e-4)

            # recover sub_train_mask
            attack_nodes = th.zeros(len(sub_train_mask), dtype=th.bool)
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
                'Method': ['age'],
                'Test Accuracy': [max_acc2],
                'Test Fidelity': [max_acc1],
                'Test F1score': [max_f1],
            })         
            metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)
            print("Test Acc {:.4f} | Test Fid {:.4f} | Test F1score {:.4f} | Time(s) {:.4f}".format(
                acc2, acc1, max_f1, np.mean(dur)))

        epoch_metrics = pd.DataFrame({
            'Num Attack Nodes': [int(th.sum(train_mask))],
            'Method': ['age'],
            'Test Accuracy': [target_performance['acc']],
            'Test Fidelity': [1],
            'Test F1score': [target_performance['f1score']],
        })         
        metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)

        log_file_path = f"{file_path}/{dataset_name}/log_age_{seed}.csv"
        metrics_df.to_csv(log_file_path, mode='w', header=False, index=False)

    if True:
        set_seed(seed)
        log_file_path = f"{file_path}/{dataset_name}/log_age_{seed}.csv"
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
            'Method': ['age'],
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
                'Method': ['age'],
                'Attack node Num': [attack_node_number],
            })
            metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)

        log_file_path = f"{file_path}/gaps/{dataset_name}/log_age_{seed}.csv"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        metrics_df.to_csv(log_file_path, mode='w', header=True, index=False)



