# -*- coding: utf-8 -*-

import dgl
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
from dgl import DGLGraph
import time
from dgl.nn.pytorch import GraphConv
from sklearn.cluster import KMeans
from attacks.utils import *
from tqdm import tqdm
import pandas as pd
from dgl.data import AmazonCoBuyComputerDataset
from dgl.data import AmazonCoBuyPhotoDataset
from dgl.data import CoauthorCSDataset
from dgl.data import CoauthorPhysicsDataset
from dgl.data import ChameleonDataset
from dgl.data import CornellDataset
from dgl.data import SquirrelDataset
import dgl.function as fn
import json
import os

time_limit = 300

# Initialization
def init_mask(C, sub_train_mask, sub_labels):
    # print(f"=========Initialization with {2 * C} Nodes==========================")
    initial_set = []
    for label in range(C):
        label_nodes = []
        for i, l in enumerate(sub_labels):
            if sub_train_mask[i] == True and l == label:
                label_nodes.append(i)
        selected_nodes = random.sample(label_nodes, k=2)  # initial pool for each class
        initial_set.extend(selected_nodes)

    # print(initial_set)
    return initial_set

    # node pool
    ## center_rank = rank_centrality(sub_g, sub_train_mask, sub_train_init, num_center, return_rank=True)
    ## selected_indices_center = center_rank[:num_center]
    ## sub_train_init[selected_indices_center] = True
    # Randomly select the rest of the initial nodes
    ## full_true_indices = th.nonzero(sub_train_mask & ~sub_train_init).squeeze()
    ## selected_indices_random = random.sample(full_true_indices.tolist(), num_random)
    ## sub_train_init[selected_indices_random] = True

    # Transform the formality and return the outcome; note the output are indicators
    # sub_train_init = th.zeros(len(sub_train_mask), dtype=th.bool)
    # sub_train_init[initial_set] = True
    # print(sub_labels[initial_set])
    # sub_train_init = th.tensor(initial_set)
    # return sub_train_init

def update_sub_train_mask(num_each, sub_train_mask, sub_train_mask_new):
    full_true_indices = th.nonzero(sub_train_mask).squeeze()
    current_true_indices = th.nonzero(sub_train_mask_new).squeeze()
    missing_indices = set(full_true_indices.tolist()) - set(current_true_indices.tolist())
    if len(missing_indices) >= num_each:
        # print(f"=========Update Random Querying Label with {num_each} Nodes==========================")
        selected_indices = random.sample(list(missing_indices), num_each)
        ## sub_train_mask_new[selected_indices] = True

    return selected_indices

    # net.eval()  # Set the model to evaluation mode
    # with torch.no_grad():
        # Implement your custom logic here to update sub_train_mask
        # This can be based on uncertainty, influence, or any other criteria
    #     new_mask = custom_algorithm(sub_train_mask, sub_labels_query, net, g, features)
    # return new_mask

# Calculate the entropy
def calculate_entropy(probs):
    return -th.sum(probs * th.log(probs + 1e-9), dim = -1)

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

# Hand-written pagerank score
def page_rank(graph, damping_factor=0.85, max_iter=100, tol=1e-8):
    num_nodes = graph.number_of_nodes()

    # Initialize the PageRank score for all nodes to be uniform
    pagerank_scores = torch.ones(num_nodes) / num_nodes
    graph.ndata['pagerank'] = pagerank_scores

    # Degree normalization factor
    # with graph.local_scope():
    graph.ndata['deg'] = graph.out_degrees().float().clamp(min=1) # Avoid dividing by 0

    for _ in range(max_iter):
        # Perform message passing (send normalized pagerank score)
        # print("Iteration ", _)
        prev_scores = pagerank_scores.clone()
        graph.ndata['h'] = pagerank_scores / graph.ndata['deg']
        graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_new'))
        # Apply PageRank formula
        pagerank_scores = damping_factor * graph.ndata['h_new'] + (1 - damping_factor) / num_nodes
        # pagerank_scores_new = (1 - damping_factor) / num_nodes + damping_factor * graph.ndata['pagerank_sum'] / \
        #                       graph.ndata['deg']

        # Check for convergence
        delta = torch.abs(pagerank_scores - prev_scores).sum().item()
        if delta < tol:
            break
        # Update pagerank scores
        graph.ndata['pagerank'] = pagerank_scores

    return graph.ndata['pagerank']


# ECE and Perturbation
def perturb_features(sub_features, noise_level = 0.05):
    noise = th.randn_like(sub_features) * noise_level
    perturbed_features = sub_features + noise
    return perturbed_features

# Take the perturbation and count the average
def perturb_avg(net, sub_g, sub_features, num_perturbations, noise_level):
    original_logits = net(sub_g, sub_features)
    # Number of classes
    num_classes = original_logits.size(-1)
    # Initialization
    cumulative_probs = th.zeros(sub_features.size(0), num_classes,
                                device=original_logits.device)
    # Perturbation
    for _ in range(num_perturbations):
        features_p = perturb_features(sub_features, noise_level=noise_level)
        logits_p = net(sub_g, features_p)
        probs_p = F.softmax(logits_p, dim=-1)
        cumulative_probs += probs_p
    # get a fair estimation for the distribution on existing label
    avg_probs = cumulative_probs / num_perturbations

    return avg_probs

# Try the traditional way: count the number of perturbed labels for each node
def rank_perturb(net, sub_g, sub_features, num_perturbations,
                        sub_train_mask, sub_train_mask_new, noise_level,
                        num_each, return_rank = True):
    original_logits = net(sub_g, sub_features)
    nodes_interest = th.nonzero(sub_train_mask & ~sub_train_mask_new).squeeze()
    original_pred = th.argmax(original_logits[nodes_interest], dim = -1)
    ## Store the outcome
    # unchanged_counts = th.zeros_like(original_pred, dtype = th.float)
    unchanged_counts = th.zeros_like(original_pred)
    # Perturbation
    for _ in range(num_perturbations):
        features_p = perturb_features(sub_features, noise_level=noise_level)
        logits_p = net(sub_g, features_p)
        labels_p = th.argmax(logits_p[nodes_interest], dim = -1)
        unchanged = labels_p.eq(original_pred)
        unchanged_counts += unchanged.int()

    # unchanged_counts_float = unchanged_counts.float()
    # unchanged_counts_float.mean()
    _, change_indices = torch.sort(unchanged_counts)
    nodes_rank_label = nodes_interest[change_indices]

    if len(nodes_rank_label) >= num_each:
        if return_rank:
            return nodes_rank_label
        else:
            print(f"=========Update Perturbation Querying Label with {num_each} Nodes==========================")
            # selected_indices = random.sample(list(missing_indices), num_each)
            selected_indices = nodes_rank_label[:num_each]
            sub_train_mask_new[selected_indices] = True
            return sub_train_mask_new


# Consider items in the embedding space
def rank_cluster(net, sub_g, sub_features, labels, total_sub_nodes,
                sub_train_mask, sub_train_mask_new, num_clusters,
                num_each, return_rank = True):
    # Work on missing indices
    full_true_indices = th.nonzero(sub_train_mask).squeeze()
    current_true_indices = th.nonzero(sub_train_mask_new).squeeze()
    missing_indices = set(full_true_indices.tolist()) - set(current_true_indices.tolist())
    # Work on prep of embedding
    labels_true = labels[total_sub_nodes]
    logits = net(sub_g, sub_features)
    prob = F.softmax(logits, dim=-1)
    labels_pred = th.argmax(prob, dim = -1)
    embedding_all = net(sub_g, sub_features, return_hidden = True)
    mismatches_queried = (labels_true != labels_pred) & sub_train_mask_new
    selected_embeddings = embedding_all[mismatches_queried].detach().numpy()
    # Try kmeans
    num_clusters_used = min(num_clusters, th.sum(mismatches_queried).item())
    # print(selected_embeddings)
    print("mismatches_queried:" + str(th.sum(mismatches_queried).item()))
    print("num_clusters_used:" + str(num_clusters_used))
    if num_clusters_used >= 1:
        kmeans = KMeans(n_clusters=num_clusters_used, random_state=0)
        kmeans.fit(selected_embeddings)
        cluster_centers = th.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        # Get back to the original field: Try to use a separate function for remaining functions
        list_missing_indices = list(missing_indices)
        embedding_pool = embedding_all[list_missing_indices]
        min_distances = find_short_dist(embedding_pool, cluster_centers)
        shuffle_order = th.argsort(min_distances)
        output_order = [list_missing_indices[i] for i in shuffle_order]
        nodes_rank_distance = torch.tensor(output_order)
    else:
        print("All nodes give the same label.")
        nodes_rank_distance = torch.tensor(list(missing_indices))

    if len(nodes_rank_distance) >= num_each:
        if return_rank:
            return nodes_rank_distance
        else:
            print(f"=========Update Cluster Querying Label with {num_each} Nodes==========================")
            # selected_indices = random.sample(list(missing_indices), num_each)
            selected_indices = nodes_rank_distance[:num_each]
            sub_train_mask_new[selected_indices] = True
            return sub_train_mask_new


# Use a separate function to write out the calculation of distance
def find_short_dist(embedding_pool, cluster_centers):
    distances = torch.cdist(embedding_pool, cluster_centers)
    min_distances, _ = torch.min(distances, dim=1)
    return min_distances


# Consider Diversity; see what we can do from here.
def rank_diversity(net, sub_g, sub_features, sub_train_mask, sub_train_mask_new, num_each, num_clusters, rho, return_rank = True):
    full_indices = th.nonzero(sub_train_mask).squeeze()
    queried_indices = th.nonzero(sub_train_mask_new).squeeze()
    candidate_indices = set(full_indices.tolist()) - set(queried_indices.tolist())
    # Get the embeddings
    embedding_all = net(sub_g, sub_features, return_hidden = True).detach().numpy()
    embedding_queried = embedding_all[queried_indices]
    kmeans = KMeans(n_clusters = num_clusters, random_state = 42)
    kmeans.fit(embedding_queried)
    cluster_centers = kmeans.cluster_centers_

    node_embeddings = th.tensor(embedding_all, dtype = th.float32)
    centroids = th.tensor(cluster_centers, dtype = th.float32)
    kmeans_labels = th.tensor(kmeans.labels_, dtype = th.int32)

    minimal_distance = th.min(th.cdist(node_embeddings, centroids, p = 2), dim = 1).values
    proposed_labels = th.min(th.cdist(node_embeddings, centroids, p = 2), dim = 1).indices

    # Closeness Scores (Distance to assigned centroid)
    close_temp = 1 / (1 + minimal_distance)
    close_normalized = (close_temp - close_temp.min()) / (close_temp.max() - close_temp.min() + 1e-10)

    # Rarity Scores (How rare as shown in )
    queried_bincount = th.bincount(kmeans_labels)
    rarity_temp = 1 / (1 + queried_bincount[proposed_labels])
    rarity_normalized = (rarity_temp - rarity_temp.min()) / (rarity_temp.max() - rarity_temp.min() + 1e-10)

    # Assemble the scores; rho is subject to tuning
    composite_scores = rho * close_normalized + (1 - rho) * rarity_normalized
    composite_scores_candidate = composite_scores[list(candidate_indices)]
    candidate_tensor = th.tensor(list(candidate_indices))
    nodes_rank_diversity = candidate_tensor[th.argsort(composite_scores_candidate, descending=True)]

    if len(nodes_rank_diversity) >= num_each:
        if return_rank:
            return nodes_rank_diversity
        else:
            print(f"=========Update Cluster Querying Label with {num_each} Nodes==========================")
            # selected_indices = random.sample(list(missing_indices), num_each)
            selected_indices = nodes_rank_diversity[:num_each]
            sub_train_mask_new[selected_indices] = True
            return sub_train_mask_new

def quantile_selection(A, B, C, index_1, index_2, index_3, sub_train_mask, sub_train_mask_new, num_each):
    elements = th.nonzero(sub_train_mask & ~sub_train_mask_new).squeeze()

    ranks_A = [compute_rank(A, el) for el in elements]
    ranks_B = [compute_rank(B, el) for el in elements]
    ranks_C = [compute_rank(C, el) for el in elements]

    weighted_ranks = []
    for i in range(len(elements)):
        weighted_rank = index_1 * ranks_A[i] + index_2 * ranks_B[i] + index_3 * ranks_C[i]
        weighted_ranks.append(weighted_rank)

    # Sort elements based on weighted ranks
    sorted_indices = np.argsort(weighted_ranks)
    sorted_elements = th.stack([elements[i] for i in sorted_indices])
    # sorted_weighted_ranks = [weighted_ranks[i] for i in sorted_indices]

    # print(f"=========Update Entropy Querying Label with {num_each} Nodes==========================")
    # selected_indices = random.sample(list(missing_indices), num_each)
    selected_indices = sorted_elements[:num_each]
    # sub_train_mask_new[selected_indices] = True

    return selected_indices

def compute_rank(tensor, element):
    return np.where(tensor == element)[0][0]

class GcnNet(nn.Module):
    def __init__(self, feature_number, label_number):
        super(GcnNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(feature_number, 16, activation=F.relu))
        self.layers.append(GraphConv(16, label_number))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, features, return_hidden = False):
        relu = nn.ReLU()
        x = F.relu(self.layers[0](g, features))
        if return_hidden:
            return x
        x = self.layers[1](g, x)
        return x


## Main Function
def attack0(dataset_name, seed, cuda, attack_node_arg = 0.25, file_path = '', LR = 1e-3, TGT_LR = 1e-2,
                EVAL_EPOCH = 1000, TGT_EPOCH = 1000, WARMUP_EPOCH = 400, dropout = False, model_performance = True, **kwargs):

    # Initialization
    device = th.device(cuda)
    set_seed(seed)
    metrics_df = pd.DataFrame(columns=['Num Attack Nodes', 'Method', 'Test Accuracy', 'Test Fidelity'])
        
    g, features, labels, node_number, train_mask, test_mask = load_data(dataset_name)
    attack_node_number = int(node_number * attack_node_arg)
    feature_number = features.shape[1]
    label_number = len(labels.unique())
    C_var = label_number

    print('The attack node number is: ', attack_node_number)


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

    ## Send the training to cuda
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

    ## Get the cuda-trained data back
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

    # generate sub-graph adj-matrix, features, labels
    total_sub_nodes = list(set(sub_graph_syn_node_index + sub_graph_node_index))
    sub_g = np.zeros((len(total_sub_nodes), len(total_sub_nodes)))
    for sub_index in range(len(total_sub_nodes)):
        sub_g[sub_index] = g_matrix[total_sub_nodes[sub_index], total_sub_nodes]

    for i in range(node_number):
        if i in sub_graph_node_index:
            test_mask[i] = 0
            train_mask[i] = 1
            continue
        if i in sub_graph_syn_node_index:
            test_mask[i] = 1
            train_mask[i] = 0
        else:
            test_mask[i] = 1
            train_mask[i] = 0

    sub_train_mask = train_mask[total_sub_nodes]

    sub_features = features_query[total_sub_nodes]
    sub_labels = labels[total_sub_nodes]

    sub_features = th.FloatTensor(sub_features)
    sub_labels = th.LongTensor(sub_labels)
    sub_train_mask = sub_train_mask
    sub_test_mask = test_mask
    # sub_g = DGLGraph(nx.from_numpy_matrix(sub_g))

    # features = th.FloatTensor(data.features)
    # labels = th.LongTensor(data.labels)
    # train_mask = th.ByteTensor(data.train_mask)
    # test_mask = th.ByteTensor(data.test_mask)
    # g = DGLGraph(data.graph)

    gcn_Net.eval()

    # =================Generate Label===================================================
    logits_query = gcn_Net(g, features)
    _, labels_query = th.max(logits_query, dim=1)

    sub_labels_query = labels_query[total_sub_nodes]
    sub_g = nx.from_numpy_array(sub_g)

    sub_g.remove_edges_from(nx.selfloop_edges(sub_g))
    sub_g.add_edges_from(zip(sub_g.nodes(), sub_g.nodes()))

    sub_g = dgl.from_networkx(sub_g) # sub_g = DGLGraph(sub_g)
    n_edges = sub_g.number_of_edges()
    # normalization
    degs = sub_g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0

    sub_g.ndata['norm'] = norm.unsqueeze(1)

    print("=========Model Extracting==========================")

    # hyperparameters get from kwargs
    # no need to change these default for now
    num_perturbations = kwargs.get('num_perturbations', 100)
    noise_level = kwargs.get('noise_level', 0.05)
    rho = kwargs.get('rho', 0.8)
    num_each = kwargs.get('num_each', 1)
    epochs_per_cycle = kwargs.get('epochs_per_cycle', 1)
    setup = kwargs.get('setup', "experiment")
    # This need to be relatively bigger to allow for more accurate classification
    if_warmup = kwargs.get('if_warmup', False)
    LR_CEGA = kwargs.get('LR_CEGA', 1e-2)
    # Tuning parameters for adaptive weight in each of the CEGA iteration
    # Default works for cora and amazonphoto and coauthorCS
    # Need specific modification for citeseer and pubmed
    curve = kwargs.get('curve', 0.3)
    init_1 = kwargs.get('init_1', 0.2)
    init_2 = kwargs.get('init_2', 0.2)
    init_3 = kwargs.get('init_3', 0.2)
    gap = kwargs.get('gap', 0.6)

    # Derivative parameters
    num_node = sub_features.shape[0]
    total_epochs = epochs_per_cycle * 18 * C_var
    total_num = 20 * C_var
    num_cycles = total_epochs // epochs_per_cycle

    # Set up adaptive weights: set the numbers then reweight them
    # For citeseer, try k = 0.5, init_1 = 0.3. The other parameters seem to be working fine
    cycles = np.linspace(0, 1, num_cycles)
    index_1 = init_1 + gap * np.exp(-1 * curve * cycles)
    index_2 = init_2 + gap * (1 - np.exp(-1 * curve * cycles))
    index_3 = init_3 * (1 - np.exp(-1 * cycles))
    total = index_1 + index_2 + index_3
    index_1 /= total
    index_2 /= total
    index_3 /= total

    # Set up output data formality
    # data_output = pd.DataFrame(columns=['Num Attack Nodes', 'Method', 'Test Accuracy', 'Test Fidelity'])

    # create GCN model
    max_acc1 = 0
    max_acc2 = 0
    max_f1 = 0
    dur = []

    if dropout == True:
        net = GCN_drop(feature_number, label_number)
    else:
        net = GcnNet(feature_number, label_number)
    optimizer = th.optim.Adam(net.parameters(), lr=LR_CEGA, weight_decay=5e-4)

    ## Set up initial set which is iteratively progressive
    train_inits = init_mask(C_var, sub_train_mask, sub_labels)
    train_inits_tensor = th.tensor(train_inits)
    sub_train_mask_new = th.zeros(len(sub_train_mask), dtype=th.bool)
    sub_train_mask_new[train_inits] = True

    ## Record the initial nodes in torch object
    nodes_queried = th.tensor([], dtype=th.long)
    nodes_queried = th.cat((nodes_queried, train_inits_tensor))

    ## Do warm up if that is ever an option
    if if_warmup == True:
        sub_train_mask_warmup = th.zeros(len(sub_train_mask), dtype=th.bool)
        sub_train_mask_warmup[train_inits] = True
        net.train()

        for epoch in range(WARMUP_EPOCH):
            logits = net(sub_g, sub_features)
            logp = F.log_softmax(logits, dim = 1)

            loss = F.nll_loss(logp[sub_train_mask_warmup], sub_labels_query[sub_train_mask_warmup])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc, f1score = evaluate(net, g, features, labels, test_mask)
            print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test F1 score {:.4f}".format(
                epoch + 1, loss.item(), acc, f1score))

        net.eval()

    ## Now start timing when the real cycles begin
    start_time = time.time()
    log_dir = f"{file_path}/timelogs/{dataset_name}/logtime_cega_{seed}"
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Learn a node in each cycle
    for cycle in range(num_cycles):
        # print(f"=========Cycle {cycle + 1}==========================")
        # print(f"========={int(sub_train_mask_new.sum())} Selected Nodes==========================")

        # Train some epochs:
        net.train()

        for epoch in range(epochs_per_cycle):
            logits = net(sub_g, sub_features)

            ## Need to get new sub_train_mask
            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logp[sub_train_mask_new], sub_labels_query[sub_train_mask_new])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #    if epoch >= 3:
        #        dur.append(time.time() - t0)
            # dur.append(time.time() - t0)

            acc1, _ = evaluate(net, g, features, labels_query, test_mask)
            acc2, f1score = evaluate(net, g, features, labels, test_mask)
            if acc1 > max_acc1:
                max_acc1 = acc1
            if acc2 > max_acc2:
                max_acc2 = acc2
            if f1score > max_f1:
                max_f1 = f1score
            # Add f1 in output
            print("Cycle {:05d} | Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Test Fid {:.4f} | Test F1score {:.4f} ".format(
                cycle + 1, epoch + 1 + cycle * epochs_per_cycle, loss.item(), acc2, acc1, max_f1))

        net.eval()

            ## Not realized here!
            # new_row = {"Epoch": epoch + 1 + cycle * epochs_per_cycle, "Loss": loss.item(), "Accuracy": acc2, "Fidelity": acc1}
            # data_output = data_output.append(new_row, ignore_index = True)
            # data_output.append(new_row)

        # Update the sub_train_mask using your specially-designed algorithm
        if sub_train_mask_new.sum() < total_num:
            # Random
            if setup == "random":
                print("Setup: Random")
                # Add the entry to the node pool nodes_queried on the supposed order
                node_queried = update_sub_train_mask(num_each, sub_train_mask, sub_train_mask_new)
                node_queried_tensor = th.tensor(node_queried)
                # node_queried_tensor = th.tensor(node_queried, dtype = th.long)
                nodes_queried = th.cat((nodes_queried, node_queried_tensor))
                sub_train_mask_new[node_queried] = True

            elif setup == "experiment":
                print("Setup: Experiment")
                ## First: Representativeness
                ## Can be replaced by other centrality measurement
                Rank1 = rank_centrality(sub_g, sub_train_mask, sub_train_mask_new, num_each, return_rank = True)
                ## Second: Uncertainty
                Rank2 = rank_entropy(net, sub_g, sub_features, sub_train_mask, sub_train_mask_new,
                            num_each, return_rank=True)
                ## Third: Diversity
                Rank3 = rank_diversity(net, sub_g, sub_features, sub_train_mask, sub_train_mask_new,
                                       num_each, C_var, rho, return_rank = True)

                if Rank1 is None:
                        print("Completed!")
                selected_indices = quantile_selection(Rank1, Rank2, Rank3, index_1[cycle], index_2[cycle], index_3[cycle],
                                                        sub_train_mask, sub_train_mask_new, num_each)
                selected_indices_tensor = selected_indices.clone().detach()
                # th.tensor(, dtype = th.long)
                nodes_queried = th.cat((nodes_queried, selected_indices_tensor))
                sub_train_mask_new[selected_indices] = True

            elif setup == "perturbation":
                print("Setup: Experiment with Perturbation")
                Rank1 = rank_centrality(sub_g, sub_train_mask, sub_train_mask_new, num_each, return_rank = True)
                Rank2 = rank_perturb(net, sub_g, sub_features, num_perturbations,
                        sub_train_mask, sub_train_mask_new, noise_level,
                        num_each, return_rank = True)
                Rank3 = rank_diversity(net, sub_g, sub_features, sub_train_mask, sub_train_mask_new,
                                       num_each, C_var, rho, return_rank = True)

                if Rank1 is None:
                        print("Completed!")
                selected_indices = quantile_selection(Rank1, Rank2, Rank3, index_1[cycle], index_2[cycle], index_3[cycle],
                                                        sub_train_mask, sub_train_mask_new, num_each)
                selected_indices_tensor = selected_indices.clone().detach()
                nodes_queried = th.cat((nodes_queried, selected_indices_tensor))
                sub_train_mask_new[selected_indices] = True
            else:
                print("Wrong Setup!")
                return 1
        else:
            print("Move on with designated nodes!")
            sub_train_mask_new = sub_train_mask_new

    ## Record time for all these cycles when the loop is complete
    node_selection_time = time.time() - start_time
    with open(log_dir, 'a') as log_file:
        log_file.write(f"CEGA {dataset_name} {seed} ")
        log_file.write(f"{node_selection_time:.4f}s\n")

    idx_train = nodes_queried.tolist()

    output_data = {
        'total_sub_nodes': total_sub_nodes,
        'idx_train': idx_train
    }

    ## Assertation and printing
    assert len(idx_train) == 20 * C_var
    print('node selection finished')
    with open(f'./node_selection/CEGA_{setup}_{dataset_name}_selected_nodes_{(20*label_number)}_{seed}.json','w')as f:
        json.dump(output_data, f)


    sub_g = sub_g.to(device)
    sub_features = sub_features.to(device)
    sub_labels_query = sub_labels_query.to(device)
    labels_query = labels_query.to(device)
    g = g.to(device)
    features = features.to(device)
    test_mask = test_mask.to(device)
    labels = labels.to(device)

    print('=========Model Evaluation==========================')
    if model_performance:
        for iter in range(2 * C_var, 21 * C_var, C_var):
            set_seed(seed)

            ## Create net from scratch
            if dropout == True:
                net_scratch = GCN_drop(feature_number, label_number)
            else:
                net_scratch = GcnNet(feature_number, label_number)
            optimizer = th.optim.Adam(net_scratch.parameters(), lr=LR, weight_decay=5e-4)

            ## set up training nodes and send them to device
            sub_train_scratch = th.zeros(sub_features.size()[0], dtype=th.bool)
            sub_train_scratch[idx_train[:iter]] = True
            sub_train_scratch = sub_train_scratch.to(device)
            net_scratch = net_scratch.to(device)

            ## Reset data
            max_acc1 = 0
            max_acc2 = 0
            max_f1 = 0
            dur = []

            for epoch in range(EVAL_EPOCH):
                if epoch >= 3:
                    t0 = time.time()

                net_scratch.train()
                logits = net_scratch(sub_g, sub_features)
                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp[sub_train_scratch], sub_labels_query[sub_train_scratch])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch >= 3:
                    dur.append(time.time() - t0)

                acc1, _ = evaluate(net_scratch, g, features, labels_query, test_mask)
                acc2, f1score = evaluate(net_scratch, g, features, labels, test_mask)
                if acc1 > max_acc1:
                    max_acc1 = acc1
                if acc2 > max_acc2:
                    max_acc2 = acc2
                if f1score > max_f1:
                    max_f1 = f1score

            # Output Epoch Scores
            epoch_metrics = pd.DataFrame({
                'Num Attack Nodes': [iter],
                'Method': ['CEGA'],
                'Test Accuracy': [max_acc2],
                'Test Fidelity': [max_acc1],
                'Test F1score': [max_f1],
            })
            metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)

            print("Test Acc {:.4f} | Test Fid {:.4f} | Test F1score {:.4f} | Time(s) {:.4f}".format(
                acc2, acc1, max_f1, np.mean(dur)))

        ## Should this be 'f1score'?
        epoch_metrics = pd.DataFrame({
            'Num Attack Nodes': [int(th.sum(train_mask))],
            'Method': ['CEGA'],
            'Test Accuracy': [target_performance['acc']],
            'Test Fidelity': [1],
            'Test F1score': [target_performance['f1score']],
        })
        metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)

        log_file_path = f"{file_path}/{dataset_name}/log_cega_{seed}.csv"
        metrics_df.to_csv(log_file_path, mode='w', header=False, index=False)

    # Set net_full for the next graph to be taken care of, which is expected to include all nodes
    if True:
        set_seed(seed)
        log_file_path = f"{file_path}/{dataset_name}/log_cega_{seed}.csv"
        if dropout == True:
            net_full = GCN_drop(feature_number, label_number)
        else: 
            net_full = GcnNet(feature_number, label_number)
        optimizer_full = th.optim.Adam(net_full.parameters(), lr=LR, weight_decay=5e-4)

        net_full = net_full.to(device)
        net = net.to(device)

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
        epoch_metrics = pd.DataFrame({
            'Num Attack Nodes': [sub_train_mask.sum().item()],
            'Method': ['cega'],
            'Test Accuracy': [perfm_attack['acc']],
            'Test Fidelity': [perfm_attack['fid']],
            'Test F1score': [perfm_attack['f1score']],
        }) 
        metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)
        log_file_path = f"{file_path}/{dataset_name}/log_cega_{seed}.csv"
        metrics_df.to_csv(log_file_path, mode='w', header=False, index=False)
        

    if False:
        ## Set up the evaluation of accuracy gap after playing on the full model
        metrics_df = pd.DataFrame()

        ## Be careful about here; maybe not the best for such range
        ## May be range(2 * C_var, 21 * C_var, C_var) instead for everytime we evaluate
        for iter_compare in tqdm(range(2 * C_var, 21 * C_var, 2), desc='Model Eval'):
            set_seed(seed)
            if dropout:
                net_eval = GCN_drop(feature_number, label_number)
            else:
                net_eval = GcnNet(feature_number, label_number)

            optimizer = th.optim.Adam(net_eval.parameters(), lr=LR, weight_decay=5e-4)

            # Recover sub_train_mask and send the terms to device
            sub_train_eval = th.zeros(sub_features.size()[0], dtype=th.bool)
            sub_train_eval[idx_train[:iter_compare]] = True
            sub_train_eval = sub_train_eval.to(device)
            net_eval = net_eval.to(device)

            max_acc2 = 0

            for epoch in range(EVAL_EPOCH):
                net_eval.train()
                logits = net_eval(sub_g, sub_features)
                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp[sub_train_eval], sub_labels_query[sub_train_eval])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc2, _ = evaluate(net_eval, g, features, labels, test_mask)
                if acc2 > max_acc2:
                    max_acc2 = acc2

            ## I have not seen where this "diff" has been used elsewhere
            diff = abs(1 - max_acc2)

            ## Why we need 'Num Attack Nodes' and 'Attack node Num' in the same dataframe? what is their difference?
            ## Strongly recommend editing the name if possible
            epoch_metrics = pd.DataFrame({
                'Num Attack Nodes': [iter_compare],
                'Accuracy': [max_acc2],
                'Accuracymax': [perfm_attack['acc']],
                'Method': ['CEGA'],
                'Attack node Num': [attack_node_number],
            })
            metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)

        log_file_path = f"{file_path}/gaps/{dataset_name}/log_CEGA_{seed}.csv"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        metrics_df.to_csv(log_file_path, mode='w', header=True, index=False)




