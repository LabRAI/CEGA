import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import random
import copy
import sys
import os

import numpy.linalg as la
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import f1_score

import dgl
from dgl.nn.pytorch import GraphConv
from dgl.data import citation_graph as citegrh
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorCSDataset, CoauthorPhysicsDataset, RedditDataset, WikiCSDataset, AmazonRatingsDataset, QuestionsDataset, RomanEmpireDataset, FlickrDataset, CoraFullDataset, CitationGraphDataset
from torch_geometric.datasets import CitationFull


# from graphConvolution import *

def get_receptive_fields_dense(cur_neighbors, selected_node, weighted_score, adj_matrix2): 
    receptive_vector = ((cur_neighbors + adj_matrix2[selected_node]) != 0) + 0
    count = weighted_score.dot(receptive_vector)
    return count

def get_current_neighbors_dense(cur_nodes, adj_matrix2):
    if np.array(cur_nodes).shape[0] == 0:
        return 0
    neighbors = (adj_matrix2[list(cur_nodes)].sum(axis=0) != 0) + 0
    return neighbors

def get_current_neighbors_1(cur_nodes, adj_matrix):
    if np.array(cur_nodes).shape[0]==0:
        return 0
    neighbors=(adj_matrix[list(cur_nodes)].sum(axis=0)!=0)+0
    return neighbors

def get_entropy_contribute(npy_m1, npy_m2):
    entro1 = 0
    entro2 = 0
    for i in range(npy_m1.shape[0]):
        entro1 -= np.sum(npy_m1[i]*np.log2(npy_m1[i]))
        entro2 -= np.sum(npy_m2[i]*np.log2(npy_m2[i]))
    return entro1 - entro2

def get_max_info_entropy_node_set(idx_used, 
                                  high_score_nodes,
                                  labels,
                                  batch_size,
                                  adj_matrix2,
                                  num_class,
                                  model_prediction):
    max_info_node_set = [] 
    high_score_nodes_ = copy.deepcopy(high_score_nodes)
    labels_ = copy.deepcopy(labels)
    for k in range(batch_size):
        score_list = np.zeros(len(high_score_nodes_))      
        for i in range(len(high_score_nodes_)):
            labels_tmp = copy.deepcopy(labels_)          
            node = high_score_nodes_[i]
            node_neighbors = get_current_neighbors_dense([node],adj_matrix2)
            adj_neigh = adj_matrix2[list(node_neighbors)]
            aay = np.matmul(adj_neigh,labels_)
            total_score = 0
            for j in range(num_class):
                if model_prediction[node][j] != 0:
                    labels_tmp[node] = 0
                    labels_tmp[node][j] = 1
                    aay_ = np.matmul(adj_neigh,labels_tmp)
                    total_score += model_prediction[node][j]*get_entropy_contribute(aay,aay_)
            score_list[i] = total_score
        idx = np.argmax(score_list)
        max_node = high_score_nodes_[idx]
        max_info_node_set.append(max_node)
        labels_[max_node] = model_prediction[max_node]
        high_score_nodes_.remove(max_node)   
    return max_info_node_set

def get_max_nnd_node_dense(idx_used, 
                           high_score_nodes, 
                           min_distance,
                           distance_aax,
                           num_ones,
                           num_node,
                           adj_matrix2,
                           gamma = 1):
     
    dmax = np.ones(num_node)

    max_receptive_node = 0
    max_total_score = 0
    cur_neighbors = get_current_neighbors_dense(idx_used, adj_matrix2)
    for node in high_score_nodes:
        receptive_field = get_receptive_fields_dense(cur_neighbors, node, num_ones, adj_matrix2)
        node_distance = distance_aax[node, :]
        node_distance = np.where(node_distance < min_distance, node_distance, min_distance)
        node_distance = dmax - node_distance
        distance_score = node_distance.dot(num_ones)
        total_score = receptive_field / num_node + gamma * distance_score / num_node
        if total_score > max_total_score:
            max_total_score = total_score
            max_receptive_node = node        
    return max_receptive_node

def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def compute_distance(_i, _j, features_aax):
    return la.norm(features_aax[_i, :] - features_aax[_j, :])

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csgraph
import sys
import time
import argparse
import torch
import random


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_data_from_grain(path="./data", dataset="cora"):
    """
    ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)
    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)
    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    """
    print("\n[STEP 1]: Upload {} dataset.".format(dataset))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Citeseer dataset contains some isolated nodes in the graph
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    print("| # of nodes : {}".format(adj.shape[0]))
    print("| # of edges : {}".format(adj.sum().sum() / 2))

    features = normalize(features)
    print("| # of features : {}".format(features.shape[1]))
    print("| # of clases   : {}".format(ally.shape[1]))

    features = torch.FloatTensor(np.array(features.todense()))
    sparse_mx = adj.tocoo().astype(np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if dataset == 'citeseer':
        save_label = np.where(labels)[1]
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()

    print("| # of train set : {}".format(len(idx_train)))
    print("| # of val set   : {}".format(len(idx_val)))
    print("| # of test set  : {}".format(len(idx_test)))

    idx_train, idx_val, idx_test = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test]))

    def missing_elements(L):
        start, end = L[0], L[-1]
        return sorted(set(range(start, end + 1)).difference(L))

    if dataset == 'citeseer':
        L = np.sort(idx_test)
        missing = missing_elements(L)

        for element in missing:
            save_label = np.insert(save_label, element, 0)

        labels = torch.LongTensor(save_label)

    return adj, features, labels, idx_train, idx_val, idx_test

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def aug_normalized_adjacency(adj):
   adj = adj # + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def aug_random_walk(adj):
   adj = adj + sp.eye(adj.shape[0])
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv = np.power(row_sum, -1.0).flatten()
   d_mat = sp.diags(d_inv)
   return (d_mat.dot(adj)).tocoo()


class GCN_drop(nn.Module):
    def __init__(self, feature_number, label_number, dropout = 0.85, nhid = 128):
        super(GCN_drop, self).__init__()

        self.gc1 = GraphConv(feature_number, nhid, bias=True)
        self.gc2 = GraphConv(nhid, label_number, bias=True)
        self.dropout = dropout

    def forward(self, g, features):
        x = F.dropout(features, self.dropout, training=self.training)
        x = F.relu(self.gc1(g, x)) 
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(g, x)
        return x

def convert_pyg_to_dgl(pyg_data):
    """
    Converts a PyTorch Geometric Data object into a DGLGraph.

    Args:
        pyg_data (torch_geometric.data.Data): PyTorch Geometric Data object.

    Returns:
        dgl.DGLGraph: The converted DGL graph.
    """
    edge_index = pyg_data.edge_index
    num_nodes = pyg_data.num_nodes

    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)

    if hasattr(pyg_data, 'x') and pyg_data.x is not None:
        g.ndata['feat'] = pyg_data.x

    if hasattr(pyg_data, 'y') and pyg_data.y is not None:
        g.ndata['label'] = pyg_data.y

    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        if hasattr(pyg_data, mask_name) and getattr(pyg_data, mask_name) is not None:
            g.ndata[mask_name] = getattr(pyg_data, mask_name)

    return g


def load_data(dataset_name):
    if dataset_name == 'cora':
        data = citegrh.load_cora()
    if dataset_name == 'citeseer':
        data = citegrh.load_citeseer()
    if dataset_name == 'pubmed':
        data = citegrh.load_pubmed()
    if dataset_name == 'amazoncomputer':
        data = AmazonCoBuyComputerDataset()
    if dataset_name == 'amazonphoto':
        data = AmazonCoBuyPhotoDataset()
    if dataset_name == 'coauthorCS':
        data = CoauthorCSDataset()
    if dataset_name == 'coauthorphysics':
        data = CoauthorPhysicsDataset()
    if dataset_name == 'reddit':
        data = RedditDataset()
    if dataset_name == 'wiki':
        data = WikiCSDataset()
    if dataset_name == 'amazonrating':
        data = AmazonRatingsDataset()
    if dataset_name == 'question':
        data = QuestionsDataset()
    if dataset_name == 'roman':
        data = RomanEmpireDataset()
    if dataset_name == 'flickr':
        data = FlickrDataset()
    if dataset_name == 'cora_full':
        data = CoraFullDataset()
    if dataset_name == 'dblp':
        data = CitationFull(root='./data/',name='DBLP')
        data = data[0]
        
    if dataset_name == 'dblp':
        g = convert_pyg_to_dgl(data)
    else: 
        g = data[0] 
        
    isolated_nodes = ((g.in_degrees() == 0) & (g.out_degrees() == 0)).nonzero().squeeze(1)
    g.remove_nodes(isolated_nodes)
    
    if dataset_name in ['cora', 'citeseer', 'pubmed', 'reddit', 'flickr']:
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        test_mask = g.ndata['test_mask']
        num_nodes = g.num_nodes()
    elif dataset_name in ['wiki']:
        features = g.ndata['feat']
        labels = g.ndata['label']
        test_mask = g.ndata['test_mask'].bool()
        train_mask = (1-g.ndata['test_mask']).bool() #
        num_nodes = g.num_nodes()        
    elif dataset_name in ['amazoncomputer', 'amazonphoto', 'coauthorCS', 'coauthorphysics', 'cora_full','dblp']:
        features = g.ndata['feat']
        labels = g.ndata['label']
        num_nodes = g.num_nodes()
        train_mask = torch.zeros(num_nodes, dtype = torch.bool)
        test_mask = torch.zeros(num_nodes, dtype = torch.bool)
        
        torch.manual_seed(42)
        indices = torch.randperm(num_nodes)
        num_train = int(num_nodes * 0.6)
        train_mask[indices[:num_train]] = True
        test_mask[indices[num_train:]] = True
        assert train_mask.sum() + test_mask.sum() == num_nodes
    elif dataset_name in ['amazonrating', 'question', 'roman']:
        features = g.ndata['feat']
        labels = g.ndata['label']
        num_nodes = g.num_nodes()  
        train_mask = g.ndata['train_mask'][:, 0]
        test_mask = g.ndata['test_mask'][:, 0]     
    return g, features, labels, num_nodes, train_mask, test_mask

def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        f1score = f1_score(labels.cpu().numpy(), indices.cpu().numpy(), average='macro')
        return correct.item() * 1.0 / len(labels), f1score


class GcnNet(nn.Module):
    def __init__(self, feature_number, label_number):
        super(GcnNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(feature_number, 16, activation=F.relu))
        self.layers.append(GraphConv(16, label_number))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, g, features):
        x = F.relu(self.layers[0](g, features))
        x = self.layers[1](g, x)
        return x
