import pandas as pd
import scipy.sparse as sp
import torch
import sys
sys.path.append("../../model")
sys.path.append("../../")
from hpc import multicore_dis, MSTKNN, multicore_nnls
from anchor import getAnchorIndex, distanceAnchorEuclidean, anchorKNN
from utils import *
import multiprocessing as mp


def load_data(aff, semantic, delta, anchor=True, m=200, spanning=True, n_jobs=-1):
    """Build the adjacency matrix, node features and ids dictionary."""
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu

    labels = pd.read_csv("../../data/regression/"+semantic+".txt", delimiter=' ', index_col=0, header=None)
    features = pd.read_csv("../../data/regression/"+aff+".txt", delimiter=' ', index_col=0, header=None)
    words = features.index.values
    labels = labels.values
    features = features.values

    idx = list(range(len(labels)))
    np.random.shuffle(idx)
    if anchor:
        
        idx_train = idx[:1000]
        idx_valid = idx[1000:2000]
        idx_test = idx[2000:]
        '''
        idx = idx[:5000]
        idx_train = list(range(1000))
        idx_valid = list(range(1000,2000))
        idx_test = list(range(2000,5000))
        labels = labels[idx] #subsampling
        features = features[idx]
        words = words[idx] #for comparing approximated sol and exact sol
        '''
        Q_index = range(features.shape[0])
        #anchor_index = getAnchorIndex(1000,m) 
        anchor_index = getAnchorIndex(len(labels),m)
        dis = distanceAnchorEuclidean(features, Q_index, anchor_index, n_jobs)
        graph = anchorKNN(dis, delta, Q_index, anchor_index, n_jobs)
    else:
        idx = idx[:5000]
        idx_train = list(range(1000))
        idx_valid = list(range(1000,2000))
        idx_test = list(range(2000,5000))

        labels = labels[idx] #subsampling
        features = features[idx]
        words = words[idx]

        Q_index = range(features.shape[0])
        dis = multicore_dis(features, Q_index, n_jobs)
        graph = MSTKNN(dis, Q_index, delta, n_jobs, spanning)
        #graph = MSTKNN(dis, Q_index, delta, n_jobs, False)
    adj = multicore_nnls(features, graph, Q_index, n_jobs, epsilon=1e-1)

    #adj = sym_normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    #adj = poly_adj(adj, 4)
    #adj = preprocess_high_order_adj(adj,5,1e-4)
    #adj = Linv(adj)

    return adj, features, labels, idx_train, idx_valid, idx_test, words

