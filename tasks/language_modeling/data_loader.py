import pandas as pd
import scipy.sparse as sp
import sys
sys.path.append("../../model")
sys.path.append("../../")
from hpc import multicore_dis, MSTKNN, multicore_nnls
from utils import *
import multiprocessing as mp


def load_data(aff, semantic, delta, ita=1e-4, spanning=True, n_jobs=-1):
    """Build the adjacency matrix, node features and ids dictionary."""
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu

    labels = pd.read_csv("./data/"+semantic+".txt", sep=' ', index_col=0, header=None)
    features = pd.read_csv("./data/"+aff+".txt", sep=' ', index_col=0, header = None)
    features, labels, semanticOuter = permuteMat(features, labels)
    Q_index = range(features.shape[0])
    dis = multicore_dis(features.values, Q_index, n_jobs)
    graph = MSTKNN(dis, Q_index, delta, n_jobs, spanning)
    adj = multicore_nnls(features.values, graph, Q_index, n_jobs, epsilon=1e-1)
    
    idx_train = np.array(range(labels.shape[0]))
    
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = features.values
    labels = labels.values

    return adj, features, labels, idx_train