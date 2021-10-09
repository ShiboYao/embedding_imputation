import pandas as pd
import scipy.sparse as sp
import sys
sys.path.append("../../../model")
sys.path.append("../../")
from hpc import multicore_dis, MSTKNN, multicore_nnls
from utils import *
import multiprocessing as mp


def permuteMat2(aff, semantic): #permute affinity mat and semantic mat
    affInd = aff.index.tolist()
    semanticInd = semantic.index.tolist() #instead of index.values.tolist()
    Pind = [i for i in semanticInd if i in affInd]
    Qind = [i for i in affInd if i not in Pind]

    PMat = aff.loc[Pind].copy()
    QMat = aff.loc[Qind].copy()

    aff = pd.concat([PMat, QMat], axis=0)
    semanticInter = semantic.loc[Pind].copy()
    semanticOuter = semantic.drop(labels=Pind, axis=0)

    return aff, semanticInter, semanticOuter, Pind, Qind, PMat


def load_data(aff, semantic, delta, base=None, corpus_size = 20000, ita=1e-4, spanning=True, n_jobs=-1):
    """Build the adjacency matrix, node features and ids dictionary."""
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu
    
    # pandas converts the string 'null' to NaN if we don't set keep_default_na=False
    labels = pd.read_csv(semantic, delimiter = ' ', index_col=0, header = None, keep_default_na=False)
    features = pd.read_csv(aff, delimiter = ' ', index_col=0, header = None, keep_default_na=False)
    features, labels, semanticOuter, Pind, Qind, PMat = permuteMat2(features, labels)
    Q_index = range(features.shape[0])
    
    if base is None:
        dis = multicore_dis(features.values, Q_index, n_jobs)
        graph = MSTKNN(dis, Q_index, delta, n_jobs, spanning)
        adj = multicore_nnls(features.values, graph, Q_index, n_jobs, epsilon=1e-1)
        adj = normalize(adj + sp.eye(adj.shape[0]))
    else:
        adj = sp.load_npz('./data/sampled_{}/graphs/adj_{}_{}.npz'.format(corpus_size, base, delta))
    
    idx_train = np.array(range(labels.shape[0]))
    features = features.values
    labels = labels.values

    return adj, features, labels, idx_train, Pind, Qind, PMat