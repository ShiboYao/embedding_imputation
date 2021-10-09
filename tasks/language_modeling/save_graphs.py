import pandas as pd
import scipy.sparse as sp
import sys
sys.path.append("../../model")
sys.path.append("../../../model")
sys.path.append("../../")
from hpc import multicore_dis, MSTKNN, multicore_nnls
from utils import *
import multiprocessing as mp
from data_loader import load_data

for base in ['glove','fast','google']:
    print(base)
    adj, features, labels, idx_train, Pind, Qind, PMat = load_data('data/sampled_30000/bioword_embeds.txt',
                                                                   'data/sampled_30000/{}_embeds.txt'.format(base), 8)
    sp.save_npz('./data/sampled_30000/graphs/adj_{}_8.npz'.format(base), adj)
    
#sp.load_npz('./data/sampled_50000/graphs/adj_google_8.npz')