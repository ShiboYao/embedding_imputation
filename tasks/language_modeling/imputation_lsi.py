import numpy as np
import pandas as pd
import sys
sys.path.append("../../model")
from lsi import iterSolveQ
from data_loader import load_data
from sklearn.neighbors import KNeighborsClassifier
import os

if __name__ == '__main__':

    basename = sys.argv[1]
    affname = sys.argv[2]
    delta = int(sys.argv[3])
    save_path = sys.argv[4]
    np.random.seed(int(sys.argv[5]))
    seed = int(sys.argv[5])
    corpus_size = int(sys.argv[6])
    
    base_path = "data/sampled_{}/{}_embeds.txt".format(corpus_size, basename)
    aff_path = "data/sampled_{}/{}_embeds.txt".format(corpus_size, affname)
    adj, features, labels, idx_train, Pind, Qind, PMat = load_data(aff_path, base_path, delta, basename, corpus_size)
    PQ = iterSolveQ(labels[idx_train], adj, 1e-3)
    print(PQ.shape, len(Pind), len(Qind))
    
    embeds = pd.concat([pd.DataFrame(Pind + Qind), pd.DataFrame(PQ)], axis = 1)
    embeds.to_csv(os.path.join(save_path,'LSI_{}_embeds_{}_{}.txt'.format(basename, delta, seed)), index=False, header=False, sep = ' ')
    print(embeds[:5])