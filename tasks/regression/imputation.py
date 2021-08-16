import numpy as np
import sys 
sys.path.append("../../model")
from lsi import iterSolveQ
from data_loader import load_data
from sklearn.metrics import mean_squared_error as mse


if __name__ == '__main__':
    if (len(sys.argv) != 5): 
        print("Specify base, aff delta and seed!")
        exit(0)

    basename = sys.argv[1]
    affname = sys.argv[2]
    delta = int(sys.argv[3])
    np.random.seed(int(sys.argv[4]))

    adj, features, labels, idx_train, idx_valid, idx_test, words = load_data(affname, basename, delta)
    PQ = iterSolveQ(labels[idx_train], adj, 1e-3)
    print("MSE: ", mse(labels[idx_test], PQ[idx_test]))
