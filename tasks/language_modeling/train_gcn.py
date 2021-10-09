import time
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.neighbors import KNeighborsClassifier

import os
import sys
sys.path.append("../../model")
from models import GCN
from propagation import PPRPowerIteration
from data_loader import load_data
import pandas as pd
from GCN_model import GCN_2017, sparse_mx_to_torch_sparse_tensor

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=600,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=400, help='batch size')
parser.add_argument('--delta', type=int, default=8,
                    help='node degree setting in MST-KNN graph')
parser.add_argument('--base', type=str, default='fast',
                    help='base embedding: self, selfhf, google, glove, fast')
parser.add_argument('--aff', type=str, default='bioword',
                    help='affinity info: aff, google, glove, fast')
parser.add_argument('--corpus_size', type=int, default=30000,
                    help='for saving purposes')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

base_path = "data/sampled_{}/{}_embeds.txt".format(args.corpus_size, args.base)
aff_path = "data/sampled_{}/{}_embeds.txt".format(args.corpus_size, args.aff)
save_path = "data/sampled_{}/GCN_embeds".format(args.corpus_size)

def train(model, optimizer, epoch, adj, idx_train, labels, batch_sz = 32):
    t = time.time()
    model.train()
    mse_loss = nn.MSELoss(reduction = "mean")
    permutation = torch.randperm(idx_train.size()[0])
    loss = 0
    for i in range(0, idx_train.size()[0], batch_sz):
        indices = idx_train[permutation[i: (i + batch_sz)]]
        output = model(features, adj, indices)
        optimizer.zero_grad()
        loss_train = mse_loss(output, labels[indices])
        loss_train.backward()
        optimizer.step()
        loss += loss_train.item()
    
    print('Epoch: {:04d}'.format(epoch+1),
          'Training Loss: {:.4f}'.format(loss),
          'Time: {:.4f}s'.format(time.time() - t))
    

def KNN(X, y, n):
    l = len(y)
    y_hat = []
    for i in range(l):
        X_train = np.delete(X, i, axis = 0)
        y_train = np.delete(y, i, axis = 0)
        neigh = KNeighborsClassifier(n_neighbors = n)
        neigh.fit(X_train, y_train)
        y_hat.extend(neigh.predict(X[i].reshape(1,-1)))
    acc = sum(np.array(y_hat) == y) / l
    return acc


def test(model, features, labels, y, idx_test):
    model.eval()
    with torch.no_grad():
        pred = model(features, idx_test)
        N = [2,5,8,10,15,20,30]
        X = torch.cat((labels,pred), dim=0)
        X = X.cpu().detach().numpy()
        result = []
        for n in N: # classification n_neighbors = 5, n_components = 30 up
            result.append(KNN(X.copy(), y.copy(), n))
    return result


def show_params(args):
    print("hidden: ", args.hidden)
    print("epochs: ", args.epochs)
    print("base: ", args.base)
    print("delta: ", args.delta)


if __name__ == "__main__":
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    show_params(args)
    # Build a word graph using the affinity matrix
    adj, features, labels, idx_train, Pind, Qind, PMat = load_data(aff_path, base_path, args.delta, args.base, args.corpus_size)
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)

    # Set up the GCN Model and the optimizer
    model = GCN_2017(nfeat=features.shape[1],
                nhid= args.hidden,
                nout= labels.shape[1],
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        labels = labels.cuda()
        adj = adj.cuda()
        idx_train = idx_train.cuda()

    t_total = time.time()
    for epoch in range(args.epochs):
        train(model, optimizer, epoch, adj, idx_train, labels, args.batch_size)

    print(f"Training Finished! Training time : {time.time() - t_total}s")

    idx_test = np.array(range(labels.shape[0],features.shape[0]))
    idx_test = torch.LongTensor(idx_test)
    model.eval()
    with torch.no_grad():
        Q = model.forward(features, adj, idx_test).cpu().numpy()
        P = labels.cpu().numpy()
    PQ = np.r_[P,Q]
    
    print(PQ.shape, len(Pind), len(Qind))
    embeds = pd.concat([pd.DataFrame(Pind + Qind), pd.DataFrame(PQ)], axis = 1)
    embeds.to_csv(os.path.join(save_path,'GCN_{}_embeds_{}_{}.txt'.format(args.base, args.delta, args.seed)), index=False, header=False, sep = ' ')
    print(embeds[:5])
