import time
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.neighbors import KNeighborsClassifier

import sys
sys.path.append("../../model")
from models import GCN
from propagation import PPRPowerIteration
from data_loader import load_data


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
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
parser.add_argument('--p', type=int, default=10,
                            help='Number of propagation.')
parser.add_argument('--alpha', type=float, default=0.1,
                            help='Teleport strength.')
parser.add_argument('--delta', type=int, default=8,
                    help='node degree setting in MST-KNN graph')
parser.add_argument('--base', type=str, default='google',
                    help='base embedding: self, selfhf, google, glove, fast')
parser.add_argument('--aff', type=str, default='aff',
                    help='affinity info: aff, google, glove, fast')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def train(model, optimizer, epoch, idx_train, labels, batch_sz = 32):
    t = time.time()
    model.train()
    mse_loss = nn.MSELoss(reduction = "mean")
    permutation = torch.randperm(idx_train.size()[0])
    loss = 0
    for i in range(0, idx_train.size()[0], batch_sz):
        indices = idx_train[permutation[i: (i + batch_sz)]]
        output = model(features, indices)
        optimizer.zero_grad()
        loss_train = mse_loss(output, labels[indices])
        loss_train.backward()
        optimizer.step()
        loss += loss_train.item()
    '''
    print('Epoch: {:04d}'.format(epoch+1),
          'Training Loss: {:.4f}'.format(loss),
          'Time: {:.4f}s'.format(time.time() - t))
    '''

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
    print("p: ", args.p)
    print("alpha: ", args.alpha)
    print("base: ", args.base)
    print("delta: ", args.delta)


if __name__ == "__main__":
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    show_params(args)
    # Build a word graph using the affinity matrix
    adj, features, labels, y, idx_train = load_data(args.aff, args.base, args.delta)
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)
    #adj = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)

    propagator = PPRPowerIteration(adj, args.alpha, args.p)

    # Set up the GCN Model and the optimizer
    model = GCN(nfeatures=features.shape[1],
                hiddenunits=[args.hidden],
                nout=labels.shape[1],
                drop_prob=args.dropout,
                propagation=propagator)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        labels = labels.cuda()
        #adj = adj.cuda()
        idx_train = idx_train.cuda()

    t_total = time.time()
    for epoch in range(args.epochs):
        train(model, optimizer, epoch, idx_train, labels, args.batch_size)

    print(f"Training Finished! Training time : {time.time() - t_total}s")

    idx_test = np.array(range(labels.shape[0],features.shape[0]))
    idx_test = torch.LongTensor(idx_test)
    result = test(model, features, labels, y, idx_test)
    print(result, '\n')
