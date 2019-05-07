from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from layers import GraphConvolution
from graphs import Graph

N = 15
H = 1000

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.fc1 = nn.Linear(N*N, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, N)
        self.drop = nn.Dropout(0.3)

    def forward(self, x, adj):
        y = adj.view(-1)
        y = self.drop(F.relu(self.fc1(y)))
        y = self.drop(F.relu(self.fc2(y)))
        y = self.drop(F.relu(self.fc3(y)))
        y = self.fc4(y)
        return y


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--n_train', type=int, default=200,
                    help='Number of train samples.')
parser.add_argument('--n_valid', type=int, default=200,
                    help='Number of valid samples.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=100,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#adj, features, labels, idx_train, idx_val, idx_test = load_data()

nfeat = 1

print('Generating train set...')
train_graphs, valid_graphs = [], []
for it in range(args.n_train):
    labeled = True if it % 4 == 0 else False
    m = np.random.randint(N-1, int(N*(N-1)/2+1))
    train_graphs.append((labeled, Graph.gen_random_graph(N, m)))

print('Generating valid set...')
for it in range(args.n_valid):
    m = np.random.randint(N-1, int(N*(N-1)/2+1))
    valid_graphs.append(Graph.gen_random_graph(N, m))

model = GCN(nfeat=nfeat,
            nhid=args.hidden,
            nclass=1,
            dropout=args.dropout)

# Model and optimizer
# model = GCN(nfeat=features.shape[1],
#             nhid=args.hidden,
#             nclass=labels.max().item() + 1,
#             dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                      lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model=model.cuda()

def train(epoch):
    tot_err, tot_nql_loss, tot_lab, tot_unlab = 0, 0, 0, 0
    random.shuffle(train_graphs)
    for i, (labeled, g) in enumerate(train_graphs):
        idx = torch.LongTensor([g.x, g.y])
        v = torch.FloatTensor(np.ones(len(g.x)))
        adj = torch.sparse.FloatTensor(idx, v, torch.Size([g.n, g.n])).to_dense()

        model.train()
        features = -torch.ones((g.n, nfeat))
        features[0,0] = 0.0

        if args.cuda:
            features = features.cuda()
            adj = adj.cuda()

        out = model.forward(features, adj)

        optimizer.zero_grad()

        dist = torch.FloatTensor([g.p[0,i] for i in range(g.n)])
        if args.cuda:
            dist = dist.cuda()
        err = torch.mean((dist - out) * (dist - out))
        tot_err += err
        tot_lab += 1

        nql_loss = 0
        for a in range(1,g.n):
            terms = 1.0
            for b in range(g.n):
                if adj[a, b]:
                    terms *= torch.abs(out[a] - (out[b] + 1))
                    nql_loss += torch.clamp(out[a] - (out[b] + 1), min=0)

            nql_loss += terms
        nql_loss += torch.abs(out[0])
        for a in range(0,g.n):
            nql_loss += torch.clamp(-out[a], min=0)

        tot_nql_loss += nql_loss
        tot_unlab += 1

        tot_loss = nql_loss
        tot_loss.backward()

        # for p in model.parameters():
        #     print('norm grad: ',torch.norm(p.grad))

        optimizer.step()

    # print('Average error: ',tot_err/float(tot_lab))
    # print('Average NQL: ', nql_loss / float(tot_unlab))


def test(epoch):
    model.eval()
    tot_err = 0
    baseline_err = 0
    all_ones = torch.ones(N)

    for i, g in enumerate(valid_graphs):
        idx = torch.LongTensor([g.x, g.y])
        v = torch.FloatTensor(np.ones(len(g.x)))
        adj = torch.sparse.FloatTensor(idx, v, torch.Size([g.n, g.n])).to_dense()

        model.eval()
        features = -torch.ones((g.n, nfeat))
        features[0,0] = 0.0

        if args.cuda:
            features = features.cuda()
            adj = adj.cuda()

        out = model.forward(features, adj)
        dist = torch.FloatTensor([g.p[0,i] for i in range(g.n)])
        if args.cuda:
            dist = dist.cuda()
            all_ones = all_ones.cuda()

        err = torch.mean((dist - out) * (dist - out))
        baseline_err += torch.mean((dist - all_ones) * (dist - all_ones))
        tot_err += err

    print(epoch,' [Valid] Average error: ',tot_err/float(len(valid_graphs)))
    #print(baseline_err/float(len(valid_graphs)))

# Train model
t_total = time.time()
for epoch in range(1, args.epochs):
    train(epoch)
    if epoch % 10 == 0:
        test(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
#test()
