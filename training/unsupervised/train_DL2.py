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

import sys
sys.path.append('pygcn/pygcn')
from utils import load_data, accuracy
from layers import GraphConvolution
from graphs import Graph
sys.path.append('../../')
import dl2lib as dl2

class GCN(nn.Module):
    def __init__(self, nclass, N, H, dropout=0.3):
        super(GCN, self).__init__()

        self.fc1 = nn.Linear(N * N, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)
        self.fc4 = nn.Linear(H, N)
        self.drop = nn.Dropout(dropout)

    def forward(self, adj):
        y = adj.view(-1)
        y = self.drop(F.relu(self.fc1(y)))
        y = self.drop(F.relu(self.fc2(y)))
        y = self.drop(F.relu(self.fc3(y)))
        y = self.fc4(y)
        return y


# Training settings
parser = argparse.ArgumentParser()
parser = dl2.add_default_parser_args(parser)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=60000,
                    help='Number of epochs to train.')
parser.add_argument('--n_train', type=int, default=300,
                    help='Number of train samples.')
parser.add_argument('--n_valid', type=int, default=150,
                    help='Number of valid samples.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=1000,
                    help='number of units in hidden layers')
parser.add_argument('-n', type=int, default=15,
                    help='number of nodes in the graph')
parser.add_argument('--baseline', type=dl2.str2bool, default=False,
                    help='run supervised learning baseline')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data

print('Generating train set...')
train_graphs, valid_graphs, test_graphs = [], [], []
for it in range(args.n_train):
    m = np.random.randint(args.n-1, int(args.n*(args.n-1)/2+1))
    train_graphs.append(Graph.gen_random_graph(args.n, m))

print('Generating valid set...')
for it in range(args.n_valid):
    m = np.random.randint(args.n-1, int(args.n*(args.n-1)/2+1))
    valid_graphs.append(Graph.gen_random_graph(args.n, m))

print('Generating test set...')
for it in range(args.n_valid):
    m = np.random.randint(args.n-1, int(args.n*(args.n-1)/2+1))
    test_graphs.append(Graph.gen_random_graph(args.n, m))

# Model and optimizer
model = GCN(nclass=1, N=args.n, H=args.hidden, dropout=args.dropout)
if args.cuda:
    model.to('cuda:0')
optimizer = optim.Adam(model.parameters(),
                      lr=args.lr, weight_decay=args.weight_decay)

def train(epoch):
    tot_err, tot_dl2_loss = 0, 0
    random.shuffle(train_graphs)
    for i, g in enumerate(train_graphs):

        model.train()
        with torch.no_grad():
            idx = torch.LongTensor([g.x, g.y])
            v = torch.FloatTensor(np.ones(len(g.x)))
            adj = torch.sparse.FloatTensor(idx, v, torch.Size([g.n, g.n])).to_dense()
            if args.cuda:
                adj = adj.cuda()

        optimizer.zero_grad()
        out = model.forward(adj)
        dist = torch.FloatTensor([g.p[0, i] for i in range(g.n)])
        if args.cuda:
            dist = dist.cuda()

        err = torch.mean((dist - out) * (dist - out))
        tot_err += err.detach()
        if not args.baseline:
            conjunction = []
            for a in range(1, g.n):
                disjunction = []
                for b in range(g.n):
                    if adj[a, b]:
                        disjunction.append(dl2.EQ(out[a], out[b] + 1))
                        conjunction.append(dl2.LEQ(out[a], out[b] + 1))
                    conjunction.append(dl2.Or(disjunction))
            conjunction.append(dl2.EQ(out[0], 0))
            for a in range(0, g.n):
                conjunction.append(dl2.GEQ(out[0], 0))
            constraint = dl2.And(conjunction)
            dl2_loss = constraint.loss(args)
            dl2_loss.backward()
            tot_dl2_loss += dl2_loss.detach()
        else:
            err.backward()

        optimizer.step()


def test(val=True, e=None):
    model.eval()
    tot_err = 0
    baseline_err = 0
    all_ones = torch.ones(args.n)
    if args.cuda:
        all_ones = all_ones.cuda()

    for i, g in enumerate(valid_graphs if val else test_graphs):
        model.eval()
        with torch.no_grad():
            idx = torch.LongTensor([g.x, g.y])
            v = torch.FloatTensor(np.ones(len(g.x)))
            adj = torch.sparse.FloatTensor(idx, v, torch.Size([g.n, g.n])).to_dense()
            if args.cuda:
                adj = adj.cuda()
            
        out = model.forward(adj)
        dist = torch.FloatTensor([g.p[0, i] for i in range(g.n)])
        if args.cuda:
            dist = dist.cuda()

        err = torch.mean((dist - out) * (dist - out))
        baseline_err += torch.mean((dist - all_ones) * (dist - all_ones))
        tot_err += err

    if e is not None:
        print(str(e) + ' ', end='')
    print('[Valid] Average error: ', tot_err/float(len(valid_graphs)))
    if val is False:
        print('[Valid] Baseline err: ', baseline_err/float(len(valid_graphs)))

# Train model
t_total = time.time()
for epoch in range(1, args.epochs):
    train(epoch)
    print('.', end='', flush=True)
    if epoch % 50 == 0:
        print()
        test(e=epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test(val=False)
