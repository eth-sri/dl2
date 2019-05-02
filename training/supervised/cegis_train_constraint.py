import argparse
import os
import torch
import numpy as np
import torch.optim as optim
import json
import time
from torchvision import datasets, transforms
from oracles import DL2_Oracle
from constraints import *
from resnet import ResNet18
from models import MLP, MnistNet
from sklearn.decomposition import PCA
import sys
sys.path.append('../../')
import dl2lib as dl2
import time

use_cuda = torch.cuda.is_available()


def embed_batch(x_batch, pca):
    n_batch = int(x_batch.size()[0])
    n_feat = x_batch[0].numel()

    big_batch = np.zeros((n_batch, n_feat))
    for i in range(n_batch):
        big_batch[i] = x_batch[i].cpu().view(-1).numpy()
    embed_batch = pca.transform(big_batch)

    x_batch = torch.from_numpy(embed_batch).float()
    return x_batch


def embed_pca(data, embed_dim):
    n_data = len(data)
    n_feat = torch.numel(data[0][0])

    tot = n_data
    ids = np.arange(n_data)
    np.random.shuffle(ids)
    ids = ids[:tot]
    x_big = np.zeros((n_data, n_feat))
    for i in range(tot):
        x_big[i] = data[ids[i]][0].view(-1).cpu().numpy()
    pca = PCA(n_components=embed_dim)
    pca.fit(x_big)
    return pca


def train(args, oracle, net, device, train_loader, optimizer, epoch, pca=None):
    t1 = time.time()
    num_steps = 0
    avg_train_acc, avg_constr_acc = 0, 0
    avg_ce_loss, avg_nql_loss = 0, 0
    ce_loss = torch.nn.CrossEntropyLoss()

    print('Epoch ', epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        x_batch, y_batch = data.to(device), target.to(device)
        n_batch = int(x_batch.size()[0])
        num_steps += 1

        if pca is not None:
            x_batch = embed_batch(x_batch, pca)
            if use_cuda:
                x_batch = x_batch.cuda()

        x_outputs = net(x_batch)
        x_correct = torch.mean(torch.argmax(x_outputs, dim=1).eq(y_batch).float())
        ce_batch_loss = ce_loss(x_outputs, y_batch)

        if epoch <= args.delay or args.nql_weight < 1e-7:
            net.train()
            optimizer.zero_grad()
            ce_batch_loss.backward()
            optimizer.step()

            if batch_idx % args.print_freq == 0:
                print('[%d] Train acc: %.4f' % (batch_idx, x_correct.item()))
            continue

        x_batches, y_batches = [], []
        k = n_batch // oracle.constraint.n_tvars
        assert n_batch % oracle.constraint.n_tvars == 0, 'Batch size must be divisible by number of train variables!'
        for i in range(oracle.constraint.n_tvars):
            x_batches.append(x_batch[i:(i + k)])
            y_batches.append(y_batch[i:(i + k)])

        net.eval()

        if oracle.constraint.n_gvars > 0:
            domains = oracle.constraint.get_domains(x_batches, y_batches)
            z_batches = oracle.general_attack(x_batches, y_batches, domains, num_restarts=1, num_iters=args.num_iters, args=args)
            _, nql_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, z_batches, args)
        else:
            _, nql_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, None, args)

        avg_train_acc += x_correct.item()
        avg_constr_acc += constr_acc.item()
        avg_ce_loss += ce_batch_loss.item()
        avg_nql_loss += nql_batch_loss.item()

        if batch_idx % args.print_freq == 0:
            print('[%d] Train acc: %.4f, Constr acc: %.4f, CE loss: %.3lf, DL2 loss: %.3lf' % (
                batch_idx, x_correct.item(), constr_acc.item(), ce_batch_loss.item(), nql_batch_loss.item()))
        
        net.train()
        optimizer.zero_grad()
        tot_batch_loss = args.nql_weight * nql_batch_loss + ce_batch_loss
        tot_batch_loss.backward()
        optimizer.step()
    t2 = time.time()
        
    avg_train_acc /= float(num_steps)
    avg_constr_acc /= float(num_steps)
    avg_nql_loss /= float(num_steps)
    avg_ce_loss /= float(num_steps)
    time = t2 - t1
    
    return avg_train_acc, avg_constr_acc, avg_nql_loss, avg_ce_loss, time


def test(args, oracle, model, device, test_loader, pca=None):
    loss = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct, constr, num_steps, pgd_ok = 0, 0, 0, 0
    
    for data, target in test_loader:
        num_steps += 1
        x_batch, y_batch = data.to(device), target.to(device)
        n_batch = int(x_batch.size()[0])

        if pca is not None:
            x_batch = embed_batch(x_batch, pca)
            if use_cuda:
                x_batch = x_batch.cuda()

        x_batches, y_batches = [], []
        k = n_batch // oracle.constraint.n_tvars
        assert n_batch % oracle.constraint.n_tvars == 0, 'Batch size must be divisible by number of train variables!'

        for i in range(oracle.constraint.n_tvars):
            x_batches.append(x_batch[i:(i + k)])
            y_batches.append(y_batch[i:(i + k)])

        if oracle.constraint.n_gvars > 0:
            domains = oracle.constraint.get_domains(x_batches, y_batches)
            z_batches = oracle.general_attack(x_batches, y_batches, domains, num_restarts=1, num_iters=args.num_iters, args=args)
            _, nql_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, z_batches, args)
        else:
            _, nql_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, None, args)

        output = model(x_batch)
        test_loss += loss(output, y_batch).item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            
        correct += pred.eq(y_batch.view_as(pred)).sum().item()
        constr += constr_acc.item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Pred. Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Constr. acc: %.4f' % (constr / float(num_steps)))

    return correct / len(test_loader.dataset), constr / float(num_steps)


parser = argparse.ArgumentParser(description='Train NN with constraints.')
parser = dl2.add_default_parser_args(parser)
parser.add_argument('--batch-size', type=int, default=128, help='Number of samples in a batch.')
parser.add_argument('--num-iters', type=int, default=50, help='Number of oracle iterations.')
parser.add_argument('--num-epochs', type=int, default=300, help='Number of epochs to train for.')
parser.add_argument('--exp-name', type=str, default='', help='Name of the experiment.')
parser.add_argument('--nql-weight', type=float, default=0.0, help='Weight of DL2 loss.')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset on which to train.')
parser.add_argument('--pretrained', action='store_true', help='Whether to use pretrained model.')
parser.add_argument('--embed', action='store_true', help='Whether to embed the points.')
parser.add_argument('--delay', type=int, default=0, help='How many epochs to wait before training with constraints.')
parser.add_argument('--print-freq', type=int, default=10, help='Print frequency.')
parser.add_argument('--report-dir', type=str, required=True, help='Directory where results should be stored')
parser.add_argument('--constraint', type=str, required=True, help='the constraint to train with: LipschitzT(L), LipschitzG(eps, L), RobustnessT(eps1, eps2), RobustnessG(eps, delta), CSimiliarityT(), CSimilarityG(),LineSegmentG()')
parser.add_argument('--embed-dim', type=int, default=40, help='embed dim')
args = parser.parse_args()

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
pca = None

if args.dataset == 'mnist':
    mnist_data = datasets.MNIST('../../data/mnist/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(mnist_data, shuffle=True, batch_size=args.batch_size, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../../data/mnist/', train=False, transform=transforms.Compose([
        transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    if args.embed:
        pca = embed_pca(mnist_data, args.embed_dim)
        model = MLP(args.embed_dim, 10, 1000, 3).to(device)
    else:
        model = MnistNet().to(device)

elif args.dataset == 'fashion':
    fashion_data = datasets.FashionMNIST('../../data/fashionmnist/', train=True, download=True,
                                         transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(fashion_data, shuffle=True, batch_size=args.batch_size, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../../data/fashionmnist/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    if args.embed:
        pca = embed_pca(fashion_data, args.embed_dim)
        model = MLP(args.embed_dim, 10, 1000, 3).to(device)
    else:
        model = MnistNet().to(device)
elif args.dataset == 'cifar10':
    transform = transforms.Compose([transforms.ToTensor()])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])  # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    cifar_data = datasets.CIFAR10('../../data/cifar10', train=True, download=True, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(cifar_data, shuffle=True, batch_size=args.batch_size, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../../data/cifar10', train=False, download=True, transform=transform_test),
        batch_size=256, shuffle=True, **kwargs)

    if args.embed:
        pca = embed_pca(cifar_data, args.embed_dim)
        model = MLP(args.embed_dim, 10, 1000, 3).to(device)
    else:
        model = ResNet18().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

def RobustnessT(eps1, eps2):
    return lambda model, use_cuda: RobustnessDatasetConstraint(model, eps1, eps2, use_cuda=use_cuda)

def RobustnessG(eps, delta):
    return lambda model, use_cuda: RobustnessConstraint(model, eps, delta, use_cuda)

def LipschitzT(L):
    return lambda model, use_cuda: LipschitzDatasetConstraint(model, L, use_cuda)

def LipschitzG(eps, L):
    return lambda model, use_cuda: LipschitzConstraint(model, eps=eps, l=L, use_cuda=use_cuda)

def CSimilarityT(delta):
    return lambda model, use_cuda: CifarDatasetConstraint(model, delta, use_cuda)

def CSimilarityG(eps, delta):
    pass # TODO

def SegmentG(eps, delta):
    return lambda model, use_cuda: PairLineRobustnessConstraint(model, eps, delta, use_cuda)

constraint = eval(args.constraint)(model, use_cuda)
oracle = DL2_Oracle(learning_rate=0.01, net=model, constraint=constraint, use_cuda=use_cuda)

opt_type = 'dataset' if constraint.n_gvars == 0 else 'local'
report_dir = os.path.dirname(
    os.path.join(args.report_dir, '%s/%s/%s' % (opt_type, args.dataset, constraint.name)))

if not os.path.exists(report_dir):
    os.makedirs(report_dir)

tstamp = int(time.time())

report_file = os.path.join(report_dir, 'report_%s_%d.json' % (args.exp_name, tstamp))
data_dict = {
    'nql_weight': args.nql_weight,
    'pretrained': args.pretrained,
    'delay': args.delay,
    'constraint_params': constraint.params(),
    'num_iters': args.num_iters,
    'train_acc': [],
    'constr_acc': [],
    'nql_loss': [],
    'ce_loss': [],
    'p_acc': [],
    'c_acc': [],
    'epoch_time': []
}

for epoch in range(1, args.num_epochs + 1):
    avg_train_acc, avg_constr_acc, avg_nql_loss, avg_ce_loss, epoch_time = \
        train(args, oracle, model, device, train_loader, optimizer, epoch, pca)
    data_dict['train_acc'].append(avg_train_acc)
    data_dict['constr_acc'].append(avg_constr_acc)
    data_dict['ce_loss'].append(avg_ce_loss)
    data_dict['nql_loss'].append(avg_nql_loss)
    data_dict['epoch_time'].append(epoch_time)

    p, c = test(args, oracle, model, device, test_loader, pca)
    data_dict['p_acc'].append(p)
    data_dict['c_acc'].append(c)
    print('Epoch Time:', epoch_time)

with open(report_file, 'w') as fou:
    json.dump(data_dict, fou, indent=4)
