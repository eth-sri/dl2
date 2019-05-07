import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

datasets = ['mnist', 'fashion', 'cifar10']
constraints = ['RobustnessT', 'RobustnessG', 'LipschitzT', 'LipschitzG', 'ClassT', 'ClassG', 'SegmentG']
table = [[None for j in range(len(datasets)*2)] for i in range(len(constraints)*2)]

parser = argparse.ArgumentParser(description='Supervised DL2 Training Result printer')
parser.add_argument('--folder', type=str, default='./results', help='Folder where results are stored.')
parser.add_argument('--strategy', '-s', type=str, choices=['last', 'max', 'maxOfLast5', 'sumMaxOfLast5', 'prodMaxOfLast5', 'sumMax', 'prodMax'], default='last', help='How to pick the values')
args = parser.parse_args()

def pick(p, c):
    if args.strategy == 'last':
        return p[-1], c[-1]
    if args.strategy == 'maxOfLast5':
        return max(p[-5:]), max(c[-5:])
    if args.strategy == 'sumMaxOfLast5':
        p, c = np.array(p[-5:]), np.array(c[-5:])
        s = p + c
        i = np.argmax(s)
        return p[i], c[i]
    if args.strategy == 'prodMaxOfLast5':
        p, c = np.array(p[-5:]), np.array(c[-5:])
        s = p * c
        i = np.argmax(s)
        return p[i], c[i]
    elif args.strategy == 'max':
        return np.array(p).max(), np.array(c).max()
    elif args.strategy == 'sumMax':
        p, c = np.array(p), np.array(c)
        s = p + c
        i = np.argmax(s)
        return p[i], c[i] 
    elif args.strategy == 'prodMax':
        p, c = np.array(p), np.array(c)
        s = p * c
        i = np.argmax(s)
        return p[i], c[i] 
    

# pass the parameters to the constraint name
def params_to_constraint(data, ds, postfix):
    params = data['constraint_params']
    # This is a bit hacky but required as some old results don't have the name field
    if 'L' in params:
        name = 'Lipschitz'
        flag = (params['L'] == 1.0 and postfix == 'T') or (params['L'] == 0.1 and postfix == 'G') or (params['L'] == 1.0 and ds == 'cifar10')
    elif 'delta' in params:
        name = 'Class'
        flag = True
    elif 'eps' in params and 'p_limit' in params:
        name = 'Segment'
        flag = True
    elif ('eps1' in params and 'eps2' in params):
        name = 'Robustness'
        flag = (params['eps1'] == 7.8 and params['eps2'] == 2.9 and (data['dl2_weight'] == 0.2 or data['dl2_weight'] == 0)) or ds == 'cifar10'
    elif ('eps' in params):
        name = 'Robustness'
        flag = True
    else:
        print(params)
        assert False
    return name + postfix, flag


print('Dataset Constraints')
print('=' * 20)
print()

T = os.path.join(args.folder, 'T')
for ds in datasets:
    print(ds)
    path = os.path.join(T, ds)
    if not os.path.exists(path):
        continue
    for report in os.listdir(path):
        report_path = os.path.join(path, report)
        with open(report_path, 'r') as f:
            data = json.load(f)
        name, flag = params_to_constraint(data, ds, 'T')
        if not flag:
            continue
        i = 2 * constraints.index(name)
        j = 2 * datasets.index(ds)
        if 'dl2' in report:
            j = j + 1
        p, c = pick(data['p_acc'], data['c_acc'])
        table[i][j] = p
        table[i + 1][j] = c
        
        
        print('\t', report, name, data['p_acc'][-1], data['c_acc'][-1])
        #print(data.keys())
    print()


print('Global Constraints')
print('=' * 20)
print()

G = os.path.join(args.folder, 'G')
for ds in datasets:
    print(ds)
    path = os.path.join(G, ds)
    if not os.path.exists(path):
        continue
    for report in [f for f in os.listdir(path) if f.endswith('.json')]:
        report_path = os.path.join(path, report)
        with open(report_path, 'r') as f:
            data = json.load(f)
        name, flag = params_to_constraint(data, ds, 'G')
        if not flag:
            continue
        i = 2 * constraints.index(name)
        j = 2 * datasets.index(ds)
        if 'dl2' in report:
            j = j + 1
        p, c = pick(data['p_acc'], data['c_acc'])
        table[i][j] = p
        table[i + 1][j] = c
        
        print('\t', report, name, data['p_acc'][-1], data['c_acc'][-1])
        #print(data.keys())
    print()

    
print()
print(np.array(table, dtype=np.float).round(4) * 100)

# for i, c in enumerate(constraints):
#     for j, ds in enumerate(datasets):
#         if table[2 * i][2 * j] is not None:
#             p_base = np.array(table[2 * i][2 * j])
#             c_base = np.array(table[2 * i + 1][2 * j])
#             p_dl2 = np.array(table[2 * i][2 * j + 1])
#             c_dl2 = np.array(table[2 * i + 1][2 * j + 1])
#             x = np.arange(len(table[2 * i][2 * j]))
#             f = plt.figure()
#             plt.plot(x, p_base, '-')
#             plt.plot(x, c_base, '-.')
#             plt.plot(x, p_dl2, '-')
#             plt.plot(x, c_dl2, '-.')
#             plt.title(c)
#             plt.savefig(f"{c}_{ds}.png")
#             #plt.show()
#             #sys.exit()
