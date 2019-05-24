import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rcParams
import sys

datasets = ['mnist', 'fashion', 'cifar10']
constraints = ['RobustnessT', 'RobustnessG', 'LipschitzT', 'LipschitzG', 'ClassT', 'ClassG', 'SegmentG']
table = [[None for j in range(len(datasets)*2)] for i in range(len(constraints)*2)]
time = [[None for j in range(len(datasets)*2)] for i in range(len(constraints))]
history = [[None for j in range(len(datasets)*2)] for i in range(len(constraints)*2)]

parser = argparse.ArgumentParser(description='Supervised DL2 Training Result printer')
parser.add_argument('--folder', type=str, default='./results', help='Folder where results are stored.')
parser.add_argument('--plot-dir', type=str, default='./plots', help='Folder where plots will be stored.')
parser.add_argument('--strategy', '-s', type=str, choices=['last', 'max', 'maxOfLast25', 'sumMaxOfLast25', 'prodMaxOfLast25', 'sumMax', 'prodMax'], default='prodMaxOfLast25', help='How to pick the values')
args = parser.parse_args()

def pick(p, c):
    if args.strategy == 'last':
        return p[-1], c[-1]
    if args.strategy == 'maxOfLast25':
        return max(p[-25:]), max(c[-25:])
    if args.strategy == 'sumMaxOfLast25':
        p, c = np.array(p[-25:]), np.array(c[-25:])
        s = p + c
        i = np.argmax(s)
        return p[i], c[i]
    if args.strategy == 'prodMaxOfLast25':
        p, c = np.array(p[-25:]), np.array(c[-25:])
        s = p * (c + 0.01) # to make sure we get the max even if c is 0
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
        flag = ((params['L'] == 1.0 and postfix == 'T') or
                (params['L'] == 0.1 and postfix == 'G' and ds != 'fashion') or
                (params['L'] == 0.3 and postfix == 'G' and ds == 'fashion' and (data['dl2_weight'] == 0.2 or data['dl2_weight'] == 0)) or
                (params['L'] == 1.0 and ds == 'cifar10'))
    elif 'delta' in params:
        name = 'Class'
        flag = (data['dl2_weight'] == 0.2 or data['dl2_weight'] == 0)
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
        assert False
    return name + postfix, flag


def traverse(folder, postfix):
    for ds in datasets:
        print(ds)
        path = os.path.join(folder, ds)
        if not os.path.exists(path):
            continue
        for report in [f for f in os.listdir(path) if f.endswith('.json')]:
            report_path = os.path.join(path, report)
            with open(report_path, 'r') as f:
                data = json.load(f)
            name, flag = params_to_constraint(data, ds, postfix)
            if not flag:
                continue
            i = 2 * constraints.index(name)
            j = 2 * datasets.index(ds)
            if 'dl2' in report:
                j = j + 1
            p, c = pick(data['p_acc'], data['c_acc'])
            if table[i][j] is not None:
                print('!')
            table[i][j] = p
            table[i + 1][j] = c
            history[i][j] = data['p_acc']
            history[i + 1][j] = data['c_acc']    
            time[constraints.index(name)][j] = np.array(data['epoch_time']).mean()
            print('\t', report, name, p, c)
        print()
    

print('Dataset Constraints')
print('=' * 20)
print()

T = os.path.join(args.folder, 'T')
traverse(T, 'T')

print('Global Constraints')
print('=' * 20)
print()

G = os.path.join(args.folder, 'G')
traverse(G, 'G')

    
print()
print('P/C Accuracy')
print(np.array(table, dtype=np.float).round(4) * 100)

print()
print('Avg Time per Epoch [s]')
print(np.array(time, dtype=np.float).round(2))


os.makedirs(args.plot_dir, exist_ok=True)

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Gill Sans MT']
#plt.rc('text', usetex=True)

fontsize=25

for i, c in enumerate(constraints):
    for j, ds in enumerate(datasets):
        if table[2 * i][2 * j] is not None:
            p_base = np.array(history[2 * i][2 * j])
            c_base = np.array(history[2 * i + 1][2 * j])
            p_dl2 = np.array(history[2 * i][2 * j + 1])
            c_dl2 = np.array(history[2 * i + 1][2 * j + 1])
            x = np.arange(len(history[2 * i][2 * j]))



            f = plt.figure(figsize=[12, 7.2]) # default size is 6.4, 4.8
            ax = plt.gca()
           
            if not (ds == 'fashion' and c == 'SegmentG'):
                l1, = plt.plot(x, p_base, '-C0', lw=2.0)
                l2, = plt.plot(x, p_dl2, '-C1', lw=2.0)
            else:
                l1, = plt.plot(x, p_base, '-C3', lw=2.0)
                l2, = plt.plot(x, p_dl2, '-C4', lw=2.0)

            plt.plot(x, c_base, ':', color=l1.get_color(), lw=4.0)
            plt.plot(x, c_dl2, ':', color=l2.get_color(), lw=4.0)

            if (ds == 'fashion' and c == 'SegmentG'):
                x = 208
                ax.text(x, 0.20, "C-Acc, Baseline", fontsize=fontsize, color=l1.get_color())
                ax.text(x, 0.50, "C-Acc, DL2", fontsize=fontsize, color=l2.get_color())
                ax.text(x, 0.88, "P-Acc, Baseline", fontsize=fontsize, color=l1.get_color())
                ax.text(x, 0.815, "P-Acc, DL2", fontsize=fontsize, color=l2.get_color())
                ax.text(75, 0.62, '${Segment}^{G}, FASHION$', fontsize=30)
                
            if (ds == 'fashion' and c == 'LipschitzT'):
                x = 208
                ax.text(x, 0.05, "C-Acc, Baseline", fontsize=fontsize, color=l1.get_color())
                ax.text(x, 0.985, "C-Acc, DL2", fontsize=fontsize, color=l2.get_color())
                ax.text(x, 0.920, "P-Acc, Baseline", fontsize=fontsize, color=l1.get_color())
                ax.text(x, 0.855, "P-Acc, DL2", fontsize=fontsize, color=l2.get_color())
                ax.text(75, 0.52, '${Lipschitz}^{T}, FASHION$', fontsize=30)
                
            if not (ds == 'fashion' and c == 'LipschitzT'):
                plt.xlabel('Epoch', fontsize=fontsize)

            if not (ds == 'fashion' and c == 'SegmentG'):
                plt.ylabel('Accuracy', rotation=0, fontsize=fontsize)
                ax.yaxis.set_label_coords(0.01, 1.02)

            #plt.xticks([10*i for i in range(])
            ax.set_yticks([0.1*(2*i) for i in range(6)])
            ax.set_yticks([0.1*(2*i + 1) for i in range(6)], minor=True)
            ax.grid(which='minor', color=(1,1,1))
            ax.yaxis.grid(True, color=(1,1,1))

            ax.set_xticks([50*i for i in range(5)])

            plt.setp(ax.get_xticklabels(), fontsize=fontsize)
            plt.setp(ax.get_yticklabels(), fontsize=fontsize)
            
            ax.set_facecolor( (0.97, 0.97, 0.97) )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_ylim([0.0, 1.025])
            ax.set_xlim([-5, 205])
            
            # if ds == 'fashion' and c == 'LipschitzT':
            #     m1, = ax.plot([], [], c='C0' , marker='s', markersize=20,
            #                   fillstyle='left', linestyle='none')
            #     m2, = ax.plot([], [], c='C3' , marker='s', markersize=20,
            #                   fillstyle='right', linestyle='none')
            #     m3, = ax.plot([], [], c='C1' , marker='s', markersize=20,
            #                   fillstyle='left', linestyle='none')
            #     m4, = ax.plot([], [], c='C4' , marker='s', markersize=20,
            #                   fillstyle='right', linestyle='none')
            #     p_line = Line2D([0], [0], color='k', linestyle='-')
            #     c_line = Line2D([0], [0], color='k', linestyle=':', linewidth=4.0)            
            #     ax.legend(((m2, m1), (m3, m4), p_line, c_line), ('Baseline', 'DL2', 'P', 'C'), loc=7, fontsize=fontsize, numpoints=1)

            #ax.legend((l1, l2), ('Baseline', 'DL2'))
            f.tight_layout()
            f = os.path.join(args.plot_dir, f"{c}_{ds}.eps")
            plt.savefig(f)
