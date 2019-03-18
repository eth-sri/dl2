import sys
sys.path.append("..")
import dl2lib.query as q
import dl2lib as dl2
from context import get_context
from evaluation_queries import get_queries
from configargparse import ArgumentParser
import argparse
import torch
import random
import numpy as np


def q1(s):
    i = q.Variable('i', (s,))
    i_s = i.sum()
    success, r, t = q.solve(q.And(1000 < i_s, i_s < 1001), args, return_values=[i_s])
    return success, r, t


def q2(c):
    i = q.Variable('i', (1,))
    success, r, t = q.solve(q.Or(i < -c, c < i), args)
    return success, r, t

def q3(c, row_up_to, col_up_to, context):

    def lt_row_constraint(i, j):
        return f'p[0, 0, {i}, {j}] < p[0, 0, {i}, {j+1}]'

    def lt_col_constraint(i, j):
        return f'p[0, 0, {i}, {j}] < p[0, 0, {i+1}, {j}]'

    def row_constraint(i):
        return ', '.join([lt_row_constraint(i, j) for j in range(27)])

    def col_constraint(j):
        return ', '.join([lt_col_constraint(i, j) for i in range(27)])

    query_constraints = [f'class(M_NN1(clamp(p + M_nine, 0, 1)), {c})']#, 'p in [-0.3, 0.3]']

    for i in range(row_up_to):
        query_constraints.append(row_constraint(i))
    for i in range(col_up_to):
        query_constraints.append(col_constraint(i))

    query = "FIND p[1, 1, 28, 28]\nS.T.\n"
    query += ",\n".join(query_constraints)
    query += "\nRETURN clamp(p + M_nine, 0, 1)"

    success, r, t = q.Query(query, context=context, args=args).run()
    return success, r, t



parser = ArgumentParser(description='DL2 Querying', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = dl2.add_default_parser_args(parser, query=True)
parser.add("--instances", type=int, default=10, required=False, help="max number of instances to run per query")
parser.add("--query", type=int, choices=[1, 2, 3], default=1, required=False, help="max number of instances to run per query")
parser.add("-a", type=int, default=None, required=False, help="argument")
parser.add("--plot", type=dl2.str2bool, default=False, required=False, help="argument")
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
random.seed(42)
np.random.seed(42)

if args.query == 1:
    s_max = args.a or 23
    for s in range(0, s_max):
        success_count = 0.0
        t_all = 0.0
        t_success = 0.0
        s = 2**s
        for k in range(args.instances):
            success, r, t = q1(s)
            t_all += t
            if success:
                t_success += t
                success_count += 1
        t_all /= args.instances
        t_success = t_success / success_count if success_count != 0 else 0.0
        print(s, f"{success_count}/{args.instances}", f"{t_all:.2f}", f"{t_success:.2f}")
elif args.query == 2:
    c_max = args.a or 18
    for c in range(0, c_max):
        success_count = 0.0
        t_all = 0.0
        t_success = 0.0
        c = 2**c
        for k in range(args.instances):
            success, r, t = q2(c)
            t_all += t
            if success:
                t_success += t
                success_count += 1
        t_all /= args.instances
        t_success = t_success / success_count if success_count != 0 else 0.0
        print(c, f"{success_count}/{args.instances}", f"{t_all:.2f}", f"{t_success:.2f}")
    pass
elif args.query == 3:
    context = get_context(args)

    def run_(row, col):
        success_count = 0.0
        t_all = 0.0
        t_success = 0.0
        for c in range(9):
            success, r, t = q3(c, row, col, context)
            t_all += t
            if success:
                t_success += t
                success_count += 1
        t_all /= 9.0
        t_success = t_success / success_count if success_count != 0 else 0.0
        print(row, col, f"{success_count}/{args.instances}", f"{t_all:.2f}", f"{t_success:.2f}")
        if success and row == 28 and col == 28 and args.plot:
            import matplotlib.pyplot as plt
            plt.imshow(r.reshape(28, 28), vmin=0, vmax=1, cmap='gray')
            plt.show()

    print()
    for row in [0, 1, 3, 5, 10, 20, 28]:
        col = 0
        run_(row, col)

    print()
    for col in [0, 1, 3, 5, 10, 20, 28]:
        row = 0
        run_(row, col)

    print()
    for both in [0, 1, 3, 5, 10, 20, 28]:
        run_(both, both)
