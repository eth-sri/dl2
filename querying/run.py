import sys
sys.path.append("..")
import dl2lib.query as q
import dl2lib as dl2
from context import get_context
from evaluation_queries import get_queries
from configargparse import ArgumentParser
import argparse
import torch
import signal
import time
import random
import numpy as np

parser = ArgumentParser(description='DL2 Querying', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = dl2.add_default_parser_args(parser, query=True)
parser.add("--instances", type=int, default=10, required=False, help="max number of instances to run per query")
parser.add("--glob-pattern", type=str, default="*.tdql", required=False, help="pattern to glob for tdql files in ./evaluation_queries")
parser.add('--dataset', choices=['MNIST', 'FASHION_MNIST', 'CIFAR', 'GTSRB', 'IMAGENT'], default=['MNIST', 'FASHION_MNIST', 'CIFAR', 'GTSRB', 'IMAGENT'], nargs='+', help='datasets to use')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
random.seed(42)
np.random.seed(42)
    
context = get_context(args)
queries = get_queries(args)


for dataset in args.dataset:
    print(dataset)
    print('=' * 10)
    print()

    for query_name in queries[dataset]:
        qs = queries[dataset][query_name]
        solved_queries = 0
        time_all = float(0.0)
        time_solved = float(0.0)

        for query_string in qs:
            solved, results, t = q.Query(query_string, context, args).run()
            time_all += float(t)
            if solved:
                time_solved += float(t)
                solved_queries += 1

        time_all = time_all / len(qs)
        time_solved = time_solved / solved_queries if solved_queries > 0 else 0.0
        print(f"{query_name}\t{solved_queries}/{len(qs)}\t{time_all:.2f}\t{time_solved:.2f}")
