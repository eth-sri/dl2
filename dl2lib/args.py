from configargparse import ArgumentParser
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_default_parser_args(parser, query=False):
    parser.add("--use-eps", type=str2bool, default=False, required=False, help="use the +epsilon translation for strict inequalities")
    parser.add("--eps", type=float, default=1e-5, required=False, help="the epsilon used for strict inequalities")
    parser.add("--or", type=str, default='mul', choices=['mul', 'min'], required=False, help="encoding for or; either multiplication or minimum")
    parser.add("--cuda", type=str2bool, default=True, required=False, help="use cuda if available")
    if query:
        parser.add("--lr", type=float, default=0.1, required=False, help="learning rate for some optimizers")
        parser.add("--opt", '--optimizer', type=str, default='lbfgsb', required=False, help="inner optimizer to use")
        parser.add("--opt-iterations", type=int, default=1, required=False, help="max iterations for inner otpimizier")
        parser.add("--use-basinhopping", type=str2bool, default=True, required=False, help="use baisin-hopping-MCMC as outer otpimizier")
        parser.add("--basinhopping-T", type=float, default=10, required=False, help="temperature T for basinhopping")
        parser.add("--basinhopping-stepsize", type=float, default=0.1, required=False, help="basinhopping step size")
        parser.add("--timeout", "-t", type=int, default=120, required=False, help="timeout in seconds")
    return parser
