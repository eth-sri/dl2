import torch
from .. import diffsat
from ..util import lmap
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.optimize as spo
import time
import sys
import signal
import re

def make_comp(op, a, b):
    a_is_inf = (isinstance(a, Fn) and a.t == 'normInf') and a.args[0].is_shape_preserving_arithmetic_in_var_const()
    b_is_inf = (isinstance(b, Fn) and b.t == 'normInf') and b.args[0].is_shape_preserving_arithmetic_in_var_const()
    if (((a.is_var() or a.is_const()) and (b.is_var() or b.is_const())) or
            (a_is_inf and op in ['le', 'lt'] and (b.is_var() or b.is_const())) or
            (b_is_inf and op in ['ge', 'gt'] and (a.is_var() or a.is_const()))):
        if a_is_inf:
            a = Fn('abs', lambda a: a.abs(), a.args[0])
        if b_is_inf:
            b = Fn('abs', lambda b: b.abs(), b.args[0])
        a = Fn('view(-1)', lambda a: a.view(-1), a)
        ashape = a.shape()
        b = Fn('view(-1)', lambda b: b.view(-1), b)
        bshape = b.shape()
        if ashape == bshape:
            return And(*[Comp(op, a[i], b[i]) for i in range(ashape[0])])
        elif ashape[0] == 1:
            return And(*[Comp(op, a, b[i]) for i in range(bshape[0])])
        elif bshape[0] == 1:
            return And(*[Comp(op, a[i], b) for i in range(ashape[0])])
        else:
            assert False, f"Shape mismatch: {ashape} {bshape}"
    else:
        return Comp(op, a, b)

class DL2Tensor:

    def __init__(self):
        pass

    def upgrade_other(self, other):
        if type(other) in [float, int, torch.Tensor, np.array, np.ndarray]:
            return Constant(other, self.cuda)
        return other
    
    def __add__(self, other):
        other = self.upgrade_other(other)
        return Fn('+', lambda a, b: a + b, self, other)

    def __mul__(self, other):
        other = self.upgrade_other(other)
        return Fn('*', lambda a, b: a * b, self, other)

    def __sub__(self, other):
        other = self.upgrade_other(other)
        return Fn('-', lambda a, b: a - b, self, other)

    def sum(self):
        return Fn('sum', lambda a: a.sum(), self)
    
    def __lt__(self, other):
        other = self.upgrade_other(other)
        return make_comp('lt', self, other)

    def __le__(self, other):
        other = self.upgrade_other(other)
        return make_comp('le', self, other)

    def __gt__(self, other):
        other = self.upgrade_other(other)
        return make_comp('gt', self, other)

    def __ge__(self, other):
        other = self.upgrade_other(other)
        return make_comp('ge', self, other)

    def eq_(self, other):
        other = self.upgrade_other(other)
        return make_comp('eq', self, other)
    
    def __neg__(self):
        return Fn('neg', lambda a: -a, self)

    def shape(self):
        if self.is_var() or self.is_const() or self.is_shape_preserving_arithmetic_in_var_const():
            with torch.no_grad():
                return self.to_diffsat(cache=False).shape
        else:
            return None

    def reset_cache(self):
        pass
        
    # [] operator
    def __getitem__(self, key):
        return Fn('[]', lambda a, b: a.__getitem__(b), self, key)

    def is_var(self):
        return isinstance(self, Variable) or (isinstance(self, Fn) and (self.t == '[]' or 'view' in self.t) and self.args[0].is_var())

    def is_const(self):
        return isinstance(self, Constant) or (isinstance(self, Fn) and (self.t == '[]' or 'view' in self.t) and self.args[0].is_const())

    def is_shape_preserving_arithmetic_in_var_const(self):
        if self.is_var or self.is_const():
            return True
        if isinstance(self, Fn):
            return self.t in ['+', '-', 'abs'] and all([a.is_shape_preserving_arithmetic_in_var_const() for a in self.args])
        return False
        return isinstance(self, Constant) or (isinstance(self, Fn) and (self.t == '[]' or 'view' in self.t) and self.args[0].is_const())
    
    # in as function name, not the operator
    def in_(self, interval):
        assert isinstance(interval, Interval)
        return And(interval.a <= self, self <= interval.b)

    def init(self, other):
        assert self.is_var()
        other = self.upgrade_other(other)
        value = other.tensor.clone().view(-1)
        var = self.to_diffsat().view(-1)
        with torch.no_grad():
            var[:] = value[:]

    def simplify(self, delete_box_constraints=False):
        return self        

class DL2Logic:

    def __init__(self):
        pass

    def get_variables(self):
        variables = []
        for arg in self.args:
            if hasattr(arg, 'get_variables'):
                variables.extend(arg.get_variables())
        return variables

    def simplify(self, delete_box_constraints=False):
        return self


class Class(DL2Logic):

    def __init__(self, net, c):
        assert isinstance(net, Fn) and net.t == '()'
        self.net = net
        self.c = c

    def __str__(self):
        return f"(class, {self.net}, {self.c})"
        
    def get_variables(self):
        return self.net.get_variables()

    def reset_cache(self):
        self.net.reset_cache()
        self.c.reset_cache()
    
    def to_diffsat(self, cache=True, reset_cache=False):
        if reset_cache:
            self.reset_cache()
        logits = self.net.to_diffsat(cache=cache)
        c = self.c.to_diffsat(cache=cache) if hasattr(self.c, 'to_diffsat') else self.c
        c = int(c)
        batch_size, nr_classes = logits.shape
        assert batch_size == 1
        constraints = []
        for k in range(nr_classes):
            if k == c:
                continue
            constraints.append(diffsat.LT(logits[0, k], logits[0, c]))
        return diffsat.And(constraints)
    
class And(DL2Logic):

    def __init__(self, *args):
        assert all(map(lambda x: isinstance(x, DL2Logic), args))
        self.args = args

    def __str__(self):
        return " and ".join(map(str, self.args))

    def reset_cache(self):
        for arg in self.args:
            arg.reset_cache()
    
    def to_diffsat(self, cache=True, reset_cache=False):
        if reset_cache:
            self.reset_cache()
        return diffsat.And(lmap(lambda x: x.to_diffsat(cache=cache), self.args))

    def get_box_constraints(self):
        boxes = []
        for arg in self.args:
            if hasattr(arg, 'get_box_constraints'):
                boxes.extend(arg.get_box_constraints())
        return boxes

    def simplify(self, delete_box_constraints=False):
        new_args = []
        for arg in self.args:
            is_box = hasattr(arg, 'is_box_constraint') and arg.is_box_constraint()
            if (delete_box_constraints and not is_box) or not delete_box_constraints:
                arg = arg.simplify(delete_box_constraints=delete_box_constraints)
                if (isinstance(arg, And) or isinstance(arg, Or)) and len(arg.args) == 0:
                    continue
                new_args.append(arg)
        return And(*new_args)

    
class Or(DL2Logic):

    def __init__(self, *args):
        assert all(map(lambda x: isinstance(x, DL2Logic), args))
        self.args = args

    def __str__(self):
        return " or ".join(map(str, self.args))

    def reset_cache(self):
        for arg in self.args:
            arg.reset_cache()
    
    def to_diffsat(self, cache=True, reset_cache=False):
        if reset_cache:
            self.reset_cache()
        return diffsat.Or(lmap(lambda x: x.to_diffsat(cache=cache), self.args))

    def get_box_constraints(self):
        return [] # we can't go over "or"
    
        
class Comp(DL2Logic):

    def __init__(self, t, a, b):
        assert isinstance(a, DL2Tensor)
        assert isinstance(b, DL2Tensor)
        self.t = t
        self.a = a
        self.b = b

    def __str__(self):
        return f"({self.t} {self.a} {self.b})"

        return " or ".join(map(str, self.args))

    def reset_cache(self):
        self.a.reset_cache()
        self.b.reset_cache()
    
    def to_diffsat(self, cache=True, reset_cache=False):
        if reset_cache:
            self.reset_cache()
        op = {'eq': diffsat.EQ,
              'lt': diffsat.LT,
              'le': diffsat.LEQ,
              'gt': diffsat.GT,
              'ge': diffsat.GEQ}[self.t]
        a = self.a.to_diffsat(cache=cache)
        b = self.b.to_diffsat(cache=cache)
        if a.shape == torch.Size([1]):
            a = a.view([])
        if b.shape == torch.Size([1]):
            b = b.view([])
        assert a.shape == torch.Size([])
        assert b.shape == torch.Size([])
        return op(a, b)

    def get_variables(self):
        return self.a.get_variables() + self.b.get_variables()

    def is_box_constraint(self):
        return (self.a.is_const() and self.b.is_var()) or (self.a.is_var() and self.b.is_const())
    
    def get_box_constraints(self):
        return [self] if self.is_box_constraint() else []

    # def isNormInf
    
    # def simplify(self):
    #     pass
    
    
class Fn(DL2Tensor):

    def __init__(self, t, fn, *args):
        self.t = t
        self.fn = fn
        self.args = args
        self.cuda = any([hasattr(a, 'cuda') and a.cuda for a in self.args])
        self.cache = None

    def __str__(self):
        return f"({self.t}, {','.join(map(str, self.args))})"

    def reset_cache(self):
        self.cache = None
        for a in self.args:
            if hasattr(a, 'reset_cache'):
                a.reset_cache()
    
    def to_diffsat(self, cache=True, reset_cache=False):
        if reset_cache:
            self.reset_cache()
        if cache and self.cache is not None:
            return self.cache
        args = [a.to_diffsat(cache=cache) if hasattr(a, 'to_diffsat') else a for a in self.args]
        result = self.fn(*args)
        if cache and self.cache is None:
            self.cache = result
        return result

    def get_variables(self):
        variables = []
        for arg in self.args:
            if hasattr(arg, 'get_variables'):
                variables.extend(arg.get_variables())
        return variables
        
class Variable(DL2Tensor):

    def __init__(self, name, shape, cuda=False):
        super().__init__()
        self.name = name
        self.shape = shape
        self.tensor = torch.zeros(self.shape)
        self.cuda = cuda
        if cuda:
            self.tensor = self.tensor.to('cuda:0')
        self.tensor.requires_grad_()

    def __str__(self):
        return self.name

    def to_diffsat(self, cache=True, reset_cache=False):
        if reset_cache:
            self.reset_cache()
        return self.tensor

    def get_variables(self):
        return [self]

    
class Constant(DL2Tensor):

    def __init__(self, value, cuda=False):
        super().__init__()
        # pytorch does not support bools
        self.value = value
        if isinstance(value, np.ndarray):
            if value.dtype == np.bool_:
                value = value.astype(np.uint8)
            self.tensor = torch.tensor(value)
        else:
            self.tensor = torch.tensor(float(value))

        self.cuda = cuda
        if cuda:
            self.tensor = self.tensor.to('cuda:0')


    def __str__(self):
        if len(str(self.value)) < 10:
            return f"({self.value})"
        else:
            return f"(Constant{list(self.tensor.shape)})"

    def to_diffsat(self, cache=True, reset_cache=False):
        if reset_cache:
            self.reset_cache()
        return self.tensor

    def get_variables(self):
        return []

class Interval:

    def __init__(self, a, b, cuda=False):
        super().__init__()
        self.a = Constant(a, cuda)
        self.b = Constant(b, cuda)

    def __str__(self):
        return f"([{self.a}, {self.b}])"    
    
class Model(DL2Tensor):

    def __init__(self, model):
        self.model = model
        self.cuda = next(model.parameters()).is_cuda

    def __call__(self, *args):
        return Fn('()', lambda a, b: a(b), self, *args)

    def __str__(self):
        return f"(Model)"

    def to_diffsat(self, cache=True, reset_cache=False):
        if reset_cache:
            self.reset_cache()
        return self.model
    
    def get_variables(self):
        return []

    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        else:
            return ModelLayer(self, attr)
    
class ModelLayer(Model):

    def __init__(self, model, layer):
        assert layer in ['p']
        self.model = model
        self.cuda = model.cuda
        self.layer = layer

    def __str__(self):
        return f"(Model.{self.layer})"

    def to_diffsat(self, cache=True, reset_cache=False):
        if reset_cache:
            self.reset_cache()
        if self.layer == 'p':
            return torch.nn.Sequential(self.model.to_diffsat(cache=cache), torch.nn.Softmax())
    
    def get_variables(self):
        return self.model.get_variables()

    def __getattr__(self, attr):
        assert False
    
            
def simplify(constraint, args):
    if args.opt == 'lbfgsb':
        boxes = constraint.get_box_constraints()
        constraint = constraint.simplify(delete_box_constraints=True)
        variables = list(set(constraint.get_variables()))
        bounds = {}
        for var in variables:
            bounds[var] = (torch.zeros_like(var.tensor).view(-1).cpu().numpy(), torch.zeros_like(var.tensor).view(-1).cpu().numpy())
            bounds[var][0][:] = -np.inf
            bounds[var][1][:] = np.inf
        for box in boxes:
            if box.a.is_const():
                const = box.a
                setop = box.b
                is_upper = box.t in ["eq", "ge", "gt"]
                is_lower = box.t in ["eq", "le", "lt"]
            else:
                const = box.b
                setop = box.a
                is_upper = box.t in ["eq", "le", "lt"]
                is_lower = box.t in ["eq", "ge", "gt"]
            var = setop.get_variables()[0]
            value = const.to_diffsat(cache=False).detach().cpu().numpy()
            if is_lower:
                bounds[var][0].__setitem__(setop.args[1], value)
            if is_upper:
                bounds[var][1].__setitem__(setop.args[1], value)
    else:
        constraint = constraint.simplify(delete_box_constraints=True)
        variables = list(set(constraint.get_variables()))
        bounds = None
    return constraint, variables, bounds

def inner_opt(constraint, variables, bounds, args):
    if args.opt == 'lbfgsb':    
        sgd = optim.SGD([v.tensor for v in variables], lr=0.0)
        for i in range(args.opt_iterations):
            satisfied = constraint.to_diffsat(cache=True).satisfy(args)
            if satisfied:
                break
            lbfgsb(variables, bounds, lambda: constraint.to_diffsat(cache=True, reset_cache=True).loss(args), lambda: sgd.zero_grad())
    else:
        optimizer = args.opt([v.tensor for v in variables], lr=args.lr)
        for i in range(args.opt_max_iterations):
            satisfied = constraint.to_diffsat(cache=True).satisfy(args)
            if satisfied:
                break
            loss = constraint.to_diffsat(cache=True, reset_cache=True).loss(args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return satisfied

def x_to_vars(x, variables, shapes_flat, shapes):
    running_shape = 0
    with torch.no_grad():
        for i, var in enumerate(variables):
            val = x[running_shape:(running_shape + shapes_flat[i])]
            running_shape += shapes_flat[i]
            var.tensor[:] = torch.from_numpy(val.reshape(shapes[i]))

def vars_to_x(variables):
    np_vars = [var.tensor.detach().cpu().numpy() for var in variables]
    shapes = [var.shape for var in np_vars]
    shapes_flat = [var.size for var in np_vars]
    x = np.stack([var.ravel() for var in np_vars]).astype(np.float64)
    return x, shapes, shapes_flat


def basinhopping(constraint, variables, bounds, args):
    x0, shapes, shapes_flat = vars_to_x(variables)
    
    def loss_fn(x):
        x_to_vars(x, variables, shapes_flat, shapes)
        return constraint.to_diffsat(cache=True).loss(args)

    def local_optimization_step(fun, x0, *losargs, **loskwargs):
        loss_before = loss_fn(x0)
        inner_opt(constraint, variables, bounds, args)
        r = spo.OptimizeResult()
        r.x, _, _ = vars_to_x(variables)
        loss_after = constraint.to_diffsat(cache=True).loss(args)
        r.success = not (loss_before == loss_after and not constraint.to_diffsat(cache=True).satisfy(args))
        r.fun = loss_after
        return r

    def check_basinhopping(x, f, accept):
        if abs(f) <= 10 * args.eps:
            x_, _, _ = vars_to_x(variables)
            x_to_vars(x, variables, shapes_flat, shapes)
            if constraint.to_diffsat(cache=True).satisfy(args):
                return True
            else:
                x_to_vars(x_, variables, shapes_flat, shapes)
        return False
    
    minimizer_kwargs = {}
    minimizer_kwargs['method'] = local_optimization_step

    satisfied = constraint.to_diffsat(cache=True).satisfy(args)
    if satisfied:
        return True
    spo.basinhopping(loss_fn, x0, niter=1000, minimizer_kwargs=minimizer_kwargs, callback=check_basinhopping,
                     T=args.basinhopping_T, stepsize=args.basinhopping_stepsize)
    return constraint.to_diffsat(cache=True).satisfy(args)

class TimeoutException(Exception):
    pass

def solve(constraint, args, return_values=None):
    def solve_(constraint, args, return_values=None):
        t0 = time.time()
        if constraint is not None:
            constraint, variables, bounds = simplify(constraint, args)
            if args.use_basinhopping:
                satisfied = basinhopping(constraint, variables, bounds, args)
            else:
                satisfied = inner_opt(constraint, variables, bounds, args)
        else:
            satisfied = True

        if return_values is None:
            ret = dict([(v.name, v.tensor.detach().cpu().numpy()) for v in variables])
        else:
            ret = [(str(r), r.to_diffsat(cache=True).detach().cpu().numpy()) for r in return_values]
            if len(ret) == 1:
                ret = ret[0][1]
            else:
                ret = dict(ret)
        t1 = time.time()
        return satisfied, ret, t1 - t0

    def timeout(signum, frame):
        raise TimeoutException()

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout)
    try:
        solved, results, t = solve_(constraint, args, return_values=None)
    except TimeoutException:
        solved, results, t = False, None, args.timeout
    signal.alarm(0)  # cancel alarms
    torch.cuda.empty_cache()
    return solved, results, t

def lbfgsb(variables, bounds, loss_fn, zero_grad_fn):
    x, shapes, shapes_flat = vars_to_x(variables)
    bounds_list = []
    for var in variables:
        lower, upper = bounds[var]
        lower = lower.ravel()
        upper = upper.ravel()
        for i in range(lower.size):
            bounds_list.append((lower[i], upper[i]))

    def f(x):
        x_to_vars(x, variables, shapes_flat, shapes)
        loss = loss_fn()
        zero_grad_fn()
        loss.backward()
        with torch.no_grad():
            f = loss.detach().cpu().numpy().astype(np.float64)
            g = np.stack([var.tensor.grad.detach().cpu().numpy().ravel() for var in variables]).astype(np.float64)
        return f, g
    x, f, d = spo.fmin_l_bfgs_b(f, x, bounds=bounds_list)
    x_to_vars(x, variables, shapes_flat, shapes)
