from .util import get_fqn
from . import api as q
import torch.nn
import os
from textx.metamodel import metamodel_from_str
import numpy as np

class Scope():

    def __init__(self, context):
        self.scope = dict(context)

    def add(self, val):
        name = val.name
        if name in self.scope:
            print(f"Trying to redefine element {name} in scope")
            return
        self.scope[name] = val

    def get_models(self):
        return dict(filter(lambda x: isinstance(x[1], torch.nn.Module), self.scope.items()))
        
    def get_variables(self):
        return dict(filter(lambda x: isinstance(x[1], q.Variable), self.scope.items()))

    def get(self, name):
        return self.scope.get(name)

class Parser:

    def __init__(self, query, context, args):
        curr_dir = os.path.dirname(__file__)
        with open(os.path.join(curr_dir, 'language.tx'), 'r') as f:
            grammar = f.read()
        mm = metamodel_from_str(grammar)
        self.parse_tree = mm.model_from_str(query)
        self.scope = Scope(context)
        self.args = args

    def parse(self):
        if self.parse_tree.find:
            return self.generate_find()
        if self.parse_tree.eval:
            return None, [self.traverse_expression(self.parse_tree.eval.exp)]
        else:
            raise NotImplementedError()

    def generate_find(self):
        self.traverse_variable_declarations(self.parse_tree.find.variable_declarations)
        constraints = self.traverse_constraints(self.parse_tree.find.constraints.constraints)
        if self.parse_tree.find.variable_initialization:
            self.traverse_variable_initialization(self.parse_tree.find.variable_initialization.initializations)
        if self.parse_tree.find.return_values:
            return_values = self.traverse_return_values(self.parse_tree.find.return_values)
        else:
            return_values = None
        return constraints, return_values

    def traverse_variable_declarations(self, variable_declarations):
        declarations = variable_declarations.declarations
        for declaration in declarations:
            self.traverse_variable_declaration(declaration)

    def traverse_variable_declaration(self, declaration):
        name = declaration.identifier
        shape = tuple(declaration.shape.dims)
        self.scope.add(q.Variable(name, shape, cuda=self.args.cuda))

    def traverse_return_values(self, returns):
        return_values = []
        for value in returns.values:
            return_values.append(self.traverse_expression(value))
        return return_values

    def traverse_function_application(self, function_name, args, layer, index):
        models = self.scope.get_models()
        args = [self.traverse_expression(arg) for arg in args]
        if function_name in models:
            ret = q.Model(models[function_name])
            if layer:
                ret = eval(f"ret.{layer}")
            ret = ret(*args)
            if index:
                ret = self.traverse_index(ret, index)
            return ret
        else:
            assert layer is None or layer.strip() == ''
            if function_name == 'class':
                return q.Class(*args)
            elif function_name == 'sum':
                return q.Fn('sum', lambda a: torch.sum(a), *args)
            elif function_name == 'abs':
                return q.Fn('abs', lambda a: torch.abs(a), *args)
            elif function_name == 'norm1':
                return q.Fn('norm1', lambda a: a.norm(1), *args)
            elif function_name == 'norm2':
                return q.Fn('norm2', lambda a: a.norm(2), *args)
            elif function_name == 'normInf':
                return q.Fn('normInf', lambda a: a.norm(float('inf')), *args)
            elif function_name == 'argmax':
                return q.Fn('argmax', lambda a: a.argmax(), *args)
            elif function_name == 'clamp':
                return q.Fn('clamp', lambda a, b, c: a.clamp(min=b, max=c), *args)
            else:
                assert False
        
    def traverse_expression(self, exp):
        def traverse_term(term):
            if term.op == '*':
                lhs = traverse_factor(term.factor)
                rhs = traverse_factor(term.rhs)
                return lhs * rhs
            else:
                return traverse_factor(term.factor)

        def traverse_factor(factor):
            if factor.function:
                return self.traverse_function_application(factor.function, factor.args, factor.layer, factor.index)
            elif factor.op:
                return self.traverse_operand(factor.op)
            else:
                return self.traverse_expression(factor.exp)

        if exp.op in ['+', '-']:
            lhs = traverse_term(exp.term)
            rhs = self.traverse_expression(exp.exp)
            return eval(f"lhs {exp.op} rhs")
        else:
            return traverse_term(exp.term)

    def traverse_constraint(self, constraint):
        if constraint.is_class:
            return self.traverse_function_application('class', constraint.args, None, None)
        lhs = self.traverse_expression(constraint.lhs)
        rhs = self.traverse_expression(constraint.rhs)
        if constraint.op == 'in':
            return lhs.in_(rhs)
        if constraint.op == '=':
            return lhs.eq_(rhs)
        return eval(f"lhs {constraint.op} rhs")
        
    def traverse_constraints(self, constraints):
        ast_constraints = []
        for constraint in constraints:
            if constraint.is_disjunction:
                c1 = self.traverse_constraint(constraint.c1)
                c2 = self.traverse_constraint(constraint.c2)
                ast_constraints.append(q.Or(c1, c2))
            else:
                c1 = self.traverse_constraint(constraint.c1)
                ast_constraints.append(c1)
        return q.And(*ast_constraints)

    def traverse_interval(self, interval):
        start = interval.start
        end = interval.end
        if end < start:
            raise ValueError(f"[{start}, {end}] is not a valid interval")
        assert start < end
        return q.Interval(start, end)

    def traverse_variable(self, var):
        name = var.identifier
        variable = self.scope.get(name)
        if not isinstance(variable, q.Variable):
            variable = q.Constant(variable, cuda=self.args.cuda)
        if var.index:
            return self.traverse_index(variable, var.index)
        else:
            return variable

    def traverse_index(self, val, idx):
        if idx.var:
            index = self.traverse_variable(idx.var).value
            if isinstance(index, np.ndarray) and index.dtype == np.bool_:
                index = np.where(index)
            return eval(f"val[index]")
        elif idx.val:
            return eval(f"val[{idx.val}]")
        
    def traverse_operand(self, operand):
        if get_fqn(operand) == 'Operand':
            operand = operand.val
        fqn = get_fqn(operand)
        if fqn == 'Constant':
            return self.traverse_constant(operand)
        elif fqn == 'Variable':
            return self.traverse_variable(operand)
        elif fqn == 'Interval':
            return self.traverse_interval(operand)
        else:
            raise TypeError(f"data type {fqn} not supported as operand: {operand}")
        return operand

    def traverse_external_variable(self, external_var):
        return self.scope.get(external_var.name)

    def traverse_constant(self, const):
        return q.Constant(const.value, cuda=self.args.cuda)

    def traverse_variable_initialization(self, initializations):
        for initialization in initializations:
            var = self.traverse_variable(initialization.var)
            rhs = self.traverse_operand(initialization.rhs)
            var.init(rhs)
