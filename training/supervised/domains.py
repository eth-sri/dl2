import numpy as np
import torch
import torch.nn.functional as F

from gurobipy import GRB, LinExpr


class Box:

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.name = 'box'

    def project(self, x):
        return torch.max(torch.min(x, self.b), self.a)

    def sample(self):
        return (self.b - self.a) * torch.rand(self.a.size()).to(self.a.device) + self.a


class CategoricalBox:

    def __init__(self, a, b, cat):
        self.a = a
        self.b = b
        self.cat = cat
        self.name = 'categorical_box'

    def project(self, x):
        x = torch.max(torch.min(x, self.b), self.a)
        for c, ids in self.cat.items():
            x[:, ids] = F.normalize(x[:, ids], p=1, dim=1)
        return x

    def sample(self):
        x = (self.b - self.a) * torch.rand(self.a.size()).to(self.a.device) + self.a
        batch_size = self.a.size()[0]
        for c, ids in self.cat.items():
            nz = torch.randint(ids[0], ids[-1] + 1, (batch_size,))
            x[:, ids] = 0.0
            x[np.arange(batch_size), nz] = 1.0
        return x

    def get_grb_vars(self, grb_model):
        x, all_cat_ids = [], []
        for ids in self.cat.values():
            all_cat_ids += ids
        for i in range(self.a.shape[1]):
            if i in all_cat_ids:
                x.append(grb_model.addVar(0, 1, vtype=GRB.BINARY, name='x_-1_{}'.format(i)))
            else:
                x.append(grb_model.addVar(self.a[0, i], self.b[0, i], name='x_-1_{}'.format(i)))
        for ids in self.cat.values():
            cat_sum = LinExpr()
            for i in ids:
                cat_sum += x[i]
            grb_model.addConstr(cat_sum, GRB.EQUAL, 1)
        return x


class Segment:

    def __init__(self, p1, p2):
        self.p1_n = p1.cpu().numpy()
        self.p2_n = p2.cpu().numpy()
        self.d = self.p2_n - self.p1_n
        self.d_norm = self.d / np.linalg.norm(self.d)
        self.name = 'segment'

    def is_empty(self):
        return False

    def project(self, x):
        dp = np.sum((x - self.p1_n) * self.d_norm)
        if dp < 0:
            return self.p1_n
        elif dp > np.linalg.norm(self.d):
            return self.p2_n
        else:
            return self.p1_n + dp * self.d_norm

    def sample(self):
        return self.p1_n + (self.p2_n - self.p1_n) * np.random.random()
