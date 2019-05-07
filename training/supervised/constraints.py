import numpy as np
import torch
import torch.nn.functional as F
from domains import *
import sys
sys.path.append('../../')
import dl2lib as dl2


def kl(p, log_p, log_q):
    return torch.sum(-p * log_q + p * log_p, dim=1)

def transform_network_output(o, network_output):
    if network_output == 'logits':
        pass
    elif network_output == 'prob':
        o = [F.softmax(zo) for zo in o]
    elif inetwork_output == 'logprob':
        o = [F.log_sofmtax(zo) for zo in o]
    return o


class Constraint:

    def eval_z(self, z_batches):
        if self.use_cuda:
            z_inputs = [torch.cuda.FloatTensor(z_batch) for z_batch in z_batches]
        else:
            z_inputs = [torch.FloatTensor(z_batch) for z_batch in z_batches]

        for z_input in z_inputs:
            z_input.requires_grad_(True)
        z_outputs = [self.net(z_input) for z_input in z_inputs]
        for z_out in z_outputs:
            z_out.requires_grad_(True)
        return z_inputs, z_outputs

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        assert False

    def loss(self, x_batches, y_batches, z_batches, args):
        if z_batches is not None:
            z_inp, z_out = self.eval_z(z_batches)
        else:
            z_inp, z_out = None, None

        constr = self.get_condition(z_inp, z_out, x_batches, y_batches)
        
        neg_losses = dl2.Negate(constr).loss(args)
        pos_losses = constr.loss(args)
        sat = constr.satisfy(args)
            
        return neg_losses, pos_losses, sat, z_inp


class LipschitzConstraint(Constraint):

    def __init__(self, net, eps, l, use_cuda=True, network_output='logits'):
        self.net = net
        self.eps = eps
        self.l = l
        self.use_cuda = use_cuda
        self.network_output = network_output
        self.n_tvars = 2
        self.n_gvars = 2
        self.name = 'LipschitzG'

    def params(self):
        return {'L': self.l, 'network_output': self.network_output}

    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 2
        n_batch = x_batches[0].size()[0]
        return [[Box(np.clip(x_batches[j][i].cpu().numpy() - self.eps, 0, 1),
                    np.clip(x_batches[j][i].cpu().numpy() + self.eps, 0, 1))
                for i in range(n_batch)]
                for j in range(2)]

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        n_batch = z_inp[0].size()[0]
        z_out = transform_network_output(z_out, self.network_output)
        return dl2.LEQ(torch.norm(z_out[0] - z_out[1], p=2, dim=1),
                   self.l * torch.norm((z_inp[0] - z_inp[1]).view((n_batch, -1)), p=2, dim=1))

class PairLineRobustnessConstraint(Constraint):

    def __init__(self, net, eps, p_limit, use_cuda=True, network_output='logits'):
        self.use_cuda = use_cuda
        self.network_output = network_output
        print("Ignoring network_output argument, using logprob such that loss becomes cross-entropy")
        self.net = net
        self.eps = eps
        self.p_limit = p_limit

        self.n_tvars = 2
        self.n_gvars = 1
        self.name = 'SegmentG'

    def params(self):
        return {'eps': self.eps, 'p_limit': self.p_limit}

    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 2
        n_batch = x_batches[0].size()[0]
        return [[Segment(x_batches[0][i], x_batches[1][i]) for i in range(n_batch)]]

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        n_batch = x_batches[0].size()[0]
        z_logits = F.log_softmax(z_out[0], dim=1)

        d1 = torch.norm(z_inp[0] - x_batches[0], dim=1)
        d2 = torch.norm(z_inp[0] - x_batches[1], dim=1)

        w1 = d1 / (d1 + d2)
        w2 = d2 / (d1 + d2)

        pred_logits_1 = z_logits[np.arange(n_batch), y_batches[0]]
        pred_logits_2 = z_logits[np.arange(n_batch), y_batches[1]]

        pre = dl2.And([dl2.BoolConst(y_batches[0] != y_batches[1]),
                       dl2.LEQ(torch.norm((x_batches[0] - x_batches[1]).view((n_batch, -1)), dim=1), self.eps)])
        ce = -(w1 * pred_logits_1 + w2 * pred_logits_2)
        
        return dl2.Implication(pre, dl2.LT(ce, self.p_limit))
    
class RobustnessConstraint(Constraint):

    def __init__(self, net, eps, delta, use_cuda=True, network_output='logits'):
        self.net = net
        self.network_output = network_output
        self.eps = eps
        self.delta = delta
        self.use_cuda = use_cuda
        self.n_tvars = 1
        self.n_gvars = 1
        self.name = 'RobustnessG'

    def params(self):
        return {'eps': self.eps, 'network_output': self.network_output}

    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 1
        n_batch = x_batches[0].size()[0]

        return [[Box(np.clip(x_batches[0][i].cpu().numpy() - self.eps, 0, 1),
                     np.clip(x_batches[0][i].cpu().numpy() + self.eps, 0, 1))
                for i in range(n_batch)]]

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        n_batch = x_batches[0].size()[0]
        z_out = transform_network_output(z_out, self.network_output)[0]
        #z_logits = F.log_softmax(z_out[0], dim=1)
        
        pred = z_out[np.arange(n_batch), y_batches[0]]

        limit = torch.FloatTensor([0.3])
        if self.use_cuda:
            limit = limit.cuda()
        return dl2.GEQ(pred, torch.log(limit))


class LipschitzDatasetConstraint(Constraint):

    def __init__(self, net, l, use_cuda=True, network_output='logits'):
        self.net = net
        self.network_output = network_output
        self.l = l
        self.use_cuda = use_cuda
        self.n_tvars = 2
        self.n_gvars = 0
        self.name = 'LipschitzT'

    def params(self):
        return {'L': self.l, 'network_output': self.network_output}

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        n_batch = x_batches[0].size()[0]

        x_out1 = self.net(x_batches[0])
        x_out2 = self.net(x_batches[1])
        x_out1, x_out2 = transform_network_output([x_out1, x_out2], self.network_output)
        
        return dl2.LEQ(torch.norm(x_out1 - x_out2, p=2, dim=1),
                       self.l * torch.norm((x_batches[0] - x_batches[1]).view((n_batch, -1)), p=2, dim=1))

I = {
  'plane': 0,
  'car': 1,
  'bird': 2,
  'cat': 3,
  'deer': 4,
  'dog': 5,
  'frog': 6,
  'horse': 7,
  'ship': 8,
  'truck': 9,
}


class CifarDatasetConstraint(Constraint):

    def __init__(self, net, margin, use_cuda=True, network_output='logits'):
        self.net = net
        self.network_output = network_output
        self.margin = margin
        self.use_cuda = use_cuda
        self.n_tvars = 1
        self.n_gvars = 0
        self.name = 'CSimilarityT'

    def params(self):
        return {'delta': self.margin, 'network_output': self.network_output}

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        x_out = self.net(x_batches[0])
        x_out = transform_network_output([x_out], self.network_output)[0]
        targets = y_batches[0]

        rules = []
        rules.append(dl2.Implication(dl2.BoolConst(targets == I['car']), dl2.GEQ(x_out[:, I['truck']], x_out[:, I['dog']] + self.margin)))
        rules.append(dl2.Implication(dl2.BoolConst(targets == I['deer']), dl2.GEQ(x_out[:, I['horse']], x_out[:, I['ship']] + self.margin)))
        rules.append(dl2.Implication(dl2.BoolConst(targets == I['plane']), dl2.GEQ(x_out[:, I['ship']], x_out[:, I['frog']] + self.margin)))
        rules.append(dl2.Implication(dl2.BoolConst(targets == I['dog']), dl2.GEQ(x_out[:, I['cat']], x_out[:, I['truck']] + self.margin)))
        rules.append(dl2.Implication(dl2.BoolConst(targets == I['cat']), dl2.GEQ(x_out[:, I['dog']], x_out[:, I['car']] + self.margin)))
        return dl2.And(rules)


class CifarConstraint(Constraint):

    def __init__(self, net, eps, margin, use_cuda=True, network_output='logits'):
        self.net = net
        self.network_output = network_output
        self.eps = eps
        self.margin = margin
        self.use_cuda = use_cuda
        self.n_tvars = 1
        self.n_gvars = 1
        self.name = 'CSimilarityG'

    def params(self):
        return {'eps': self.eps, 'delta': self.margin, 'network_output': self.network_output}

    def get_domains(self, x_batches, y_batches):
        assert len(x_batches) == 1
        n_batch = x_batches[0].size()[0]

        return [[Box(np.clip(x_batches[0][i].cpu().numpy() - self.eps, 0, 1),
                     np.clip(x_batches[0][i].cpu().numpy() + self.eps, 0, 1))
                for i in range(n_batch)]]
    
    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        z_out = transform_network_output(z_out, self.network_output)[0]
        targets = y_batches[0]

        rules = []
        rules.append(dl2.Implication(dl2.BoolConst(targets == I['car']), dl2.GEQ(z_out[:, I['truck']], z_out[:, I['dog']] + self.margin)))
        rules.append(dl2.Implication(dl2.BoolConst(targets == I['deer']), dl2.GEQ(z_out[:, I['horse']], z_out[:, I['ship']] + self.margin)))
        rules.append(dl2.Implication(dl2.BoolConst(targets == I['plane']), dl2.GEQ(z_out[:, I['ship']], z_out[:, I['frog']] + self.margin)))
        rules.append(dl2.Implication(dl2.BoolConst(targets == I['dog']), dl2.GEQ(z_out[:, I['cat']], z_out[:, I['truck']] + self.margin)))
        rules.append(dl2.Implication(dl2.BoolConst(targets == I['cat']), dl2.GEQ(z_out[:, I['dog']], z_out[:, I['car']] + self.margin)))
        return dl2.And(rules)
     
class RobustnessDatasetConstraint(Constraint):

    def __init__(self, net, eps1, eps2, use_cuda=True, network_output='logits'):
        self.net = net
        self.network_output = network_output
        print("Ignoring network_output argument, using prob and logprob to obtain KL divergence")
        self.eps1 = eps1
        self.eps2 = eps2
        self.use_cuda = use_cuda
        self.n_tvars = 2
        self.n_gvars = 0
        self.name = 'RobustnessT'

    def params(self):
        return {'eps1': self.eps1, 'eps2': self.eps2}

    def get_condition(self, z_inp, z_out, x_batches, y_batches):
        n_batch = x_batches[0].size()[0]

        x_out1, x_out2 = self.net(x_batches[0]), self.net(x_batches[1])
        
        x_probs1 = F.softmax(x_out1, dim=1)
        x_logprobs1 = F.log_softmax(x_out1, dim=1)
        x_logprobs2 = F.log_softmax(x_out2, dim=1)

        kl_div = kl(x_probs1, x_logprobs1, x_logprobs2)

        close_x = dl2.LT(torch.norm((x_batches[0] - x_batches[1]).view((n_batch, -1)), dim=1), self.eps1)
        close_p = dl2.LT(kl_div, self.eps2)

        return dl2.Implication(close_x, close_p)
