import numpy as np
import torch


class DL2_Oracle:

    def __init__(self, learning_rate, net, constraint, use_cuda):
        self.learning_rate = learning_rate
        self.net = net
        self.constraint = constraint
        self.use_cuda = use_cuda

    def attack(self, x_batch, y_batch, domains, num_restarts, num_iters):
        n_batch = x_batch.size()[0]

        for retry in range(num_restarts):
            z_batch = np.concatenate([np.expand_dims(domains[i].sample(),axis=0) for i in range(n_batch)], axis=0)

            for it in range(num_iters):
                avg_neg_loss, avg_pos_loss, sat, z_inputs = self.loss(self.net, x_batch, y_batch, z_batch, self.use_cuda)
                avg_neg_loss.backward(retain_graph=True)
                z_batch_grad = np.sign(z_inputs.grad.data)
                z_batch -= self.learning_rate * z_batch_grad
                for i in range(n_batch):
                    z_batch[i] = domains[i].project(z_batch[i])

            return z_batch

    def general_attack(self, x_batches, y_batches, domains, num_restarts, num_iters, args):
        """ Minimizes DL2 loss with respect to z_1, ..., z_M.

        :param x_batches: List of N tensors, each tensor has shape batch_size x D
        :param y_batches: List of N tensors, each tensor has shape batch_size x num_classes
        :param domains: Nested list of Domain objects of shape M x batch_size, D_i is domain of variable z_i
        :param num_restarts: Number of times to restart the sampling
        :param num_iters: Number of iterations to perform in each restart
        :return: List of values for each of variables z_1, ..., z_M
        """
        n_gvars = len(domains)

        for retry in range(num_restarts):
            z_batches = [domains[j].sample() for j in range(n_gvars)]
            for it in range(num_iters):
                neg_losses, pos_losses, sat, z_inputs = self.constraint.loss(x_batches, y_batches, z_batches, args)

                avg_neg_loss = torch.mean(neg_losses)
                avg_neg_loss.backward()
                for i in range(n_gvars):
                    z_batches[i] -= self.learning_rate * torch.sign(z_inputs[i].grad.data)
                    z_batches[i] = domains[i].project(z_batches[i])
            return z_batches

    def evaluate(self, x_batches, y_batches, z_batches, args):
        neg_losses, pos_losses, sat, _ = self.constraint.loss(x_batches, y_batches, z_batches, args)
        if not isinstance(sat, np.ndarray):
            sat = sat.cpu().numpy()
        # constr_acc = np.mean(sat)
        # return torch.mean(neg_losses), torch.mean(pos_losses), constr_acc
        return neg_losses, pos_losses, sat
