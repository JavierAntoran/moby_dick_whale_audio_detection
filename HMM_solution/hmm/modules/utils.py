from __future__ import print_function
import ConfigParser
import collections
import h5py, sys
import numpy as np
import torch
from torch.autograd import Variable
import gzip
import os
import math
from torch.optim import Optimizer

try:
    import cPickle as pickle
except:
    import pickle


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)

def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out

class AdamW(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1


                p.data.addcdiv_(-step_size, exp_avg, denom)

                if group['weight_decay'] != 0:
                    decay_step_size = -step_size * group['weight_decay']
                    p.data.add_(decay_step_size, p.data)

        return loss

suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']


def humansize(nbytes):
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes)
    return '%s%s' % (f, suffixes[i])


# ----------------------------------------------------------------------------------------------------------------------
def cprint(color, text, **kwargs):
    if color[0] == '*':
        pre_code = '1;'
        color = color[1:]
    else:
        pre_code = ''
    code = {
        'a': '30',
        'r': '31',
        'g': '32',
        'y': '33',
        'b': '34',
        'p': '35',
        'c': '36',
        'w': '37'
    }
    print("\x1b[%s%sm%s\x1b[0m" % (pre_code, code[color], text), **kwargs)
    sys.stdout.flush()


# ----------------------------------------------------------------------------------------------------------------------
def one_hot(y, nb_classes):
    nb_samples = y.shape[0]
    Y = np.zeros((nb_samples, nb_classes))
    Y[np.arange(nb_samples), y] = 1
    return Y


def torch_onehot(y, Nclass, cuda=True):
    if cuda:
        y_onehot = torch.FloatTensor(y.shape[0], Nclass).cuda()
    else:
        y_onehot = torch.FloatTensor(y.shape[0], Nclass)
    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    return y_onehot


def save_obj(obj, filename):
    with open(filename, 'w') as f:
        pickle.dump(obj, f, protocol=2)


def load_obj(file):
    if not isinstance(file, str):
        return pickle.load(f)

    root, ext = os.path.splitext(file)
    if ext == '.gz':
        with gzip.open(file, 'r') as f:
            return pickle.load(f)
    else:
        with open(file, 'r') as f:
            return pickle.load(f)

def windower(x, M, N):
    # M avance entre vetanas
    # N windowsize

    T   = x.shape[0]
    m   = np.arange(0, T-N+1, M) # comienzos de ventana
    L   = m.shape[0] # N ventanas
    ind = np.expand_dims(np.arange(0, N), axis=1) * np.ones((1,L)) + np.ones((N,1)) * m
    X   = x[ind.astype(int)]
    return X.transpose()

def shuffle_in_unison(a, b, c=None):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    if c is not None:
        np.random.set_state(rng_state)
        np.random.shuffle(c)


def gen_crossval_split(data, labels, Npart, Nparts, shuffle=False):
    ndat = int(data.shape[0] / Nparts)
    assert ndat * Nparts == data.shape[0]

    i = Npart
    d0 = data[:ndat * (i)]
    d1 = data[ndat * (i + 1):]
    l0 = labels[:ndat * (i)]
    l1 = labels[ndat * (i + 1):]

    return np.concatenate((d0, d1), axis=0), np.concatenate((l0, l1)), data[ndat *(i):ndat * (i + 1)], labels[ndat * (
        i):ndat * (i + 1)]