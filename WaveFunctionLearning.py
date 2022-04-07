# -*- coding: iso-8859-1 -*-
from __future__ import division
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy.sparse import linalg
from numpy import linalg as la
import itertools
'''Reading the configuration file'''
import json
with open('Config.json') as f:
    config = json.load(f)


class Linear(object):
    def __init__(self, input_shape, output_shape, bias = 0., mean=0., variance=0.01):
        self.bias = bias
        self.parameters = [mean + variance * np.random.randn(input_shape, output_shape),
                           mean + variance * np.random.randn(output_shape)]
        self.parameters_deltas = [None, None]

    def forward(self, x, *args):
        self.x = x
        return np.matmul(x, self.parameters[0]) + self.bias*self.parameters[1]

    def backward(self, delta):
        self.parameters_deltas[0] = self.x.T.dot(delta)
        self.parameters_deltas[1] = np.sum(delta, 0)
        return delta.dot(self.parameters[0].T)


class Convolution(object):
    def __init__(self, kernel_size, output_shape, bias = 0., mean=0., variance=0.01):
        self.kernel_size = kernel_size
        self.bias = bias
        self.output_shape = output_shape
        self.parameters = [mean + variance * np.random.randn(kernel_size, output_shape),
                           mean + variance * np.random.randn(output_shape)]
        self.parameters_deltas = [None, None]

    def vroll(self, i):
        return np.matmul(np.roll(self.x,i,axis = 1)[:,0:self.kernel_size],self.parameters[0])

    def forward(self, x, *args):
        self.x = x
        return sum(imap(self.vroll,range(len(x[0])))) + self.bias*self.parameters[1]

    def backward(self, delta):
        xshape = self.x.shape[1]
        self.parameters_deltas[0] = np.pad(np.sum(self.x, axis=1).reshape((len(delta), 1)),
                                        ((0, 0), (self.kernel_size - 1, 0)), 'edge').T.dot(delta)
        self.parameters_deltas[1] = np.sum(delta, 0)
        back_par1 = np.sum(self.parameters[0].T, axis=1) * int(xshape / self.kernel_size)
        back_par = np.array([np.sum(np.roll(self.parameters[0].T, i, axis=1)[:, 0:(xshape % self.kernel_size)], axis=1) +
                             back_par1 for i in range(xshape)])
        return delta.dot(back_par.T)


class lrf_Linear(object):
    def __init__(self, system_size, receptive_field, overlap, output_dim, bias = 0., mean=0., variance=0.01):
        self.bias = bias
        self.output_dim = output_dim
        self.overlap = overlap
        self.receptive_field = receptive_field
        self.system_size = system_size
        self.shift = self.receptive_field-self.overlap
        self.shifts = np.arange(0,self.system_size,self.shift)
        self.parameters = [mean + variance * np.random.randn(receptive_field, output_dim),
                           mean + variance * np.random.randn(output_dim)]
        self.parameters_deltas = [None, None]

    def vroll(self, i):
        return np.matmul(np.roll(self.x,i*self.shift,axis = 1)[:,0:self.receptive_field],self.parameters[0][:,i])

    def groll(self, i):
        return np.sum(np.roll(self.x,i*self.shift,axis = 1)[:,0:self.receptive_field]*self.delta[:,i].reshape((len(self.x),1)),axis=0)

    def forward(self, x, *args):
        self.x = x
        return np.array(list(imap(self.vroll,range(len(self.shifts))))).T + self.bias * self.parameters[1]

    def backward(self, delta,*args):
        self.delta = delta
        self.parameters_deltas[0] = np.array(list(imap(self.groll,range(self.output_dim)))).T
        self.parameters_deltas[1] = np.sum(delta, 0)
        return delta.dot(self.parameters[0].T)


class F(object):
    '''base class for functions with no parameters.'''

    def __init__(self):
        self.parameters = []
        self.parameters_deltas = []


class Tanh(F):
    def forward(self, x, *args):
        self.x = x
        self.y = np.tanh(x)
        return self.y

    def backward(self, delta):
        return delta * (1. - np.power(self.y, 2.))


class ReLu(F):
    def forward(self, x, *args):
        self.x = x
        self.y = np.maximum(0., x)
        return self.y

    def backward(self, delta):
        return delta * np.heaviside(self.y,0.)


class Triangle(F):
    def forward(self,x ,*args):
        self.x = x
        self.y = np.mod(x , 2. * np.sign(x) ) - np.sign(x)
        return 2. * np.absolute( self.y ) - 1.

    def backward(self, delta):
        return delta*2.*np.sign(self.y)


class Sigmoid(F):
    def forward(self, x, *args):
        self.x = x
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, delta):
        return delta * ((1. - self.y) * self.y)


class Sinc(F):
    def forward(self,x ,*args):
        self.x = x
        self.y = np.sin(self.x)/self.x
        return self.y

    def backward(self, delta):
        return delta*(np.cos(self.x)/self.x - self.y/self.x)


class Cos(F):
    def forward(self,x ,*args):
        self.x = x
        return np.cos(self.x)

    def backward(self, delta):
        return delta*(-1.)*np.sin(self.x)


class Softmax(F):
    def forward(self, x, *args):
        self.x = x
        xtmp = x - x.max(axis=-1, keepdims=True)  # to avoid overflow
        exps = np.exp(xtmp)
        self.y = exps / exps.sum(axis=-1, keepdims=True)
        return self.y

    def backward(self, delta):
        return delta * self.y - self.y * (delta * self.y).sum(axis=-1, keepdims=True)


class Energy(object):
    def __init__(self, Ham):
        self.h = Ham.T + Ham

    def forward(self, x):
        self.x = x.T[0]
        self.u = self.x.dot(H.dot(self.x))
        self.v = la.norm(self.x) ** 2
        self.e = self.u / self.v
        return self.e

    def backward(self):
        U = self.h.dot(self.x)
        V = 2. * self.x
        return ((U * self.v - V * self.u) / self.v ** 2).reshape((nstates, 1))


class CrossEntropy(F):
    def forward(self, x, p, *args):
        # p is target probability. In MNIST dataset, it represents a one-hot label.
        self.p = p
        self.x = x
        y = -p * np.log(np.maximum(x, 1e-15))
        return y.sum(-1)

    def backward(self, delta):
        return -delta[..., None] * 1. / np.maximum(self.x, 1e-15) * self.p


class Mean(F):
    def forward(self, x, *args):
        self.x = x
        return x.mean()

    def backward(self, delta):
        return delta * np.ones(self.x.shape) / np.prod(self.x.shape)


class SGD(object):
    def __init__(self, learning_rate=0.01, decay=1e-4, net_conf=None):
        self.learning_rate = [learning_rate] * len(net_conf)
        self.decay = decay

    def update(self, g, node, par):
        node = int(node/2.)
        self.learning_rate[node] *= 1. / (1. + self.decay)
        return -self.learning_rate[node] * g


class Adagrad(object):
    def __init__(self, learning_rate=0.01, epsilon=1e-7, net_conf=None):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.g_sum = [[np.zeros(net_conf[i][0]), np.zeros(net_conf[i][1])] for i in range(len(net_conf))]

    def gradient_square_sum(self, g, node, par):
        self.g_sum[node][par] += np.square(g)

    def adapted_lr(self, node, par):
        return self.learning_rate / (np.sqrt(self.g_sum[node][par]) + self.epsilon)

    def update(self, g, node, par):
        node = int(node/2.)
        self.gradient_square_sum(g, node, par)
        return -self.adapted_lr(node, par) * g


class Adadelta(object):
    def __init__(self, gamma, epsilon, net_conf):
        self.gamma = gamma
        self.epsilon = epsilon
        self.Eg = [[np.zeros(net_conf[i][0]), np.zeros(net_conf[i][1])] for i in range(len(net_conf))]
        self.EdO = [[np.zeros(net_conf[i][0]), np.zeros(net_conf[i][1])] for i in range(len(net_conf))]

    def moving_average_g(self, g, node, par):
        self.Eg[node][par] = self.gamma * self.Eg[node][par] + (1 - self.gamma) * (np.square(g))

    def moving_average_dO(self, dO, node, par):
        self.EdO[node][par] = self.gamma * self.EdO[node][par] + (1 - self.gamma) * (np.square(dO))

    def learning_rate(self, node, par):
        RMS_g = np.sqrt(self.Eg[node][par] + self.epsilon)
        RMS_dO = np.sqrt(self.EdO[node][par] + self.epsilon)
        return -RMS_dO / RMS_g

    def update(self, g, node, par):
        node = int(node/2.)
        self.moving_average_g(g, node, par)
        dO = self.learning_rate(node, par) * g
        self.moving_average_dO(dO, node, par)
        return dO

def get_net_conf(net):
    conf = []
    for i in np.arange(0,len(net[0:-1]),2):
        w = np.array(net[i].parameters[0])
        b = np.array(net[i].parameters[1])
        conf.append([w.shape, b.shape])
    return conf

class early_stop(object):
    def __init__(self,threshold = 5e-16, gamma = 0.9999):
        self.threshold = threshold
        self.gamma = gamma
        self.moving_average = 2.

    def learnstop(self, loss):
        self.moving_average = self.gamma * self.moving_average + (1 - self.gamma)*loss
        if abs(self.moving_average-loss) < self.threshold: return True
        return False

'''Network'''
def make_net(layer_conf):
    layer_conf = list(layer_conf)
    layer_conf.append(1)
    layer_conf.append(N)
    architecture = config["Network"]["Architecture"]
    net = []
    '''Here we use globals() which is a dictonary containing all callable global functions'''
    for i in np.arange(0,len(architecture),3):
        net.append(globals()[architecture[i]](layer_conf[int(i/3)-1],layer_conf[int(i/3)],architecture[i+1]))
        net.append(globals()[architecture[i+2]]())
    net.append(globals()[config["Network"]["Loss"]](H))
    return net


'''Pre-Training'''

'''Actual Restricted Boltzmann Machine'''
def p_cosh(b,W,pre_states):
    return np.prod(2*np.cosh(np.matmul(pre_states, W)+b),axis = 1)#b, ausschalten?

'''Calculating the Log Likelihood'''
def unsupervised_loglike(T,*args):
    layer1 = args[0]
    layer2 = args[1]
    pre_states = args[2]
    b, W = T[0:layer2], np.reshape(T[layer2::], (layer1, layer2))
    psi = p_cosh(np.zeros(len(b)),W,pre_states)
    u = np.log(psi/np.sum(psi))
    return (-1.)*np.sum(u)

'''Optimizing the Log Likelihood'''
def variation(f,layer1,layer2,pre_states,i):
    b_ = config["Network"]["Architecture"][3*i+1]*np.random.rand(layer2)
    W_ = np.random.rand(layer1, layer2)
    args = (layer1,layer2,pre_states)
    res = sp.optimize.minimize(f, np.concatenate((b_, np.reshape(W_, layer1 * layer2))), args,
                               method='BFGS', tol=1.E-15, options={'gtol': 1E-15, 'disp': False, 'maxiter': 500000})
    return f(res.x,*args), res.x

'''Layer-wise pre-training'''
def pre_train(net,layer_conf):
    layer_conf = list(layer_conf)
    layer_conf.append(1)
    layer_conf.append(N)
    nlayer = len(layer_conf)-1
    log = np.zeros(nlayer)
    pre_states = states
    for i in range(0,nlayer):
        layer1 = layer_conf[i-1]
        layer2 = layer_conf[i]
        lossen, pre_tr_W = variation(unsupervised_loglike,layer1,layer2,pre_states,i)
        log[i] = lossen
        net[2*i].parameters = [np.reshape(pre_tr_W[layer2::], (layer1, layer2)), pre_tr_W[0:layer2]]
        '''Evaluate the linear function'''
        pre_states = net[2*i].forward(pre_states)
        '''Evaluate the nonlinear function'''
        pre_states = net[2*i+1].forward(pre_states)
    return log


def training(net, opt, training_set, num_epoch):
    result = net_forward(training_set)
    learning_curve = np.ones((2,num_epoch))
    print('Before Training.\nTest loss = %.4f, energy = %.3f' % (loss(result), result))
    for epoch in range(num_epoch):
        '''even though we do not use result, one forward pass is required'''
        result = net_forward(states)
        net_backward()
        '''update network parameters'''
        for node in np.arange(0, len(net[0:-1]), 2):
            update = net[node].parameters_deltas
            net[node].parameters[0] += opt.update(update[0], node, 0)
            net[node].parameters[1] += opt.update(update[1], node, 1)
        result = net_forward(states)
        learning_curve[:,epoch] = result, loss(result)
        '''early training abort'''
        if loss(result) < config["Test"]["Precision"]: break
    print('After Training.\nTest loss = %.16f, energy = %.3f' % (loss(result), result))
    return np.min(learning_curve[0][~np.isnan(learning_curve[0])]), np.min(learning_curve[1][~np.isnan(learning_curve[1])]), learning_curve


def reset_net(net):
    conf = get_net_conf(net)
    mean = 0.
    variance = 0.1
    for node in np.arange(0, int(len(net[0:-1])/2), 1):
        net[2*node].parameters = [mean + variance * np.random.randn(conf[node][0][0],conf[node][0][1]),
                                mean + variance * np.random.randn(conf[node][1][0])]
        net[node].parameters_deltas = [None,None]


def net_forward(x):
    for node in net[0:-1]:
        x = node.forward(x)
    return net[-1].forward(x)


def net_backward():
    y_delta = net[-1].backward()
    for node in reversed(net[0:-1]):
        y_delta = node.backward(y_delta) 
    return y_delta


def wavefunc(x):
    for node in net[0:-1]:
      x = node.forward(x)
    return x.T[0]/la.norm(x.T[0])


def loss(en):
    return (E_ED-en)/E_ED


def get_weights(net):
    weights = []
    for i in np.arange(0, len(net[0:-1]), 2):
        w = np.array(net[i].parameters[0])
        b = np.array(net[i].parameters[1])
        weights.append([w, b])
    return weights


'''System configuration'''
N = config["System"]["N"]
if config["System"]["SignTransform"]:
    if config["System"]["TotalSz"] == "0":
        from AFH_Positive import AFH_Positive as AFH
        ham_short = 'AFH-Sz0-p'
    else:
        from AFH_Positive_fb import AFH_Positive as AFH
        ham_short = 'AFH-p'
else:
    if config["System"]["TotalSz"] == "0":
        from AFH_Negative import AFH_Negative as AFH
        ham_short = 'AFH-Sz0-pm'
    else:
        from AFH_Negative_fb import AFH_Negative as AFH
        ham_short = 'AFH-pm'

nstates, states, H, E_ED, Psi_ED = AFH(N).getH()


'''Test settings'''
num_epoch = int(config["Test"]["Epochs"])
L_sizes_set = [np.arange(config["Test"]["L_min"][i],config["Test"]["L_max"][i],config["Test"]["Steps"])
               for i in range(len(config["Test"]["L_max"]))]
config_set = list(itertools.product(*L_sizes_set))


'''Preparing output files'''
precision_log = np.ones(len(config_set))
histogram_log = np.ones((len(config_set),config["Test"]["Repetitions"]))
weights_log = []
learning_log = []

for conf in range(len(config_set)):
    weights_log.append([])
    learning_log.append([])
    for i in range(config["Test"]["Repetitions"]):
        net = make_net(config_set[conf])
        net_conf = get_net_conf(net)
        optmizer = Adagrad(0.1, 1e-7, net_conf)
        if  config["Test"]["Pre-Training"]: ptr_out = pre_train(net, config_set[conf])
        en, lossen, learning_curve = training(net,optmizer, states, num_epoch)
        histogram_log[conf,i] = lossen
        if lossen < precision_log[conf]:
            precision_log[conf] = lossen
            del weights_log[-1]
            weights_log.append(get_weights(net))
            del learning_log[-1]
            learning_log.append(learning_curve)
        np.savez(config["Network"]["Name"]+'_'+ham_short+'_N' + str(N) + '_PreTrain-'+str(config["Test"]["Pre-Training"])+'_PlotFile',
                 config_set,precision_log,histogram_log,weights_log,learning_log)
