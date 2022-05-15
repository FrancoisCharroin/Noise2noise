#import torch.tensor
from torch import empty,cat,arange, Tensor
from torch.nn.functional import fold , unfold
import math

class Module(object):
    def forward ( self , * input ) :
        raise NotImplementedError
    def backward ( self , * gradwrtoutput ) :
        raise NotImplementedError
    def param ( self ) :
        return []



class StructFoo(object):
    pass



class SGD():
    def __init__(self, lr=0.01, momentum=0.0, clip_norm=None, lr_scheduler=None, **kwargs):
        
        C = self.cache
        H = self.hyperparameters
        momentum, clip_norm = H["momentum"], H["clip_norm"]
        lr = self.lr_scheduler(self.cur_step, cur_loss)

        if param_name not in C:
            C[param_name] = zeros_like(param_grad)

        # scale gradient to avoid explosion
        t = math.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        update = momentum * C[param_name] + lr * param_grad
        self.cache[param_name] = update
        return param - update
               
class Sequential(Module):
    def __init__(self, *modules):
        self.modules = modules

    def forward(self, data_in):
        x = self.modules[0].forward(data_in)
        for module in self.modules[1:]:
            x = module.forward(x)
        return x

    def backward(self, grad_dl_dout):
        grad = self.modules[-1].backward(grad_dl_dout)
        for module in reversed(self.modules[:-1]):
            grad = module.backward(grad)
        return grad

    def sgd(self, eta, momentum=0.):
        for module in self.modules:
            module.sgd(eta, momentum)

    def param(self):
        return [module.param() for module in self.modules]

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def reset_params(self):
        for module in self.modules:
            module.reset_params()
            module.zero_grad()

class ReLU(Module):
    def __init__(self):
        self.grad = StructFoo()
        self.grad.data_in = None

    def forward(self, data_in):
        self.grad.data_in = data_in
        return max(self.grad.data_in, torch.tensor(0.))

    def backward(self, grad_dl_dout):
        return grad_dl_dout * (self.grad.data_in > 0)

    def reset_params(self):
        self.grad.data_in = None



class Sigmoid(Module):
    def __init__(self):
        self.grad = StructFoo()
        self.grad.data_in = None

    def forward(self, data_in):
        self.grad.data_in = data_in
        return 1 / (1 + math.exp(-self.grad.data_in))

    def backward(self, grad_dl_dout):
        return grad_dl_dout * (1 / (1 + math.exp(-self.grad.data_in))) * (1 - 1 / (1 + math.exp(-self.grad.data_in)))

    def reset_params(self):
        self.grad.data_in = None

class LossMSE(Module):
    def __init__(self):
        pass

    def forward(self, output, target):
        return math.pow(target - output, 2).mean()

    def backward(self, output, target):
        return 2 * (output - target).div(output.size(0))

