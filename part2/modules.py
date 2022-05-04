import torch.tensor
import math

class Module(object):
    def forward(self, *data_in):
        raise NotImplementedError

    def backward(self, *grad_dl_dout):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        pass

    def sgd(self, eta, momentum):
        pass


class StructFoo(object):
    pass


class Linear(Module):
    def __init__(self, dim_in, dim_out, non_linearity="linear", slope=0.2):
        """PARAM NOT FOUND FOR ELU"""
        if non_linearity == "tanh":
            self.xavier_gain = 5 / 3
        elif non_linearity == "relu":
            self.xavier_gain = math.sqrt(2)
        elif non_linearity == "selu":
            self.xavier_gain = 3 / 4
        elif non_linearity == "elu":
            self.xavier_gain = 1
        elif non_linearity == "leaky_relu":
            self.xavier_gain = math.sqrt(2 / (1 + math.pow(slope, 2)))
        elif non_linearity == "linear" or "sigmoid" or "conv":
            self.xavier_gain = 1
        else:
            raise NotImplementedError

        self.params = StructFoo()

        # Xavier initialization
        self.params.weights = torch.empty((dim_out, dim_in)).normal_(0, self.xavier_gain * math.sqrt(
            2 / (dim_in + dim_out)))
        self.params.biases = torch.empty(dim_out).normal_(0, 1)

        self.grad = StructFoo()
        self.grad.weights = torch.zeros_like(self.params.weights)
        self.grad.biases = torch.zeros_like(self.params.biases)

        self.grad.weights_speed = torch.zeros_like(self.params.weights)
        self.grad.biases_speed = torch.zeros_like(self.params.biases)

    def forward(self, data_in):
        self.grad.data_in = data_in
        out = data_in @ self.params.weights.t() + self.params.biases
        return out

    def backward(self, grad_dl_dout):
        """grad_dl_out derivatative of the loss w.r.t the activation function"""
        self.grad.biases.add_(grad_dl_dout.sum())
        grad_dl_dw = grad_dl_dout.t() @ self.grad.data_in
        self.grad.weights.add_(grad_dl_dw)
        grad_dl_din = grad_dl_dout @ self.params.weights
        return grad_dl_din

    def sgd(self, eta, momentum=0.):

        self.grad.weights_speed.mul_(momentum).add_(self.grad.weights * eta)
        self.grad.biases_speed.mul_(momentum).add_(self.grad.biases * eta)

        self.params.weights.sub_(self.grad.weights_speed)
        self.params.biases.sub_(self.grad.biases_speed)

    def param(self):
        return [(self.params.weights, self.grad.weights, self.grad.weights_speed),
                (self.params.biases, self.grad.biases, self.grad.biases_speed)]

    def zero_grad(self):
        self.grad.weights.zero_()
        self.grad.biases.zero_()

    def reset_params(self):
        dim_out, dim_in = self.params.weights.shape

        self.params.weights = torch.empty((dim_out, dim_in)).normal_(0, self.xavier_gain * math.sqrt(
            2 / (dim_in + dim_out)))
        self.params.biases = torch.empty(dim_out).normal_(0, 1)

        self.grad.weights_speed = torch.zeros_like(self.params.weights)
        self.grad.biases_speed = torch.zeros_like(self.params.biases)


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


""" Activations Functions"""


class ReLU(Module):
    def __init__(self):
        self.grad = StructFoo()
        self.grad.data_in = None

    def forward(self, data_in):
        self.grad.data_in = data_in
        return torch.max(self.grad.data_in, torch.tensor(0.))

    def backward(self, grad_dl_dout):
        return grad_dl_dout * (self.grad.data_in > 0)

    def reset_params(self):
        self.grad.data_in = None


class LeakyReLU(Module):
    """common value of slope : [0.1 - 0.3]"""

    def __init__(self, slope=0.2):
        self.grad = StructFoo()
        self.grad.data_in = None
        self.slope = slope

    def forward(self, data_in):
        self.grad.data_in = data_in
        return torch.max(self.grad.data_in, torch.tensor(0.)) + torch.min(self.slope * self.grad.data_in,
                                                                          torch.tensor(0.))

    def backward(self, grad_dl_dout):
        return grad_dl_dout * ((self.grad.data_in > 0) + self.slope * (self.grad.data_in < 0))

    def reset_params(self):
        self.grad.data_in = None


class ELU(Module):
    """common value of alpha : [0.1 - 0.3]"""

    def __init__(self, alpha=0.2):
        self.grad = StructFoo()
        self.grad.data_in = None
        self.alpha = alpha

    def forward(self, data_in):
        self.grad.data_in = data_in
        data_neg = self.alpha * (torch.exp(self.grad.data_in) - 1)
        data_pos = self.grad.data_in
        return torch.max(data_pos, torch.tensor(0.)) + torch.min(data_neg, torch.tensor(0.))

    def backward(self, grad_dl_dout):
        return grad_dl_dout * ((self.grad.data_in > 0) + self.alpha * (self.grad.data_in < 0))

    def reset_params(self):
        self.grad.data_in = None


class SeLU(Module):
    """alpha and lamda optimal value from Pytorch"""

    def __init__(self, alpha=1.6733, lambda_=1.0507):
        self.grad = StructFoo()
        self.grad.data_in = None
        self.alpha = alpha
        self.lambda_ = lambda_

    def forward(self, data_in):
        self.grad.data_in = data_in
        data_neg = self.lambda_ * self.alpha * (torch.exp(self.grad.data_in) - 1)
        data_pos = self.lambda_ * self.grad.data_in
        return torch.max(data_pos, torch.tensor(0.)) + torch.min(data_neg, torch.tensor(0.))

    def backward(self, grad_dl_dout):
        return grad_dl_dout * self.lambda_ * ((self.grad.data_in > 0) + self.alpha * (self.grad.data_in < 0))

    def reset_params(self):
        self.grad.data_in = None


class Tanh(Module):
    def __init__(self):
        self.grad = StructFoo()
        self.grad.data_in = None

    def forward(self, data_in):
        self.grad.data_in = data_in
        return torch.tanh(data_in)

    def backward(self, grad_dl_dout):
        return grad_dl_dout * (1 - torch.tanh(self.grad.data_in).pow(2))

    def reset_params(self):
        self.grad.data_in = None


class Sigmoid(Module):
    def __init__(self):
        self.grad = StructFoo()
        self.grad.data_in = None

    def forward(self, data_in):
        self.grad.data_in = data_in
        return 1 / (1 + torch.exp(-self.grad.data_in))

    def backward(self, grad_dl_dout):
        return grad_dl_dout * (1 / (1 + torch.exp(-self.grad.data_in))) * (1 - 1 / (1 + torch.exp(-self.grad.data_in)))

    def reset_params(self):
        self.grad.data_in = None


"""Losses"""


class LossMSE(Module):
    def __init__(self):
        pass

    def forward(self, output, target):
        return torch.pow(target - output, 2).mean()

    def backward(self, output, target):
        return 2 * (output - target).div(output.size(0))


class LossBCE(Module):
    def __init__(self):
        pass

    def forward(self, output, target):
        sigmoid = torch.div(1, 1 + torch.exp(-1 * output))
        return -(target * torch.log(sigmoid) + (1 - target) * torch.log(1 - sigmoid))

    def backward(self, output, target):
        sigmoid = torch.div(1, 1 + torch.exp(-1 * output))
        return torch.div((sigmoid - target), (sigmoid * (1 - sigmoid)))


