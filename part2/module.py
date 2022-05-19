from torch import empty
from torch.nn.functional import fold, unfold
import torch
import time
import math


class Module(object):

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        parameters = self.param()
        if parameters:
            for p in parameters:
                p[1].zero_()


class Conv2d(Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1):
        self.in_ = channels_in
        self.out_ = channels_out
    
        self.kernel_size = kernel_size
        self.k_0 = self.kernel_size[0]
        self.k_1 = self.kernel_size[1]
        self.stride = stride
        
        # Uniform distribution
        
        k = 1/(self.in_ * self.k_0 * self.k_1)
        self.bias = empty(self.out_).uniform_(-(1/(self.in_ * self.k_0 *  self.k_1))**0.5,
                                              (1/(self.in_ * self.k_0 *  self.k_1))**0.5)

        

        self.g_b = empty(self.out_).zero_()


        self.weight = empty(self.out_, self.in_, self.k_0, self.k_1)
        self.weight = self.weight.uniform_(-k**.5, k**.5)

        self.g_w = empty(self.out_, self.in_, self.k_0, self.k_1)
        self.g_w = self.g_w.zero_()


    def forward(self, input):
        self.x = input
        
        co = unfold(self.x, kernel_size=self.kernel_size, stride=self.stride)
        
        
        res = self.weight.view(self.out_, -1) @ co
        res += self.bias.view(1, -1, 1)
        
        par_res = self.x.shape[2] - self.k_0
        par_res = par_res / self.stride
        par_res = math.floor(par_res)
        
        return  res.view(self.x.shape[0], self.out_, par_res + 1, -1)

    def backward(self, gradwrtoutput):

        r_grad = gradwrtoutput.permute(1, 2, 3, 0).reshape(self.out_, -1)
        conv_res = unfold(self.x, kernel_size=self.kernel_size, stride=self.stride)
        r_x =conv_res.permute(2, 0, 1).reshape(r_grad.shape[1], -1)
        self.g_w.data = (r_grad @ r_x)
        #print(self.g_w.shape)
        self.g_w.data = self.g_w.data.reshape(self.weight.shape)
        #print(self.g_w.shape)
        self.g_b.data = gradwrtoutput.sum(axis = (0, 2, 3))

        w_reshape = self.weight.reshape(self.out_, -1)
        dx_res = w_reshape.t() @ r_grad
        dx_res = dx_res.reshape(conv_res.permute(1, 2, 0).shape)
        dx_res = dx_res.permute(2, 0, 1)
        
        
        dim_inp = (self.x.shape[2], self.x.shape[3])
        return fold(dx_res, dim_inp, kernel_size=self.kernel_size, stride=self.stride)

    def param(self):
        return [(self.weight, self.g_w), (self.bias, self.g_b)]


class TransposeConv2d(Module):

    def __init__(self, channels_in, channels_out, kernel_size, stride=1):
        self.in_ = channels_in
        self.out_ = channels_out
        
        self.kernel_size = kernel_size
        self.k_0 = self.kernel_size[0]
        self.k_1 = self.kernel_size[1]
        
        
        self.stride = stride

        k = 1/(self.out_ * self.k_0 * self.k_1)


        self.bias = empty(self.out_).uniform_(-k**.5, k**.5)
        self.g_b = empty(self.out_).zero_()

       
        self.weight = empty(self.in_, self.out_, 
                            self.k_0, self.k_1).uniform_(-k**.5, k**.5)
        
        self.g_w = empty(self.in_, self.out_, 
                        self.k_0, self.k_1).zero_()


    def forward(self, input):

        self.x = input
        
        r_x = self.x.permute(1, 2, 3, 0)
        r_x = r_x.reshape(self.in_, -1)
        
        r_w = self.weight
        r_w = r_w.reshape(self.in_, -1)
        
        uf_out = r_w.t() @ r_x
        uf_out = uf_out.reshape(uf_out.shape[0], -1, self.x.shape[0])
        uf_out = uf_out.permute(2, 0, 1)

        width = self.x.shape[3] - 1
        width =  width*self.stride
        width += self.k_1
        
        height = self.x.shape[2] - 1
        height = height*self.stride
        height += self.k_0 
        

        fold_res = fold(uf_out, (height, width), kernel_size = self.kernel_size,
                    stride = self.stride)
        fold_res += self.bias.view(1, -1, 1, 1)
        return fold_res

    def backward(self, grad):

        r_x = self.x.permute(1, 2, 3, 0)
        r_x = r_x.reshape(self.in_, -1)
        
        d_unf = unfold(grad, kernel_size = self.kernel_size,
                       stride=self.stride)
        r_g = d_unf.permute(2, 0, 1).reshape(r_x.shape[1], -1)

        self.g_w.data = r_x @ r_g
        self.g_w.data = self.g_w.data.reshape(self.weight.shape)
        
        self.g_b.data = grad.sum(axis = (0, 2, 3))
        
        x_g =self.weight.view(self.in_, -1) @ d_unf
        
        return x_g.view(x_g.shape[0], self.in_, self.x.shape[2],
                          self.x.shape[3])

    def param(self):
        return [(self.weight, self.g_w), (self.bias, self.g_b)]

class Sequential(Module):

    def __init__(self, *layers):
        self.all_layers = list(layers)

    def forward(self, input):
        for single_layer in self.all_layers:
            input = single_layer.forward(input)
        return input

    def backward(self , grad):
        
        for single_layer in reversed(self.all_layers):
            grad = single_layer.backward(grad)
        return grad

    def param(self):
        
        tot_par = []
        
        for single_layer in self.all_layers:
            
            for param in single_layer.param():
                
                tot_par.append(param)
                
        return  tot_par


class SGD(Module):
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for [param, g_params] in self.params:
            param.data -= self.lr * g_params

    def param (self):
        return self.params

class ReLU(Module) :

    def __init__(self):
        self.inp = 0

    def forward (self, input):
        self.inp = input
        #mask = self.inp<0
        self.inp[ self.inp < 0 ] = 0
        return self.inp

    def backward (self, x_g):
        self.grad = x_g * (self.inp > 0)
        return self.grad



class Sigmoid(object):

    def __init__(self):
        self.inp = 0

    def forward(self, input):
        self.inp = input
        bottom = 1 + torch.exp(-self.inp)
        sol = 1/bottom
        return sol

    def backward(self, grad):
        
        resu = grad
        bottom = 1 + torch.exp(-self.inp) * (1 - 1 / (1 + torch.exp(-self.inp))) 
        multi = 1/bottom 
        resu = resu * multi
        
        return resu

    def param(self):
        return  []

class MSE(Module):

    def __init__(self):
        super().__init__()

    def forward(self, predic, target):
        self.prediction = predic
        self.target = target
        square_error = (predic - target)**2
        self.loss = torch.mean(square_error)
        return self.loss

    def backward(self):
        top = 2 * (self.prediction - self.target)
        tot = self.prediction.shape[0]*self.prediction.shape[1]*self.prediction.shape[2]*self.prediction.shape[3]
        bottom = tot
        self.grad = top / bottom
        return self.grad

    def param(self):
        return []
