from torch.nn.functional import fold, unfold
import torch
import math

class Conv2d():
    def __init__(self, channels_in, channels_out, kernel_size, stride=1, bias = 1):
        self.in_ = channels_in
        self.out_ = channels_out
        self.kernel_size = kernel_size
        self.k_0 = self.kernel_size[0]
        self.k_1 = self.kernel_size[1]
        self.stride = stride
        self.bias_bool = bias
        
        # Uniform distribution for the bias as in pytorch conv2D 
        
        self.bias = torch.Tensor(self.out_).uniform_(-(1/(self.in_ * self.k_0 * self.k_1))**0.5,
                                              (1/(self.in_ * self.k_0 *  self.k_1))**0.5)

        

        self.g_b = torch.Tensor(self.out_)
        self.weight = torch.Tensor(self.out_, self.in_, self.k_0, self.k_1)
        self.weight = self.weight.uniform_(-(1/(self.in_ * self.k_0 * self.k_1))**0.5,
                                              (1/(self.in_ * self.k_0 *  self.k_1))**0.5)

        self.g_w = torch.Tensor(self.out_, self.in_, self.k_0, self.k_1)
        
    def param(self):
        return [(self.weight, self.g_w), (self.bias, self.g_b)]

    def forward(self, input):
        self.x = input
        
        co = unfold(self.x, kernel_size=self.kernel_size, stride=self.stride)
        res = self.weight.view(self.out_, -1) @ co
        if self.bias_bool:
            res += self.bias.view(1, -1, 1)
        
        par_res = self.x.shape[2] - self.k_0
        par_res = par_res / self.stride
        par_res = math.floor(par_res)
        
        return  res.view(self.x.shape[0], self.out_, par_res + 1, -1)

    def backward(self, gradwrtoutput):
       # inspired by https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py
        
       
        
        r_grad = gradwrtoutput.permute(1, 2, 3, 0)
        
        r_grad = r_grad.reshape(self.out_, -1)
        
        conv_res = unfold(self.x, kernel_size=self.kernel_size, stride=self.stride)
        r_x =conv_res.permute(2, 0, 1).reshape(r_grad.shape[1], -1)
        self.g_w.data = (r_grad @ r_x)
        
        self.g_w.data = self.g_w.data.reshape(self.weight.shape)
        #sum with respect to axis 1
        self.g_b.data = gradwrtoutput.sum(axis = (0, 2, 3))
        w_reshape = self.weight.reshape(self.out_, -1)
        dx_res = w_reshape.t() @ r_grad
        dx_res = dx_res.reshape(conv_res.permute(1, 2, 0).shape)
        dx_res = dx_res.permute(2, 0, 1)
        
        
        dim_inp = (self.x.shape[2], self.x.shape[3])
        return fold(dx_res, dim_inp, kernel_size=self.kernel_size, stride=self.stride)

   

class Sequential():

    def __init__(self, *layers):
        self.all_layers = list(layers)
        self.tot_par = []
        
    def param(self):
        
        self.tot_par = []
        
        for single_layer in self.all_layers:
            if (single_layer != Sigmoid) and (single_layer != ReLU):
                for param in single_layer.param():
                    
                    self.tot_par.append(param)
                
        return self.tot_par

    def forward(self, input):
        for single_layer in self.all_layers:
            input = single_layer.forward(input)
        return input

    def backward(self , grad):
        
        for single_layer in reversed(self.all_layers):
            grad = single_layer.backward(grad)
        return grad




class SGD():
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    def param (self):
        return self.params

    def step(self):
        for [param, g_params] in self.params:
            param.data -= self.lr * g_params
    def zero_grad(self):
        parameters = self.param()
        if parameters:
            for p in parameters:
                p[1].zero_()

    def forward(self, *input):
        return []

    def backward(self, *gradwrtoutput):
        return []
class ReLU() :

    def __init__(self):
        self.inp = 0
    def param(self):
        return  []
    def forward (self, input):
        self.inp = input
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

class MSE():

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
    
    
class UpSampling2D():
    
    def __init__(self,channels_in, channels_out, kernel_size, size=(4,4), stride=1):
        
        self.size = size
        self.in_ = channels_in
        self.out_ = channels_out
        
        self.kernel_size = kernel_size
        self.k_0 = self.kernel_size[0]
        self.k_1 = self.kernel_size[1]
        
        
        self.stride = stride

        self.bias = torch.Tensor(self.out_).uniform_(-(1/(self.in_ * self.k_0 * self.k_1))**0.5,
                                              (1/(self.in_ * self.k_0 *  self.k_1))**0.5)
        
        self.g_b = torch.Tensor(self.out_).zero_()

       
        self.weight = torch.Tensor(self.out_, self.in_, self.k_0, self.k_1)
        self.weight = self.weight.uniform_(-(1/(self.in_ * self.k_0 * self.k_1))**0.5,
                                              (1/(self.in_ * self.k_0 *  self.k_1))**0.5)
        
        self.g_w = torch.Tensor(self.in_, self.out_, 
                        self.k_0, self.k_1).zero_()

    def forward(self, input):
        # Repeat each axis as specified by size
        print(input.size())
        self.x = input.repeat_interleave( 4, dim=3).repeat_interleave( 4, dim=2)
        print(self.x.size())
        co = unfold(self.x, kernel_size=self.kernel_size, stride=self.stride)
        print(co.size())
        res = self.weight.view(self.out_, -1) @ co
        print(res.size())
        res += self.bias.view(1, -1, 1)
        self.H_out = math.floor((self.x.shape[2] - 1*(self.kernel_size[0]-1) -1 )/self.stride) + 1 
        self.W_out = math.floor((self.x.shape[3] - 1*(self.kernel_size[1]-1) -1 )/self.stride) + 1
        print(res.view(self.x.shape[0], self.out_, self.H_out , -1).size())
        return  res.view(self.x.shape[0], self.out_, self.H_out, -1)
        

    def backward(self, gradwrtoutput):
        # Down sample input to previous shape
        
        r_grad = gradwrtoutput.permute(1, 2, 3, 0)
        
        r_grad = r_grad.reshape(self.out_, -1)
        
        conv_res = unfold(self.x, kernel_size=self.kernel_size, stride=self.stride)
        r_x =conv_res.permute(2, 0, 1).reshape(r_grad.shape[1], -1)
        self.g_w.data = (r_grad @ r_x)
        
        self.g_w.data = self.g_w.data.reshape(self.weight.shape)
        #sum with respect to axis 1
        self.g_b.data = gradwrtoutput.sum(axis = (0, 2, 3))
        w_reshape = self.weight.reshape(self.out_, -1)
        dx_res = w_reshape.t() @ r_grad
        dx_res = dx_res.reshape(conv_res.permute(1, 2, 0).shape)
        dx_res = dx_res.permute(2, 0, 1)
        
        
        dim_inp = (self.x.shape[2], self.x.shape[3])
        resul = fold(dx_res, dim_inp, kernel_size=self.kernel_size, stride=self.stride)
        resul = resul[:, :, ::self.size[0], ::self.size[1]]
        return resul
    def param(self):
        return [(self.weight, self.g_w), (self.bias, self.g_b)]
    
