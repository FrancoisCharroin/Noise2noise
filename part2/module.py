from torch.nn.functional import fold, unfold
import torch
import math

class Conv2d():
    def __init__(self, channels_in, channels_out, kernel_size, stride=1, bias = 1):
        self.in_ = channels_in
        self.out_ = channels_out
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else: self.kernel_size = kernel_size
        self.k_0 = self.kernel_size[0]
        self.k_1 = self.kernel_size[1]
        self.stride = stride
        self.bias_bool = bias
        
        # Uniform distribution for the bias and weights as in pytorch conv2D 
        
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
        # from the informations of the project
        self.x = input
        
        self.co = unfold(self.x, kernel_size=self.kernel_size, stride=self.stride)
        res = self.weight.view(self.out_, -1) @ self.co
        if self.bias_bool:
            res += self.bias.view(1, -1, 1)
        
        par_res = self.x.shape[2] - self.k_0
        par_res = par_res / self.stride
        par_res = math.floor(par_res)
        
        return  res.view(self.x.shape[0], self.out_, par_res + 1, -1)

    def backward(self, gradwrtoutput):
        #inspired by https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/layers.py
        r_grad = gradwrtoutput.permute(1, 2, 3, 0)
        r_grad = r_grad.reshape(self.out_, -1)
        self.g_b.data = gradwrtoutput.sum(dim=[0, 2, 3])
        
        r_x =self.co.transpose(2, 1).transpose(1, 0).reshape(r_grad.shape[1], -1)
        self.g_w.data = (r_grad @ r_x)
        self.g_w.data = self.g_w.data.reshape(self.weight.shape)
        #sum with respect to axis 1
        
        w_reshape = self.weight.reshape(self.out_, -1).T
        dx_res = w_reshape @ r_grad
        dx_res = dx_res.reshape(self.co.permute(1, 2, 0).shape)
        dx_res = dx_res.transpose(2, 1).transpose(1, 0)
        
        dim_inp = (self.x.shape[2], self.x.shape[3])
        return fold(dx_res, dim_inp, kernel_size=self.kernel_size, stride=self.stride)
    def zero_grad(self):
        self.g_w.zero_()
        self.g_b.zero_()

class Sequential():

    def __init__(self, *layers):
        #Include also acyivation functions
        self.all_layers = layers
        self.parameters = self.param()
    def zero_grad(self):
        #Put the gradients of the weight and bias to zero
        for layer in self.all_layers:
            layer.zero_grad()
    def param(self):
        self.tot_par = []
        for single_layer in self.all_layers:
            for param in single_layer.param():
                self.tot_par.insert(len(self.tot_par),param)
        return self.tot_par
    
    def set_param(self, params):

        self.parameters = params

    def forward(self, input):
        for single_layer in self.all_layers:
            input = single_layer.forward(input)
        return input

    def backward(self , grad):
        for single_layer in reversed(self.all_layers):
            grad = single_layer.backward(grad)
        return grad

class SGD():
    def __init__(self, params ,lr):
        self.params = params
        self.lr = lr              
    def param (self):
        return self.params
    def step(self):
        for [param, g_params] in self.params:
            param.data = param.data - self.lr * g_params
    def forward(self, *input):
        return None
    def backward(self, *gradwrtoutput):
        return None

class ReLU() :
    def __init__(self):
        self.inp = 0
    def param(self):
        return  []
    def forward (self, input):
        return torch.max(input, torch.tensor(0.))

    def backward (self, x_g):
        derivative = self.inp > 0
        self.grad = x_g * derivative
        return self.grad

    def zero_grad(self):
        pass

class Sigmoid():
    def __init__(self):
        pass
    def param(self):
        return  []
    def forward(self, input):
        self.inp = input
        bottom = 1 + torch.exp(-self.inp)
        sol = 1/bottom
        return sol
    
    def backward(self, grad):
        resu = grad
        bottom = 1 + torch.exp(-self.inp)*(1-1/(1+torch.exp(-self.inp))) 
        multi = 1/bottom 
        resu = resu * multi  
        return resu
    def zero_grad(self):
        pass
class MSE():
    def __init__(self):
        pass
    def param(self):
        pass
    def forward(self, predic, target):
        self.prediction = predic
        self.target = target
        self.err = predic - target
        square_error = (self.err)**2
        self.loss = torch.mean(square_error)
        return self.loss

    def backward(self):
        top = 2 * (self.err)
        tot = self.prediction.shape[0]*self.prediction.shape[1]*self.prediction.shape[2]*self.prediction.shape[3]
        bottom = tot
        self.grad = top / bottom
        return self.grad

class NearestUpsampling():
    
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
        #upsampling pytorch
        #print(input.size())
        self.x = input.repeat_interleave( self.size[1], dim=3).repeat_interleave( self.size[0], dim=2)
        #print(self.x.size())
        co = unfold(self.x, kernel_size=self.kernel_size, stride=self.stride)
        #print(co.size())
        res = self.weight.view(self.out_, -1) @ co
        #print(res.size())
        res += self.bias.view(1, -1, 1)
        par_res = self.x.shape[2] - self.k_0
        par_res = par_res / self.stride
        par_res = math.floor(par_res)
        
        return  res.view(self.x.shape[0], self.out_, par_res + 1, -1)
        

    def backward(self, gradwrtoutput):
        
        r_grad = gradwrtoutput.permute(1, 2, 3, 0)
        r_grad = r_grad.reshape(self.out_, -1)
        self.g_b.data = gradwrtoutput.sum(dim=[0, 2, 3])
        
        conv_res = unfold(self.x, kernel_size=self.kernel_size, stride=self.stride)
        r_x =conv_res.transpose(2, 1).transpose(1, 0).reshape(r_grad.shape[1], -1)
        self.g_w.data = (r_grad @ r_x)
        
        self.g_w.data = self.g_w.data.reshape(self.weight.shape)
        
        w_reshape = self.weight.reshape(self.out_, -1)
        dx_res = w_reshape.t() @ r_grad
        dx_res = dx_res.reshape(conv_res.permute(1, 2, 0).shape)
        dx_res = dx_res.transpose(2, 1).transpose(1, 0)
        
        
        dim_inp = (self.x.shape[2], self.x.shape[3])
        resul = fold(dx_res, dim_inp, kernel_size=self.kernel_size, stride=self.stride)
        
        n = self.size[0]
        
        
        #adding the grad together
        dim_1 = int(resul.shape[2] * resul.shape[3]/4)
        step_1 = resul.reshape(20,48,dim_1,n).sum(3)
        
        
        step_2 = step_1.reshape(20,48,int(resul.shape[2]),int(resul.shape[2]/4))
        
        dim_2 = int(resul.shape[2]*resul.shape[2]/4/4)
        dim_3 = int(resul.shape[2]/4)
        step_3 = step_2.transpose(2,3).reshape(20,48,dim_2,4).sum(3).reshape(20,48,dim_3,dim_3).transpose(2,3)
        
                        
        
        return step_3
    def param(self):
        return [(self.weight, self.g_w), (self.bias, self.g_b)]
    def zero_grad(self):
        self.g_w.zero_()
        self.g_b.zero_()
        
