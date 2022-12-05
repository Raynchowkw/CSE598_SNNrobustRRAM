from fileinput import filename
from urllib.request import FileHandler
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

import numpy as np
import math
import random
import torch.nn.functional as F

import logging
# TODO: manage these variables into a file
from __main__ import shift, shifter, sram_init0, tern, load_sram_w

######## ========== inplace model's weights adding var ==========
def addDeviceVariation_SA(wts, stddevVar): #wts -> model.parameters(), model.named_parameters(), model.named_buffers()
    """
    This function adds variation
    """
    #name=='stage_1.0.conv_a.weight'or'stage_1.0.conv_b.weight'or'stage_1.1.conv_a.weight' or'stage_1.1.conv_b.weight' or'stage_1.2.conv_a.weight'or'stage_1.2.conv_b.weight'or 'stage_2.0.conv_a.weight' or'stage_2.0.conv_b.weight'or'stage_2.1.conv_a.weight' or'stage_2.1.conv_b.weight' or'stage_2.2.conv_a.weight' or'stage_2.2.conv_b.weight' or'stage_3.0.conv_a.weight' or'stage_3.0.conv_b.weight'or'stage_3.1.conv_a.weight'or'stage_3.1.conv_b.weight' or'stage_3.2.conv_a.weight' or'stage_3.2.conv_b.weight':
    if stddevVar != 0.0:
        for name,param in wts:
			#print(name,"adding variartion===================================")
			# if ('conv' in name) and ('weight' in name) :
            if ('conv' in name):
                # logger.info(str(name))
                w_l =param.data
                w_npy = w_l.cpu().numpy()
                max = np.amax(w_npy)
				# min = max/350
                min = np.amin(w_npy) #We are assuming the lowest resistance state will be the minimum value; non-linear distribution of levels
                shape_w = np.shape(w_l)
                var_exp = np.exp(np.random.normal(0, stddevVar, shape_w))
                w_npy =   w_npy * var_exp
                signMat = np.ones(shape_w, dtype=np.float32)
                signMat[np.where(w_npy < 0.0)] = -1.0
                w_npy = np.absolute(w_npy) #TODO: move it to after clamping               
                w_npy[np.where(w_npy > max)] = max
                w_npy[np.where(w_npy < min)] = min #! if min<0, no w_npy<min because absolute()
                #import pdb;pdb.set_trace()
                # w_npy = np.reshape(w_npy, shape_w)
                w_npy = w_npy * signMat
                w_npy = w_npy.astype(np.float32)
                param.data=torch.from_numpy(w_npy).cuda()
	#print("Finished adding device variations and no stuck-at-faults")

######## ========== not inplace model's weights adding var ==========
def addDeviceVariation_w(wts, stddevVar):
    if stddevVar != 0.0:
        w_l = wts
        w_npy = w_l.cpu().numpy()
        max = np.amax(w_npy)
        # min = max/350
        min = np.amin(w_npy) #We are assuming the lowest resistance state will be the minimum value; non-linear distribution of levels
        shape_w = np.shape(w_l)
        var_exp = np.exp(np.random.normal(0, stddevVar, shape_w))
        w_npy =   w_npy * var_exp
        signMat = np.ones(shape_w, dtype=np.float32)
        signMat[np.where(w_npy < 0.0)] = -1.0
        w_npy = np.absolute(w_npy)                
        w_npy[np.where(w_npy > max)] = max
        w_npy[np.where(w_npy < min)] = min
        #import pdb;pdb.set_trace()
        w_npy = w_npy * signMat
        w_npy = w_npy.astype(np.float32)
        weights = torch.from_numpy(w_npy).cuda()
        return weights
    else:
        return wts

### ==== inplace model's weight adding var bit-by-bit ====
## == credit to Zhenyu
def addDeviceVar_bbb(wts, stdvar, bits): #FIXME: need to chagne for XNOR according to level
    if stdvar != 0.0:
        for name,param in wts:
            if ('conv' in name) and ('weight' in name): 
                shape_w = param.size()
                # unique=wts.unique()
                weight_q = param.data
                if bits==1:

                    var_exp_1 =  torch.exp(torch.empty(shape_w).normal_(0, 0.5))
                    weight_q = weight_q * var_exp_1.cuda()

                elif bits==2:
                    #lsb
                    weight_2bit_1_1=torch.clone(weight_q)
                    #print_log(unique,log)
                    weight_2bit_1_1[weight_2bit_1_1.abs()>0.4]=0
                    
                    weight_2bit_1=torch.clone(weight_2bit_1_1).cuda()
                    #if self.iter==0:
                    #    self.var_exp_2_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.2760)).cuda()
                    var_exp_2_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.276)).cuda()
                    weight_2bit_vat1=torch.clone(weight_2bit_1*var_exp_2_1).cuda()
                    #print_log(torch.linalg.norm(weight_2bit_vat1),log)
                    
                    #MSB
                    weight_2bit_1_2=torch.clone(weight_q)
                    weight_2bit_1_2[weight_2bit_1_2.abs()<0.9]=0
                    weight_2bit_2=torch.clone(weight_2bit_1_2)
                    #if self.iter==0:
                    #   self.var_exp_2_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.1035)).cuda()
                    var_exp_2_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.1035)).cuda()
                    weight_2bit_vat2=torch.clone(weight_2bit_2*var_exp_2_2).cuda()
                    #import pdb;pdb.set_trace()
                    weight_q=torch.add(weight_2bit_vat1,weight_2bit_vat2).cuda()

                    #import pdb;pdb.set_trace()
                    #	
                elif bits == 3: #w_bit = 3bit

                    weight_3bit_1_1=torch.clone(weight_q)
                    weight_3bit_1_1[weight_3bit_1_1.abs()>0.143]=0
                    weight_3bit_1=torch.clone(weight_3bit_1_1).cuda()
                    #if self.iter==0:
                    #	self.var_exp_3_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.3549)).cuda()
                    var_exp_3_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.3549)).cuda()
                    #print(var_exp_3_1.norm())
                    weight_3bit_vat1=torch.clone(weight_3bit_1*var_exp_3_1).cuda()

                    
                    #-----------------------2nd bit--------------


                    weight_3bit_1_2=torch.clone(weight_q)
                    weight_3bit_1_2[weight_3bit_1_2.abs()<0.42]=0
                    weight_3bit_1_2[weight_3bit_1_2.abs()>0.43]=0
                    
                    weight_3bit_2=torch.clone(weight_3bit_1_2).cuda()
                    #if self.iter==0:

                    #	self.var_exp_3_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.2259)).cuda()
                    var_exp_3_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.2259)).cuda()
                    weight_3bit_vat2=torch.clone(weight_3bit_2*var_exp_3_2).cuda()

                    #-----------------------3rd bit------------------------
                    weight_3bit_1_3=torch.clone(weight_q)
                    weight_3bit_1_3[weight_3bit_1_3.abs()<0.7]=0
                    #import pdb;pdb.set_trace()
                    weight_3bit_3=torch.clone(weight_3bit_1_3).cuda()

                    #if self.iter==0:
                    #	self.var_exp_3_3 = torch.exp(torch.empty(shape_w).normal_(0, 0.1898)).cuda()
                    var_exp_3_3 = torch.exp(torch.empty(shape_w).normal_(0, 0.1898)).cuda()
                    weight_3bit_vat3=torch.clone(weight_3bit_3*var_exp_3_3).cuda()
                    #print(weight_3bit_vat3)

                    weight_3bit=torch.add(weight_3bit_vat1,weight_3bit_vat2)
                    weight_3bit=torch.add(weight_3bit,weight_3bit_vat3)

                    weight_q=weight_3bit.cuda()				

                elif bits==4:

                    #-----------------------1st bit--------------
                    weight_4bit_1_1=torch.clone(weight_q)
                    weight_4bit_1_1[weight_4bit_1_1.abs()>0.07]=0
                    weight_4bit_1=torch.clone(weight_4bit_1_1).cuda()
                    #if self.iter==0:
                    #	self.var_exp_3_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.3549)).cuda()
                    var_exp_4_1 = torch.exp(torch.empty(shape_w).normal_(0, 0.2015)).cuda()
                    #print(var_exp_3_1.norm())
                    weight_4bit_vat1=torch.clone(weight_4bit_1*var_exp_4_1).cuda()

                    #-----------------------2nd bit--------------

                    weight_4bit_1_2=torch.clone(weight_q)
                    weight_4bit_1_2[weight_4bit_1_2.abs()<0.19]=0
                    weight_4bit_1_2[weight_4bit_1_2.abs()>0.21]=0
                    
                    weight_4bit_2=torch.clone(weight_4bit_1_2).cuda()

                    var_exp_4_2 = torch.exp(torch.empty(shape_w).normal_(0, 0.0643)).cuda()
                    weight_4bit_vat2=torch.clone(weight_4bit_2*var_exp_4_2).cuda()

                    #-----------------------3rd bit------------------------
                    weight_4bit_1_3=torch.clone(weight_q)
                    weight_4bit_1_3[weight_4bit_1_3.abs()<0.32]=0
                    weight_4bit_1_3[weight_4bit_1_3.abs()<0.47]=0
                    #import pdb;pdb.set_trace()
                    weight_4bit_3=torch.clone(weight_4bit_1_3).cuda()


                    var_exp_4_3 = torch.exp(torch.empty(shape_w).normal_(0, 0.0513)).cuda()
                    weight_4bit_vat3=torch.clone(weight_4bit_3*var_exp_4_3).cuda()

                    #-----------------------4th bit------------------------
                    weight_4bit_1_4=torch.clone(weight_q)
                    weight_4bit_1_4[weight_4bit_1_4.abs()<0.59]=0
                    #import pdb;pdb.set_trace()
                    weight_4bit_4=torch.clone(weight_4bit_1_4).cuda()


                    var_exp_4_4 = torch.exp(torch.empty(shape_w).normal_(0, 0.0442)).cuda()
                    weight_4bit_vat4=torch.clone(weight_4bit_4*var_exp_4_4).cuda()

                    #-----------------------4th bit------------------------
                    weight_4bit=torch.add(weight_4bit_vat1,weight_4bit_vat2)
                    weight_4bit=torch.add(weight_4bit,weight_4bit_vat3)
                    weight_4bit=torch.add(weight_4bit,weight_4bit_vat4)

                    weight_q=weight_4bit.cuda() 
                param.data=weight_q
## ====== add var finish ======

####### ========= quant method ==========
########### ------- Ternary quant --------- 
### credit to : Zhezhi He, Li Yang in Fan's group, https://github.com/elliothe/Ternarized_Neural_Network/blob/master/models/tern_resnet_cifar.py
class _quanFunc(torch.autograd.Function):

    def __init__(self, tfactor):
        super(_quanFunc,self).__init__()
        # print("init_tFactor=", tfactor)
        self.tFactor = tfactor

    @staticmethod
    def forward(self, input):
        # quantization configuration
        self.tFactor = 0.05
        # print("self.TF:",self.tFactor)
        self.save_for_backward(input)
        max_w = input.abs().max()
        self.th = self.tFactor*max_w #threshold
        output = input.clone().zero_()
        self.W = input[input.ge(self.th)+input.le(-self.th)].abs().mean()
        output[input.ge(self.th)] = self.W
        output[input.lt(-self.th)] = -self.W
        # print(type(output))
        return output

    @staticmethod
    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

## quant layer
class quanConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False, var=0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,bias)
        self.var=var
        tfactor_list = [0.05]
        # self.ternquant = _quanFunc(tfactor=tfactor_list[0]).apply
    # def forward(self, input):
    #     tfactor_list = [0.05, 0.1, 0.15, 0.2]
    #     weight = _quanFunc(tfactor=tfactor_list[0])(self.weight)
    #     output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
    #     for tfactor in tfactor_list[1:]:
    #         weight = _quanFunc(tfactor=tfactor)(self.weight)
    #         output += F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    #     return output 
    def forward(self, input):
        tfactor_list = [0.05]
        weight = _quanFunc(tfactor=tfactor_list[0]).apply(self.weight)
        weight = addDeviceVariation_w(weight, self.var)
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return output 

## sram layer
class quanConv2d_sram(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1,bias=False, var=0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.var=var
        self.register_buffer("WR",torch.zeros(self.weight.size()))
    # def forward(self, input):
    #     tfactor_list = [0.05, 0.1, 0.15, 0.2]
    #     weight = _quanFunc(tfactor=tfactor_list[0])(self.weight)
    #     output = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
    #     for tfactor in tfactor_list[1:]:
    #         weight = _quanFunc(tfactor=tfactor)(self.weight)
    #         output += F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    #     return output 
    def forward(self, input):
        tfactor_list = [0.05] #! manual setting
        weight = _quanFunc(tfactor=tfactor_list[0])(self.weight)
        weight_rram = _quanFunc(tfactor=tfactor_list[0])(self.WR)
        weight_rram = addDeviceVariation_w(weight_rram, self.var)
        output_sram = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output_rram = F.conv2d(input, weight_rram, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output = (output_rram + output_sram) / 2
        return output 

########### ------- DoReFa quant ---------
## TODO: to be added
########### ------- DoReFa quant ---------

####### ========== quant method end ==========

## activation BP method: surrogate, credit to Yale 
class Surrogate_BP_Function(torch.autograd.Function):


    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp)) # ? only abs on one side
    # ? why not torch.le(torch.abs(rand_inp * rescale_fac), torch.abs(inp))

## --- sram conv layer by register_buffer() --- ##
class Conv2d_sram(nn.Conv2d):
    # logger = logging.getLogger("__main__")
    # logger.info("creating sram-conv")
    # logger.info(str(__main__.shifter))
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                                padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_sram, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # self.w_bit_sram=w_bit_sram
        self.register_buffer("WR",torch.zeros(self.weight.size())) # "WR" is the model.register_buffer's name
    
    def forward(self, input, order=None): #TODO: it's already sram, quantized
        x=input

        #! weight should be quant before and unquant after. This is handled outside this class in the training.
        # if self.sram:
        outputRegular_RRAM = F.conv2d(x,self.WR, self.bias, self.stride, self.padding, self.dilation, self.groups)
        outputRegular_SRAM = F.conv2d(x,self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # else:
        #     outputRegular_RRAM = F.conv2d(x,self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        #     outputRegular_SRAM = 0
        # logger.info(torch.count_nonzero(outputRegular_SRAM))
        if shift:
            output = outputRegular_RRAM + outputRegular_SRAM * shifter #shift
        else:
            output = outputRegular_RRAM + outputRegular_SRAM # no shift
            # print(outputRegular_SRAM[3][4][2])
        #     # print("no sram")
        #     output = outputRegular_RRAM # no sram, remain same
        
        if not sram_init0:
            output = output/2 # because threshold unchanged
        return output

## SNN BNTT model by Yale: https://github.com/Intelligent-Computing-Lab-Yale/BNTT-Batch-Normalization-Through-Time
class SNN_VGG5_BNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=0.95, img_size=28,  num_cls=10, bntt_flag = False, sram = False):
        super(SNN_VGG5_BNTT, self).__init__()

        self.sram = sram
        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        self.bntt_flag = bntt_flag

        # print (">>>>>>>>>>>>>>>>>>> VGG 5 >>>>>>>>>>>>>>>>>>>>>>")
        # print ("***** time step per batchnorm".format(self.batch_num))
        # print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        ## TODO: use block or configuration table to form sram layers or normal layer, not use branch for each layer
        if self.sram:
            self.conv1 = Conv2d_sram(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag) #! it should be 3 for input channel
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag) 
        # one bn layer for each time step
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2) #! avgPool, not MP in ANN vgg
        if self.sram:
            self.conv2 = Conv2d_sram(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        else:
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        if self.sram:
            self.conv3 = Conv2d_sram(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        else:
            self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)


        self.fc1 = nn.Linear((self.img_size//4)*(self.img_size//4)*128, 1024, bias=bias_flag)
        self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(1024, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt_fc]
        self.pool_list = [self.pool1, False, self.pool2]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None

        # Initialize the firing thresholds of all the layers
        # no ini for pooling layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                print(m)
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)




    def forward(self, inp, dev_param, wts):

        batch_size = inp.size(0) #! make sure input.shape[0] is batch size
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3]

        mem_fc1 = torch.zeros(batch_size, 1024).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()



        for t in range(self.num_steps):
            #TODO: add var
            if t==0:
                addDeviceVariation_SA(wts, dev_param)
            
            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                if self.bntt_flag is True:
                    mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))

                else:
                    mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + (self.conv_list[i](out_prev))

                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0 # nomatter how negative it is?
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda() #? what is the shape
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()


                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()


            out_prev = out_prev.reshape(batch_size, -1)

            if self.bntt_flag is True:
                mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            else:
                mem_fc1 = self.leak_mem * mem_fc1 + (self.fc1(out_prev))

            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev) #! no reset for mem_fc2, but averrage at the end

        out_voltage = mem_fc2 / self.num_steps


        return out_voltage # how will the output be classified?



class SNN_VGG9_BNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=0.95, img_size=32,  num_cls=10, bntt_flag = False, sram=False, var=0):
        super(SNN_VGG9_BNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        self.bntt_flag = bntt_flag
        self.sram = sram
        # print (">>>>>>>>>>>>>>>>>>> VGG 9 >>>>>>>>>>>>>>>>>>>>>>")
        # print ("***** time step per batchnorm".format(self.batch_num))
        # print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        if self.sram:
            self.conv1 = Conv2d_sram(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        elif tern:
            self.conv1 = quanConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag, var=var)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        if self.sram:
            self.conv2 = Conv2d_sram(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        elif tern:
            self.conv2 = quanConv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag, var=var)
        else:
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        if self.sram:
            self.conv3 = Conv2d_sram(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        elif tern:
            self.conv3 = quanConv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag, var=var)
        else:
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        if self.sram:
            self.conv4 = Conv2d_sram(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        elif tern:
            self.conv4 = quanConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag, var=var)
        else:
            self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        if self.sram:
            self.conv5 = Conv2d_sram(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        elif tern:
            self.conv5 = quanConv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag, var=var)        
        else:
            self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        if self.sram:
            self.conv6 = Conv2d_sram(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        elif tern:
            self.conv6 = quanConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag, var=var)        
        else:
            self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        if self.sram:
            self.conv7 = Conv2d_sram(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        elif tern:
            self.conv7 = quanConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag, var=var)        
        else:
            self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)


        self.fc1 = nn.Linear((self.img_size//8)*(self.img_size//8)*256, 1024, bias=bias_flag)
        self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(1024, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt7, self.bntt_fc]
        self.pool_list = [False, self.pool1, False, self.pool2, False, False, self.pool3]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)




    def forward(self, inp, dev_param, wts):

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv4 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv5 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv6 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv7 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7]

        mem_fc1 = torch.zeros(batch_size, 1024).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()



        for t in range(self.num_steps):
            if t==0 and not tern:
            #FIXME: change to ITE
                addDeviceVariation_SA(wts, dev_param)
            # addDeviceVar_bbb(wts,dev_param,bits) # for zy_var

            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                if self.bntt_flag is True:
                    mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](
                        self.conv_list[i](out_prev))
                    # no sense to plot output of conv because input is random, hard to compare
                else:
                    mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + (self.conv_list[i](out_prev))

                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()


                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()


            out_prev = out_prev.reshape(batch_size, -1)

            if self.bntt_flag is True:
                mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            else:
                mem_fc1 = self.leak_mem * mem_fc1 + (self.fc1(out_prev))
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        out_voltage = mem_fc2 / self.num_steps


        return out_voltage

## not test yet
class SNN_VGG11_BNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=0.95, img_size=32,  num_cls=10):
        super(SNN_VGG11_BNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps

        print (">>>>>>>>>>>>>>>>> VGG11 >>>>>>>>>>>>>>>>>>>>>>>")
        print ("***** time step per batchnorm".format(self.batch_num))
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False




        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool4 = nn.AvgPool2d(kernel_size=2)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt8 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool5 = nn.AdaptiveAvgPool2d((1,1))


        self.fc1 = nn.Linear(512, 4096, bias=bias_flag)
        self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(4096, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt7, self.bntt8, self.bntt_fc]
        self.pool_list = [self.pool1, self.pool2, False, self.pool3, False, self.pool4, False, self.pool5]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)




    def forward(self, inp):

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv3 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4).cuda()
        mem_conv4 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4).cuda()
        mem_conv5 = torch.zeros(batch_size, 512, self.img_size // 8, self.img_size // 8).cuda()
        mem_conv6 = torch.zeros(batch_size, 512, self.img_size // 8, self.img_size // 8).cuda()
        mem_conv7 = torch.zeros(batch_size, 512, self.img_size // 16, self.img_size // 16).cuda()
        mem_conv8 = torch.zeros(batch_size, 512, self.img_size // 16, self.img_size // 16).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7, mem_conv8]

        mem_fc1 = torch.zeros(batch_size, 4096).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()



        for t in range(self.num_steps):

            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()


                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()


            out_prev = out_prev.reshape(batch_size, -1)

            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)


        out_voltage = mem_fc2 / self.num_steps

        return out_voltage

class SNN_VGG19_BNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=0.95, img_size=32,  num_cls=10, bntt_flag = False, sram=False, var=0):
        super(SNN_VGG19_BNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        self.bntt_flag = bntt_flag

        print (">>>>>>>>>>>>>>>>> VGG19 >>>>>>>>>>>>>>>>>>>>>>>")
        print ("***** time step per batchnorm".format(self.batch_num))
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt8 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt9 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt10 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt11 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt12 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool4 = nn.AvgPool2d(kernel_size=2)

        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt13 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt14 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt15 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt16 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        # self.pool5 = nn.AdaptiveAvgPool2d((1,1))
        self.pool5 = nn.AvgPool2d(kernel_size=2)


        self.fc1 = nn.Linear((self.img_size//32)*(self.img_size//32)*512, 1024, bias=bias_flag)
        self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(1024, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, 
                        self.conv9, self.conv10, self.conv11, self.conv12, self.conv13, self.conv14, self.conv15, self.conv16]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt7, self.bntt8,
                        self.bntt9, self.bntt10, self.bntt11, self.bntt12, self.bntt13, self.bntt14, self.bntt15, self.bntt16, 
                        self.bntt_fc]
        self.pool_list = [False, self.pool1, False, self.pool2, False, False, False, self.pool3, False, False, False, self.pool4, 
                         False, False, False, self.pool5]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, inp, dev_param, wts):
        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv4 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv5 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv6 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv7 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv8 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv9 = torch.zeros(batch_size, 512, self.img_size//8, self.img_size//8).cuda()
        mem_conv10 = torch.zeros(batch_size, 512, self.img_size//8, self.img_size//8).cuda()
        mem_conv11 = torch.zeros(batch_size, 512, self.img_size//8, self.img_size//8).cuda()
        mem_conv12 = torch.zeros(batch_size, 512, self.img_size//8, self.img_size//8).cuda()
        mem_conv13 = torch.zeros(batch_size, 512, self.img_size//16, self.img_size//16).cuda()
        mem_conv14 = torch.zeros(batch_size, 512, self.img_size//16, self.img_size//16).cuda()
        mem_conv15 = torch.zeros(batch_size, 512, self.img_size//16, self.img_size//16).cuda()
        mem_conv16 = torch.zeros(batch_size, 512, self.img_size//16, self.img_size//16).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7,
                         mem_conv8, mem_conv9, mem_conv10, mem_conv11, mem_conv12, mem_conv13,
                         mem_conv14, mem_conv15, mem_conv16]
        mem_fc1 = torch.zeros(batch_size, 1024).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()

        for t in range(self.num_steps):
            
            if t == 0:
                addDeviceVariation_SA(wts, dev_param)
            
            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                if self.bntt_flag is True:
                    mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](
                        self.conv_list[i](out_prev))
                else:
                    mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + (self.conv_list[i](out_prev))

                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()


                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()            

            out_prev = out_prev.reshape(batch_size, -1)

            if self.bntt_flag is True:
                mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            else:
                mem_fc1 = self.leak_mem * mem_fc1 + (self.fc1(out_prev))
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        out_voltage = mem_fc2 / self.num_steps

        return out_voltage                    

