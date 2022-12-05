#############################################
#   @author: Runcong K @ASU, PI: Prof. Cao  #
#############################################

#--------------------------------------------------
# Imports
#--------------------------------------------------
from collections import OrderedDict # if typing is not supported, use collections
# from typing import OrderedDict
import torch.optim as optim
import torchvision
from   torch.utils.data.dataloader import DataLoader
from   torchvision import transforms
# from model import func

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import os.path
import numpy as np
import torch.backends.cudnn as cudnn
from utills import *

import logging
import time

cudnn.benchmark = True
cudnn.deterministic = True

#--------------------------------------------------
# Parse input arguments
#--------------------------------------------------
parser = argparse.ArgumentParser(description='SNN trained with BNTT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed',                  default=1234,        type=int,   help='Random seed')
parser.add_argument('--num_steps',             default=20,    type=int, help='Number of time-step')
parser.add_argument('--batch_size',            default=256,       type=int,   help='Batch size')
parser.add_argument('--lr',                    default=0.1,   type=float, help='Learning rate')
parser.add_argument('--leak_mem',              default=0.99,   type=float, help='Leak_mem')
parser.add_argument('--arch',              default='vgg9',   type=str, help='Dataset [vgg9, vgg5]')
parser.add_argument('--dataset',              default='cifar10',   type=str, help='Dataset [cifar10, cifar100]')
parser.add_argument('--num_epochs',            default=120,       type=int,   help='Number of epochs')
parser.add_argument('--num_workers',           default=4, type=int, help='number of workers')
parser.add_argument('--train_display_freq',    default=5, type=int, help='display_freq for train')
parser.add_argument('--test_display_freq',     default=5, type=int, help='display_freq for test')

parser.add_argument('--quant',     default=1, type=int, help='quantization-bits')
# parser.add_argument('--sram_quant',     action='store_true', help='quantization-sram')

sram_preload=parser.add_mutually_exclusive_group() 

sram_preload.add_argument('--load_sram_w',     action='store_true', help='ini sram w sram weight')
sram_preload.add_argument('--load_rram_w',     action='store_true', help='ini sram w rram weight')
sram_preload.add_argument('--sram_init0',     action='store_true', help='sram_init_zeros')

parser.add_argument('--sram_noqt',     action='store_true', help='sram not quant for eva')
parser.add_argument('--shift',     action='store_true', help='shift sram to MSB of rram')
parser.add_argument('--tern',     action='store_true', help='ternerize qt')
parser.add_argument('--Dorefa',     action='store_true', help='Dorefa qt')
parser.add_argument('--shift_bit',    default=0, type=int, help='number of bits to shift left')
parser.add_argument('--sram_bit',    default=1, type=int, help='sram quant to #bit')
parser.add_argument('--XNORBP',    action='store_true', help='self-define XNOR BP')

parser.add_argument('--var_scale',    default=1, type=float, help='var scale')

### for pruning
parser.add_argument('--swp', action = 'store_true', help='prune sram')
parser.add_argument('--group',    default=16, type=float, help='group size for group lasso pruning')
parser.add_argument('--lamda',    default=0.0001, type=float, help='for group lasso pruning')
parser.add_argument('--ratio',    default=1.0, type=float, help='group lasso intra group weight min penalty')
parser.add_argument('--th',    default=0.005, type=float, help='prune threshold')
parser.add_argument('--noprune1',  action = 'store_false', help='default not prune 1st conv')


## for sram training
parser.add_argument('--eva', action = 'store_true')
parser.add_argument('--sram', action = 'store_true')
parser.add_argument('--no_var', action = 'store_true')
parser.add_argument('--ct', action = 'store_true', help='Fine tune')

## ------- file interface ------
parser.add_argument('--model_name', default= '', type=str, help='saved_model_name')
parser.add_argument('--log_name', default = 'random_log', type = str, 
                help='name of saved log file')
parser.add_argument('--load_rram',  type = str, metavar='PATH', default='',
                help='checkpoint name of loaded model')
parser.add_argument('--load_sram',  type = str, metavar='PATH', default='',
                help='checkpoint name of loaded model')


# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--gpu_id', type=int, default=0,
					help='device range [0,ngpu-1]')

global args
args = parser.parse_args()
## --- test args --- ##
# if args.model_name:
#     print('yes', args.model_name)
# else:
#     print('no')
# exit()

## ------ var to pass through files ------ ##
sram = args.sram
shift = args.shift
log_name = args.log_name
sram_init0 = args.sram_init0
tern = args.tern
load_sram_w = args.load_sram_w
# global shifter  #? shifteer is assigned before global declaration?
# move to MSB
# shifter = 2 ** (args.quant - 1) #TODO: move into function
shifter = 2 ** (args.shift_bit) 

bits = args.quant

###### ----- get date and time -----
from datetime import date, datetime
today = date.today()

BNTT_flag = True #!! need this, or accuracy goes low

#--------------------------------------------------
# Initialize file names
#--------------------------------------------------
if not os.path.isdir('save'):
    os.mkdir('save')
log_dir = 'save/' + 'log_' +  str(today) # saved directory name with date
model_dir = 'save_model/modelsave_' +  str(today) 
if args.eva:
    log_dir = log_dir + '_eva'
if sram:
    log_dir = log_dir + '_sram' # saved directory name with date
if args.tern:
    log_dir = log_dir + '_tern' # saved directory name with date
if not args.ct and not args.eva:
    log_dir = log_dir + '_tfs' # saved directory name with date
# exit() 
if os.path.isdir(log_dir) is not True:
    os.mkdir(log_dir)
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
# global log_dir2
# log_dir2 = log_dir + '/log'
# if os.path.isdir(log_dir2) is not True:
#     os.mkdir(log_dir2)

### import ###
from model import * # why I import here is because 
#                     # log_dir2 and log_name should be assigned first  
#                     # then model's class which use them can be imported.

# user_foldername is saved_checkpoint name
if args.model_name:
    user_foldername = args.log_name
else:
    ## used for saved_model_name
    if BNTT_flag:
        user_foldername = ((args.dataset)+(args.arch)+'_BNTT' +
                                '_timestep'+str(args.num_steps) +'_lr'+str(args.lr) + 
                                '_epoch' + str(args.num_epochs) + '_leak' + str(args.leak_mem) + 
                                '_r' + str(args.quant) + '_nonvar_' + str(args.no_var) +
                                '_sram_' + str(args.sram) + str(args.sram_bit) +'_ct_' + str(args.ct)+'_tern_'+str(args.tern)+
                                '_shift'+ str(args.shift_bit)
                            )
        # add sram
        if args.sram:
            user_foldername = user_foldername + '_srLD'
            # sram preload value
            if args.load_sram_w:
                user_foldername += 'sr'
            elif args.load_rram_w:
                user_foldername += 'rr'
            elif args.sram_init0:
                user_foldername += 'ini0'
            else: #xavier
                user_foldername += 'xavier'
        if args.eva:
            ind_ech = user_foldername.find('_epoch')
            ind_leak = user_foldername.find('_leak')
            ind_ct = user_foldername.find('_ct_')
            user_foldername = user_foldername[:ind_ech]+user_foldername[ind_leak:ind_ct]+'eva'
        # if args.sram_bit != 0:
        if args.var_scale:
            ind = user_foldername.find('_nonvar_')
            user_foldername = user_foldername[:ind] + '_var' +str(args.var_scale)+user_foldername[ind:]
        # print(user_foldername)
        # sys.exit()
    else:
        user_foldername = ((args.dataset)+(args.arch)+
                                '_timestep'+str(args.num_steps) +'_lr'+str(args.lr) + 
                                '_epoch' + str(args.num_epochs) + '_leak' + str(args.leak_mem) + 
                                '_quant' + str(args.quant) + '_nonvar_' + str(args.no_var) +
                                '_sram_' + str(args.sram) +'_ct_' + str(args.ct)+'_tern_'+str(args.tern)+
                                '_shift_'+ str(args.shift)
                            )

user_foldername = str(today)+user_foldername

#-------------- *** initialize logger ***
logger_file = os.path.join(log_dir,"log_{}".format(args.log_name))
logging.basicConfig(filename = logger_file, format='%(process)d-%(asctime)s %(message)s',
                    filemode='w',level = logging.INFO) #! rewrite now, default is append - 'a'
##*** because of basicConfig, other logger will also write to this file.
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
## --- if I need to overwrite the logfile, do this:
## handler = logging.FileHandler(log_filename, 'w+')
## logger.addHandler(handler)
logger = logging.getLogger('main_log')
logger.addHandler(ch)
#-------------- *** finish initializing logger ***

#------------------- print what I am running -----------------
print_args_list = ['arch','batch_size','ct','gpu_id','group','lamda','load_rram','load_sram', 'load_rram_w', 'load_sram_w','lr','num_epochs','num_steps','quant','sram','swp','th','var_scale','log_name']
logger.info("------ Running command: ------")
logger.info(str(sys.argv))
# logger.info(str('python train.py \\'))
# for k,v in sorted(vars(args).items()):
#     if (k in print_args_list):
#         logger.info('{} {} \\'.format(k,v))
# exit()
#-------------------  -------------------

## ---------------- GPU specify, CUDA setting -------------------
## ---------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


if args.ngpu == 1:
	# make only device #gpu_id visible, then
	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
### below lines cannot be put into same if block... weird
if args.ngpu == 1:    
    # logger.info("------------ ******** cuda device: {}********------------ ".format(torch.cuda.current_device()))
    logger.info("args.gpu_id: {}".format(str(args.gpu_id))) ## TODO: it's always 0, is it correct?
    # logger.info("are they the same?")
    # device = torch.cuda.current_device()
use_cuda = torch.cuda.is_available()

## ---------------------------------------------------------
# exit()

# ************ write file setting **********
""" def print_log(string, logfile):
    print("{}".format(string))
    logfile.write('{}\n'.format(string))
    logfile.flush() """
# log_file = open(os.path.join(log_dir,"log","log_{}".format(args.log_name)), 'w')
# log_file = open(os.path.join(log_dir2,"log_{}".format(args.log_name)), 'w') # remember to close

## add whether noise here

#--------------------------------------------------
# Initialize seed
#--------------------------------------------------
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
#random.seed(seed)

#--------------------------------------------------
# SNN configuration parameters
#--------------------------------------------------
# Leaky-Integrate-and-Fire (LIF) neuron parameters
leak_mem = args.leak_mem

# SNN learning and evaluation parameters
batch_size      = args.batch_size
batch_size_test = args.batch_size*2 #? why test batch size diff?
num_epochs      = args.num_epochs
num_steps       = args.num_steps
lr   = args.lr


# ## --------- just for debug -------------- ##
# img_size = 32
# num_cls = 10

##=============== timing report logging ==========
start_time = datetime.now()
logger.info("======== starting time : {}::{} ============".format(str(today),start_time.strftime("%H:%M:%S")))
start_time = time.time()

#--------------------------------------------------
# Instantiate the SNN model and optimizer
#--------------------------------------------------
if args.dataset == 'cifar10':
    num_cls = 10
    img_size = 32
elif args.dataset == 'cifar100':
    num_cls = 100
    img_size = 32
else:
    logger.info("not implemented yet..")
    exit()
#--------------------------
## add var
#--------------------------
if args.no_var: # for tern, it's adding noise in forward.
    dev_param = 0
else:
    if (bits == 1):
        dev_param = 0.08164
    elif (bits == 2):
        dev_param = 0.12542
    elif (bits == 3):
        dev_param = 0.2062
    elif (bits == 4):
        dev_param = 0.34597
    elif (bits == 8):
        dev_param = 0.369039
if args.var_scale:
    dev_param = dev_param * args.var_scale
# -------------
if args.arch == 'vgg19':
    # model = SNN_VGG9_BNTT(num_steps = num_steps, leak_mem=leak_mem, img_size=img_size,  num_cls=num_cls, sram =sram, shift = shift, shifter = shifter)
    model = SNN_VGG19_BNTT(num_steps = num_steps, leak_mem=leak_mem, img_size=img_size,  num_cls=num_cls, bntt_flag = BNTT_flag, sram=sram, var=dev_param)
elif args.arch == 'vgg9':
    # model = SNN_VGG9_BNTT(num_steps = num_steps, leak_mem=leak_mem, img_size=img_size,  num_cls=num_cls, sram =sram, shift = shift, shifter = shifter)
    model = SNN_VGG9_BNTT(num_steps = num_steps, leak_mem=leak_mem, img_size=img_size,  num_cls=num_cls, bntt_flag = BNTT_flag, sram=sram, var=dev_param)
# elif args.arch == 'vgg11':
#     model = SNN_VGG11_BNTT(timesteps = num_steps, leak_mem=leak_mem, img_size=img_size,  num_cls=num_cls)
elif args.arch == 'vgg5':
    # model = SNN_VGG5_BNTT(num_steps = num_steps, leak_mem=leak_mem, img_size=img_size,  num_cls=num_cls, sram =sram, shift = shift, shifter = shifter)
    model = SNN_VGG5_BNTT(num_steps = num_steps, leak_mem=leak_mem, img_size=img_size,  num_cls=num_cls, bntt_flag = BNTT_flag, sram = sram)

else:
    logger.info("not implemented yet..")
    exit()

## ---------------
## move to device
## ---------------
if use_cuda:
    if args.ngpu > 1: #parallel computation
        # net = torch.nn.DataParallel(net, device_ids=[1,2])
        model = torch.nn.DataParallel(model) # careful when parallel, as it should be also loaded as prl
    model = model.cuda()
#-------------------------------------------------
# print conv layer names
# print("----------------- print created model -----------------")
# for para in model.state_dict():
#     print(para) #, "\t", model.state_dict()[para].size())
    # if '.WR' in para:
    #     print(model.state_dict()[para][1]) ## suppose to be all 0s


#------------------------------------------------- -------------------------------------------------
# --------------------------- load model from rram quant pretrained ----------------------
#------------------------------------------------- -------------------------------------------------

if args.ct or args.eva:
    if args.load_sram:
        load_file = args.load_sram #No_var train
        state_dict_sram = torch.load(load_file)['state_dict']
    load_file = args.load_rram #var train
    state_dict_rram = torch.load(load_file)['state_dict']

#### -------------------- loading models ---------------- ####

## ---------------- avoid loading pretrained rram to sram ------------ ##
## --------- create new list, then to update
## credit to: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113

    if args.sram: #! sram must have pretrained model
        logger.info(" ---------  sram loading model --------- ")
        logger.info(" =========  rram weitght stored to register buffer (named_register_buffers()) ========= ")
        logger.info(" =========  sram weitght stored to layer.weight (named_parameters()) ========= ")

        model_dict = model.state_dict().copy() #! inplace or copy? copy!

        ## -- load param except conv weight (conv only has weight, so no need to care bias)
        pretrained_dict = OrderedDict()
        pretrained_dict = {k:v for k, v in state_dict_rram.items() if ('conv' not in k and 'weight' not in k)}
        model_dict.update(pretrained_dict)
        
        ## --  init sram weight --
        if args.sram_init0:
            initial_conv_dict = OrderedDict()
            initial_conv_dict = {k: torch.zeros_like(v, requires_grad=True) for k, v in state_dict_rram.items() if ('conv' in k and 'weight' in k)}
            model_dict.update(initial_conv_dict)
        elif args.load_sram_w:
            sram_w = OrderedDict()
            sram_w = {k:v for k,v in state_dict_sram.items() if ('conv' in k and 'weight' in k)}
            model_dict.update(sram_w)
        elif args.load_rram_w:
            rram_w = OrderedDict()
            rram_w = {k:v for k,v in state_dict_rram.items() if ('conv' in k and 'weight' in k)}
            model_dict.update(rram_w)            
        #else: conv will stays xavier ini

        model.load_state_dict(model_dict,False) # not load RB yet
        # print("----------------- print new model -----------------")

        ### ------------ print conv to see if ini 0s for debug --------- ###
        """ for param, val in model.state_dict().items():
            if 'conv' in param:
                print(param, ":", val) # suppose to be 0s
        """        
        # print(model.state_dict()['conv1.weight'][2])
        # exit()

        ##update register buffer value using pretrained rram weights
        ## --------- load RB as rram now --- ################
        ## FIXME: need to be customized: for ternary, DoReFa
        for name,buf in model.named_buffers():
            if ".WR" in name and "_1" in name:
                # weight = state_dict[name[0:5]+"_1.weight"].detach().clone()
                weight = state_dict_rram[name[0:5]+"_1.weight"].clone()
                buf.data = weight.data
            elif ".WR" in name:
                # weight = state_dict[name[0:5]+".weight"].detach().clone()
                weight = state_dict_rram[name[0:5]+".weight"].clone()
                buf.data = weight.data    

    else: # no sram, only rram
        logger.info("==== only load rram weight into layer.weight === ")
        model.load_state_dict(state_dict_rram, True)
        # model.load_state_dict(state_dict)
    # *********** debug ********* FIXME:
    # print(model.state_dict().keys())
    # exit() 
## ------------- no load model ------------
## ------------- ------------ ------------
else:
    logger.info("--- === train from screcth --- ===")
    
# ------------- check weight or WR after init here --------------
# for name,para in model.named_buffers():
#     if "WR" in name and 'conv' in name:
#         print(name)
#         print(torch.min(para))
#         # print(buf.equal(state_dict[name[0:5]+"_1.weight"]))
# print("===========")
# for name,para in state_dict.items():
#     if "weight" in name and 'conv' in name:
#         print(name)
#         print(torch.min(para))

# exit()
    # elif ".WR" in name:
    #     print(buf.equal(state_dict[name[0:5]+".weight"]))

# for name,wt in model.named_parameters():
#     if "conv" in name and "weight" in name:
#         print(name,":", wt[2,0:2,0:2])
        
# exit()   

#-----------quant class -----------
class BinOp():
    def __init__(self, model, sram = False):
        # count the number of Conv2d
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d): # if isinstance(m, Conv2d_sram) #* same
            # if isinstance(m, Conv2d_sram): # new conv layer with buffer
                count_Conv2d = count_Conv2d + 1
        logger.info("num_of_conv:{}".format(count_Conv2d))
        self.sram = sram
        # return #! for debug
        # start_range = 1
        #raw_input()
        self.num_of_params = count_Conv2d # num of conv #TODO: test
        if self.sram:
            self.num_of_params = self.num_of_params * 2
        self.saved_params = []   # store conv weights
        # self.target_params = []  
        self.target_modules = [] # store conv weights
        # index = -1
        # print(self.num_of_params) # = count-Conv2d
        for m in model.modules():  #TODO: change this to store differnt
            if isinstance(m, nn.Conv2d): #TODO: how to make this conv2d_sram
                logger.info(str(m))
                # index = index + 1
               
                # logger.info('Making k-bit') #Know which layers weights are being made k-bit #? why copy
                ######* add WR to list for quant #####
                tmp = m.weight.data.clone()
                if self.sram:
                    tmp_buf = m.WR.data.clone()
                self.saved_params.append(tmp) # .weight, trainable
                if self.sram:
                    self.saved_params.append(tmp_buf) # .WR, untrainable 
                self.target_modules.append(m.weight) # not clone, so doing in-place operation on model param
                if self.sram:
                    self.target_modules.append(m.WR) # append WR after weight each layer

    def binarization(self):
        #self.meancenterConvParams()
        #self.clampConvParams()
        self.save_params() # * save before quant, and load after forward
        self.binarizeConvParams()

    ## conv weight - mean of that
    def meancenterConvParams(self):
        for index in range(self.num_of_params): # loop through conv layer
            s = self.target_modules[index].data.size()
            #print(index)
            #print(s)
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self): # save processed target_modules to saved_params
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data) 
    
    # inverse of save_params
    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    ##  --- quant method --- ##
    def XNOR_quant(self,x, index):
        # num_bits = weight_conn # defined in global later
        if args.sram and args.shift and index%2==0: #FIXME: why args.shift here?
            num_bits = args.sram_bit # defined in global later
        else:
            num_bits = args.quant
        xmax = x.abs().max()
        v0 = 1
        v1 = 2
        v2 = -0.5
        # y = num_bits[index]#+std[index]#torch.normal(0,std[index], size=(1,1))#2.**num_bits[index] - 1.
        y = 2**num_bits-1 #+std[index]#torch.normal(0,std[index], size=(1,1))#2.**num_bits[index] - 1.
        #print(y)
        x = x.add(v0).div(v1)
        #print(x)
        x = x.mul(y).round_()
        x = x.div(y)
        x = x.add(v2)
        x = x.mul(v1)
        # n_bits = args.quant
        y = 2**(num_bits-1) #! made mistake before, not 2**bits - 1
        W_sbits = torch.round(x * y)
        W_sbits = W_sbits / y
        return W_sbits

    ##  --- quant --- ##
    #### just quant target_module.data
    def binarizeConvParams_old(self):#TODO: let this also quant sram
 
        # num_bits = weight_conn # defined in global later
        num_bits = 0 # defined in global later
        

        for index in range(self.num_of_params * 2): # loop through every conv layer, 2 for each layer
            #### --------
            ## index % 2 == 0 : weight for sram
            ## index % 2 == 1 : RB for fixed RRAM

            # if (index % 2 == 0): #! if only eva rram, uncomment this, comment next line
            # if (index % 2 == 1) or args.sram_quant: # just apply quant to rram weight in RB, or also that in weight for sram
            if (index % 2 == int(args.sram)) or args.sram_quant: # just apply quant to rram weight in RB, not that in weight for sram
                #n = self.target_modules[index].data[0].nelement()
                #s = self.target_modules[index].data.size()
                #m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                #        .sum(2, keepdim=True).sum(1, keepdim=True).div(n)

                #if index in kbit_conn:
                    #k-bit weights
                x = self.target_modules[index].data
                #TODO: need to change the branch condition
                #TODO: need to change the value

                ## ------------ not quant first sram.weight ----------
                # if index == 0:
                #     continue

                ## ------------ not quant first ----------
                # if normal_quant:
                if not args.tern and not args.Dorefa: #XNOR quant
                    xmax = x.abs().max()
                    v0 = 1
                    v1 = 2
                    v2 = -0.5
                    #? num_bits[i] are same
                    y = num_bits[index]#+std[index]#torch.normal(0,std[index], size=(1,1))#2.**num_bits[index] - 1.
                    #print(y)
                    x = x.add(v0).div(v1)
                    #print(x)
                    x = x.mul(y).round_()
                    x = x.div(y)
                    x = x.add(v2)
                    x = x.mul(v1)
                    n_bits = args.quant
                    W_sbits = torch.round(x * 2**(n_bits-1))
                    W_sbits = W_sbits / 2**(n_bits-1)
                    # if index == 0:
                    #     print('saving weights')
                    #     torch.save(W_sbits, './sw_outputs/sw_q_wt1')
                self.target_modules[index].data = W_sbits
                # print(self.target_modules[index].unique())

    ## --- quant --- ###
    ###  update SNN ###
    ## ------------ ###
    def binarizeConvParams(self):#TODO: let this also quant sram
        
        # num_bits = weight_conn # defined in global later
        
        for index in range(self.num_of_params): # loop through every conv layer, 2 for each layer

            # if (index % 2 == 0): #! if only eva rram, uncomment this, comment next line
            # if (index % 2 == 1) or args.sram_quant: # just apply quant to rram weight in RB, or also that in weight for sram
            
            if args.sram_noqt: # just for testing
                if index % 2 == 0: # sram
                    continue
            
            x = self.target_modules[index].data
                #TODO: need to change the branch condition
                #TODO: need to change the value

                ## ------------ not quant first sram.weight ----------
                # if index == 0:
                #     continue

                ## ------------ not quant first ----------

                ### --- ternary quant ---
            if args.tern: ## only tern quant sram in .weight
                pass
                # if index == 0:
                #     thl = -0.1
                #     thr = 0.1
                # else:
                #     if args.quant == 8:
                #         thl = 0.05
                #         thr = 0.25
                #     else:
                #         thl = 0.15
                #         thr = 0.25
                # W_sbits = ternarizeConvParams(x,thl,thr)
                ### --- ternary quant ---

                # if normal_quant:
            else: # XNOR quant
                W_sbits = self.XNOR_quant(x, index)
            self.target_modules[index].data = W_sbits
            # print(self.target_modules[index].unique())

    ### prune ###
    ### ---------------------------- ###
    def LayerWisePrune(self):
        for index in range(self.num_of_params):
            # only prune sram
            if index % 2 == 1:
                continue
            if args.noprune1:
                if index == 0:
                    continue
            ## constant threshold prune
            with torch.no_grad(): # should use saved_param here
                prune_mask = torch.zeros_like(self.saved_params[index])
                th = args.th * self.saved_params[index].data.abs().max()
                prune_mask[self.saved_params[index].abs().gt(th)] = 1
            # apply mask to weight
            x=self.target_modules[index].data
            w_prune = x * prune_mask
            self.target_modules[index].data=w_prune

    #? what is this doing? quant for grad?
    #TODO: let this only update graduate of conv2d?
    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            if self.sram and index % 2 == 1: # when sram, no need update RB -rram
                continue
            weight = self.target_modules[index].data
            n = weight[0].nelement() #output channel 
            s = weight.size()
            m = weight.norm(1, 3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m=m.clone()
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
            # self.target_modules[index].grad.data = \
            #         self.target_modules[index].grad.data.mul(m)
            m = m.mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(3, keepdim=True)\
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)

##TODO: global func now, make it into class later
# ternarize to -1, 0, 1
def ternarizeConvParams(x, thres_l, thres_r):
    x[x.ge(thres_r)] = 1
    x[x.le(thres_l)] = -1
    x[torch.logical_and(x<thres_r,x>thres_l)] = 0
    return x

def prune_loss(model,lamda,group):
    # lamda = torch.tensor(args.lamda).cuda()
    lamda = torch.tensor(lamda).cuda()
    reg_g1 = torch.tensor(0.).cuda()
    reg_g2 = torch.tensor(0.).cuda()
    # group_ch = args.group_ch
    group_ch = group
    # == group-wise defined
    count = 0

    # find the weights for SRAM and do group glasso
    for name,param in model.named_parameters():
        # if name.find('conv_a.weight')>=1 or name.find('conv_b.weight')>=1:
        if 'conv' in name:
            if count != 0:  
                w_l=param     
                kw = w_l.size(2)          

                num_group = w_l.size(0) * w_l.size(1) // group_ch
                # print("wl.size(1)=", w_l.size(1))
                #print(num_group,"num_____________________________group")
                w_l = w_l.view(w_l.size(0), w_l.size(1) // group_ch, group_ch, w_l.size(2), w_l.size(3))
                #print(w_l,w_l.shape)
                w_l = w_l.contiguous().view(num_group, group_ch , kw , kw)
                #print(w_l)
                reg_g1 += glasso_thre(w_l, 1)
            #print(count,"count---------------")
            count += 1
        #if args.arch == 'vgg16_bn':                
        #	if isinstance(m, nn.Linear):
        #		w_f = m.weight
        #		reg_g2 += glasso_thre(w_f, 1)  # channel-wise pruning  
        #		count += 1
    loss = lamda * (reg_g1 )    
    return loss

def glasso_thre(var, dim=0):
	if len(var.size()) == 4:
		var = var.contiguous().view((var.size(0), var.size(1) * var.size(2) * var.size(3)))
	
	a = var.pow(2).sum(dim=dim).pow(1/2)
	mean_a  = a.mean()
	a = torch.min(a, args.ratio*mean_a)
	# thre = args.ratio*mean_a
	# ratio < thre
	return a.sum()

def get_prune_rate(model,group,logger):
    all_num = 0
    group_ch = group
    all_group = 0
    all_nonsparse = 0
    count_num_one = 0
    count=0
    for name,param in model.named_parameters():
        if 'conv' in name:
            if count!=0:
                w_l = param
                
                #print(w_l.size(1))
                # with torch.no_grad():
                #     w_l_nonzero=torch.zeros_like(w_l)

                #     th = 0.005*w_l.abs().max()
                #     w_l_nonzero[w_l.abs().gt(th)]=1
                    
                #     w_l=quantize_normal_fn_1_1(w_l,d_width_sram) #! why quant here

                #     w_l=w_l*w_l_nonzero

                kw=w_l.size(2)
                count_num_layer = w_l.size(0) * w_l.size(1) * kw * kw
                all_num += count_num_layer  # how many elements in this layer

                count_one_layer = len(torch.nonzero(w_l)) # how many elements not zero 
                count_num_one += count_one_layer


                # if((w_l.size(1)//group_ch)==0):
                #     IPython.embed()
                
                num_group = w_l.size(0) * w_l.size(1) // group_ch
                w_l = w_l.view(w_l.size(0), w_l.size(1) // group_ch, group_ch, w_l.size(2), w_l.size(3))
                w_l = w_l.contiguous().view((num_group, group_ch , kw , kw))
                #print(w_l.norm(p=2,dim=1).norm(p=2,dim=1).norm(p=2,dim=1))
                group_nonzeros=len(torch.nonzero(w_l.norm(p=2,dim=1).norm(p=2,dim=1).norm(p=2,dim=1)))
                

                num_nonzeros = group_nonzeros
                all_group += num_group
                all_nonsparse += num_nonzeros            
            count+=1
    overall_sparsity = 1 - (count_num_one / all_num)
    #sparsity = weight_total_zero.item()/(weight_total)
    group_sparsity = 1 - (all_nonsparse / all_group)
    print("The model overall sparsity of:", overall_sparsity,"group sparsity",group_sparsity)
    logger.info(
    '  The model overall sparsity of: {overall_sparsity:.7f} group sparsity {group_sparsity:.7f}'.format(overall_sparsity=overall_sparsity,group_sparsity=group_sparsity))
#global k
# global weight_conn

bits = args.quant
# weight_conn=2**bits-1

#? why?
# just leave it here now
# weight_conn=np.array([2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 
#                          2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 
#                          2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1, 2**bits-1])


if args.tern:
    pass
else:
    bin_op = BinOp(model,args.sram)
# for w in bin_op.target_modules:
#     print(torch.count_nonzero(w))
# bin_op.binarization() #! for debug

# exit()

#--------------------------------------------------
# Load  dataset
#--------------------------------------------------

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #TODO: std value is not same as ANN
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


if args.dataset == 'cifar10':
    num_cls = 10
    img_size = 32  

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)
elif args.dataset == 'cifar100':
    num_cls = 100
    img_size = 32

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)
else:
    logger.info("not implemented yet..")
    exit()



trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

#--------------------------------------------------
# Loaded  dataset
#--------------------------------------------------

criterion = nn.CrossEntropyLoss()

if use_cuda:
     ### --- if pretrained model is not parallelly trained, 
     ### --- don't do parallel
    criterion.cuda()

# Configure the loss function and optimizer
criterion = nn.CrossEntropyLoss() #? why need this line? it is already on GPU?
optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9,weight_decay=1e-4) # only parameters()
if args.sram: #* TODO:epoch 50
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) #FIXME: when do I change this?
else:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
# FIXME: where is 'scheduler' used?
best_acc = 0


#--------------------------------------------------
# Train the SNN using surrogate gradients
#--------------------------------------------------
logger.info('********** SNN training and evaluation **********')
train_loss_list = []
test_acc_list = []


## CODE FOR TRAINING 
if (not args.eva):    
    logger.info("========== begin training and testing =========")
    ## ============= ###
    ## ===train=== ###
    ## ============= ###
    for epoch in range(num_epochs):
        train_loss = AverageMeter()
        model.train()
        
        for i, data in enumerate(trainloader):
            # print("=== begin data ", i, "===")
            #* for debug: compare pre/post train
            # pre_module = bin_op.target_modules.copy()
            # for ele in pre_module:
            #     ele.detach()
            if not args.tern and not args.Dorefa:
                bin_op.binarization()
            if args.swp:
                bin_op.LayerWisePrune() #compute mask based on FP weight, mask out low weight (in-place)
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            
            if args.sram:
                output = model(inputs, dev_param, model.named_buffers())
            else:
                output = model(inputs, dev_param, model.named_parameters())

            loss   = criterion(output, labels)
            #FIXME: 6/12/2022, change order of restore and backward -- ref from train_old


            prec1 = accuracy(output, labels, topk=(1, ))
            train_loss.update(loss.item(), labels.size(0))

            loss.backward()
            # if args.XNORBP:
            #     bin_op.updateBinaryGradWeight()
            if not args.tern and not args.Dorefa:
                bin_op.restore()
            if args.swp: #FIXME: testing: loss backward on FP weight
                # bin_op.restore()
                pr_loss = prune_loss(model,args.lamda,args.group)
                pr_loss.backward()
            optimizer.step()
        # calculate prune rate

        if (epoch+1) % args.train_display_freq ==0:
            if args.swp:
                bin_op.save_params()
                bin_op.LayerWisePrune()
                get_prune_rate(model,args.group,logger)
                bin_op.restore()
            # print_log("Epoch: {}/{};".format(epoch+1, num_epochs), "########## Training loss: {}".format(train_loss.avg))
            logger.info("Epoch: {}/{};" "########## Training loss: {}".format(epoch+1, num_epochs,train_loss.avg))

        adjust_learning_rate(optimizer, epoch, num_epochs)


        ## ============= ###
        ## ===validate=== ###
        ## ============= ###
        if (epoch+1) %  args.test_display_freq ==0:
            acc_top1, acc_top5 = [], []
            model.eval()

            
            with torch.no_grad():
                for j, data in enumerate(testloader, 0):
                    if not args.tern and not args.Dorefa:

                        bin_op.binarization() #! this should be inside each data batch
                    if args.swp:
                        bin_op.LayerWisePrune()
                    images, labels = data
                    images = images.cuda()
                    labels = labels.cuda()

                    if args.sram:
                        out = model(images, dev_param, model.named_buffers()) 
                    else:
                        out = model(images, dev_param, model.named_parameters()) 
                    prec1 = accuracy(out, labels, topk=(1,))
                    # acc_top1.append(float(prec1))  #TODO: remove float here
                    acc_top1.append(prec1)
                    # acc_top5.append(float(prec5))
                    
                    if not args.tern and not args.Dorefa:
                        bin_op.restore() # need to restore weights without variance

            top_np_list = []
            for top_list in acc_top1:
                top_np = [item.detach().cpu().numpy() for item in top_list]
                top_np_list.append(top_np)
            # print(top_np_list)
            # test_accuracy = torch.mean(torch.stack(torch.stack(acc_top1)))
            test_accuracy = np.mean(top_np_list)
            # logger.info ("test_accuracy : {}". format(test_accuracy), log_file)
            logger.info("test_accuracy : {}". format(test_accuracy))

            # Model save
            if best_acc < test_accuracy:
                best_acc = test_accuracy

                model_dict = {
                        'global_step': epoch + 1,
                        'state_dict': model.state_dict(),
                        'accuracy': test_accuracy}
                if (sram):
                    torch.save(model_dict, model_dir+'/'+user_foldername+'_sram_bestmodel.pth.tar') #TODO: need to change for bit
                else:
                    torch.save(model_dict, model_dir+'/'+user_foldername+'_rram_bestmodel.pth.tar') #TODO: need to change for bit



## CODE FOR INFERENCE WITH NOISE (Put dev_param = 0 if you do not want to include noise during inference)
else:
    logger.info("========== begin dir eva =========")
    ## LOAD THE PARMETERS FOR THE TRAINED SNN MODEL FOR INFERENCE 
    #! only load for direct inference
    # acc_top1, acc_top5 = [], []
    acc_top1 = []
    model.eval()

    ## add code to load conv.weight into conv.WR (reg buffer) for rram

    with torch.no_grad():
        for j, data in enumerate(testloader, 0):
            # print("j=", j)
            # model.load_state_dict(state_dict)
            if not args.tern and not args.Dorefa:
                bin_op.binarization()
            if args.no_var:
                dev_param = 0
            else:
                if (bits == 1):
                    # logger.info("add 1-bit var")
                    dev_param = 0.08164
                elif (bits == 2):
                    dev_param = 0.12542
                elif (bits == 3):
                    dev_param = 0.2062
                elif (bits == 4):
                    dev_param = 0.34597   
                elif (bits == 8):
                    dev_param = 0.369039                                
            if args.var_scale:
                dev_param = dev_param * args.var_scale
            # dev_param = 0.08164 # Here, I have added noise for 1-bit ReRAM weights as I have loaded the 1-bit SNN model  
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            if args.sram:
                out = model(images, dev_param, model.named_buffers())
            else:
                out = model(images, dev_param, model.named_parameters())

            prec1 = accuracy(out, labels, topk=(1,))
            acc_top1.append((prec1))
            # acc_top5.append(float(prec5))
            if not args.tern and not args.Dorefa:
                bin_op.restore()

            #! for debug
            # break
        
    ## for top1 
    top_np_list = []
    for top_list in acc_top1:
        top_np = [item.detach().cpu().numpy() for item in top_list]
        top_np_list.append(top_np)
    # print(top_np_list)
    # test_accuracy = torch.mean(torch.stack(torch.stack(acc_top1)))
    test_accuracy = np.mean(top_np_list)
        # test_accuracy = np.mean(acc_top1)
    logger.info ("test_accuracy : {}". format(test_accuracy))

# log_file.close()

logger.info("best accurary: {}".format(best_acc))
end_time = datetime.now()
logger.info("======== ending time : {} ============".format(end_time.strftime("%H:%M:%S")))
end_time = time.time()

cost_time = end_time - start_time
cost_time = time.strftime("%H:%M:%S", time.gmtime(cost_time))
logger.info("======== cost time : {} ============".format(cost_time))
sys.exit(0)

def bn_update(model, bits, trainloader=trainloader_bn): # need to design trainloader
    acc_top1=[]
    for epoch in range(1):#or 2 or more
        train_loss=AverageMeter()
#         model.train()
        for data in trainloader:
            bin_op.binarization()
            if True:
                if (bits == 1):
                    # logger.info("add 1-bit var")
                    dev_param = 0.08164
                elif (bits == 2):
                    dev_param = 0.12542
                elif (bits == 3):
                    dev_param = 0.2062
                elif (bits == 4):
                    dev_param = 0.34597   
                elif (bits == 8):
                    dev_param = 0.369039                                
            if args.var_scale:
                dev_param = dev_param * args.var_scale
            inputs,labels=data
            inputs = inputs.cuda()
            labels=labels.cuda()
            
            optimizer.zero_grad()
            if args.sram:
                output = model(inputs, dev_param, model.named_buffers())
            else:
                output = model(inputs, dev_param, model.named_parameters())
            loss   = criterion(output, labels)
            bin_op.restore()

            prec1 = accuracy(output, labels, topk=(1, ))
            train_loss.update(loss.item(), labels.size(0))

            loss.backward()
            optimizer.step()
            
            break # just feed one batch now, can modify to feed several batches
        print("train_loss:", train_loss)
