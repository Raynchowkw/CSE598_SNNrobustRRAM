import os
import argparse

parser = argparse.ArgumentParser(description='Choose command to run')
parser.add_argument("--n", type=int,
                    help = "choose one program to run")
parser.add_argument('--task', type =str, choices = ['tfs_nonvar','tfs','var_eva',
                    'tfs_zy','var_eva_zy','sram','sram_sft','sram_sft_bits','sram_other'],
                    help = 'run specific task')
args=parser.parse_args()
#### ---------------------- sram --------------------
####---------------------- ------------------------
if args.task == 'sram':
    log_list= ['sram_q1_new_lr002','sram_q2_new','sram_q3_new', 'sram_q4_new']
    if args.n==1:
        os.system('CUDA_VISIBLE_DEVICES=0 python train.py \
            --num_steps 20 --lr 0.02 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant 1 \
            --sram \
            --ct \
            --load_model "/home2/rkuang1/SNN_tr/SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant1_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 80 \
            --log_name %s \
            ' %(log_list[0])
            )
    if args.n==2:
        os.system('CUDA_VISIBLE_DEVICES=1 python train.py \
            --num_steps 20 --lr 0.06 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 2 \
            --quant 3 \
            --sram \
            --ct \
            --load_model "/home2/rkuang1/SNN_tr/SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant3_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 80 \
            --log_name %s \
            ' %(log_list[1])
            )
    if args.n==3:
        os.system('CUDA_VISIBLE_DEVICES=3 python train.py \
            --num_steps 20 --lr 0.06 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant 4 \
            --sram \
            --ct \
            --load_model "/home2/rkuang1/SNN_tr/SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant4_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 80 \
            --log_name %s \
            ' %(log_list[2])
            )
    if args.n==4:
        os.system('CUDA_VISIBLE_DEVICES=3 python train.py \
            --num_steps 20 --lr 0.06 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 2 \
            --quant 8 \
            --sram \
            --ct \
            --load_model "/home2/rkuang1/SNN_tr/SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant8_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 80 \
            --log_name %s \
            ' %(log_list[3])
            )
# %%
## ==================== sram 1bit + shift =================
####---------------------- ------------------------
if args.task == 'sram_sft':
    log_list= ['sram_q1_sft','sram_q3_sft','sram_q4_sft', 'sram_q8_sft']
    if args.n==3:
        os.system('CUDA_VISIBLE_DEVICES=0 python train.py \
            --num_steps 20 --lr 0.06 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 1 \
            --quant 4 \
            --sram \
            --shift \
            --ct \
            --load_model "/home2/rkuang1/SNN_tr/SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant4_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 80 \
            --log_name %s \
            ' %(log_list[2])
            )
    if args.n==4:
        os.system('CUDA_VISIBLE_DEVICES=0 python train.py \
            --num_steps 20 --lr 0.06 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant 8 \
            --sram \
            --shift \
            --ct \
            --load_model "/home2/rkuang1/SNN_tr/SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant8_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 80 \
            --log_name %s \
            ' %(log_list[3])
            )

# %%
## ==================== sram 1bit + shift n bit =================
####---------------------- ------------------------
if args.task == 'sram_sft_bits':
    # log_list= ['sram_q0_sft0',
    #            'sram_q3_sft0','sram_q3_sft1','sram_q3_sft2',
    #            'sram_q4_sft0','sram_q4_sft1','sram_q4_sft2','sram_q4_sft3',
    #            'sram_q8_sft3','sram_q8_sft3','sram_q8_sft6','sram_q8_sft7']
    if args.n==0:
        os.system('CUDA_VISIBLE_DEVICES=0 python train.py \
            --num_steps 20 --lr 0.05 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant 3 \
            --sram \
            --sram_bit 2 \
            --shift \
            --shift_bit -3\
            --ct \
            --load_model "/home2/rkuang1/SNN_tr/SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant3_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 60 \
            --log_name %s \
            ' %("rram3b_sram2b_Rsht3")
            )
    if args.n==1:
        os.system('CUDA_VISIBLE_DEVICES=0 python train.py \
            --num_steps 20 --lr 0.04 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 2 \
            --quant 3 \
            --sram \
            --sram_bit 2 \
            --shift \
            --shift_bit -1\
            --ct \
            --load_model "/home2/rkuang1/SNN_tr/SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant3_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 60 \
            --log_name %s \
            ' %("rram3b_sram2b_Rsht1")
            )

# %%
if args.task == 'var_eva':
    log_list = ['rram_eva_q1','rram_eva_q3','rram_eva_q4','rram_eva_q8']
    if args.n==1:
        os.system('CUDA_VISIBLE_DEVICES=0 python train.py \
            --num_steps 20 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id %s \
            --quant 1 \
            --eva \
            --load_model "/home2/rkuang1/SNN_tr/SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant1_nonvar_True_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --log_name %s \
            ' %(str(args.n-1), log_list[args.n-1])
            )   
    if args.n==2:
        os.system('CUDA_VISIBLE_DEVICES=1 python train.py \
            --num_steps 20 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id %s \
            --quant 3 \
            --eva \
            --load_model "/home2/rkuang1/SNN_tr/SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant3_nonvar_True_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --log_name %s \
            ' %(str(args.n-1), log_list[args.n-1])
            )   
    if args.n==3:
        os.system('CUDA_VISIBLE_DEVICES=2 python train.py \
            --num_steps 20 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id %s \
            --quant 4 \
            --eva \
            --load_model "/home2/rkuang1/SNN_tr/SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant4_nonvar_True_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --log_name %s \
            ' %(str(1), log_list[args.n-1])
            )   
    if args.n==4:
        os.system('CUDA_VISIBLE_DEVICES=3 python train.py \
            --num_steps 20 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id %s \
            --quant 8 \
            --eva \
            --load_model "/home2/rkuang1/SNN_tr/SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant8_nonvar_True_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --log_name %s \
            ' %(str(args.n-1), log_list[args.n-1])
            )   
## qt tfs no_var (2022-4-4)
## q1 tfs
# log_list = ['q1_tfs_novar','q3_tfs_novar','q4_tfs_novar', 'q8_tfs_novar']
# for i in log_list:
""" os.system('CUDA_VISIBLE_DEVICES=0 python train.py \
    --num_steps 20 --lr 0.1 \
    --arch "vgg9" \
    --dataset "cifar10" \
    --batch_size 128 \
    --leak_mem 0.95 \
    --num_workers 4  \
    --gpu_id 0 \
    --quant 1 \
    --no_var \
    --num_epochs 100 \
    --log_name %s \
    ' %(log_list[0])
     ) """
""" os.system('CUDA_VISIBLE_DEVICES=1 python train.py \
    --num_steps 20 --lr 0.1 \
    --arch "vgg9" \
    --dataset "cifar10" \
    --batch_size 128 \
    --leak_mem 0.95 \
    --num_workers 4  \
    --gpu_id 1 \
    --quant 3 \
    --no_var \
    --num_epochs 100 \
    --log_name %s \
    ' %(log_list[1])
     ) """
""" os.system('CUDA_VISIBLE_DEVICES=2 python train.py \
    --num_steps 20 --lr 0.1 \
    --arch "vgg9" \
    --dataset "cifar10" \
    --batch_size 128 \
    --leak_mem 0.95 \
    --num_workers 4  \
    --gpu_id 2 \
    --quant 4 \
    --no_var \
    --num_epochs 100 \
    --log_name %s \
    ' %(log_list[2])
     ) """
""" os.system('CUDA_VISIBLE_DEVICES=3 python train.py \
    --num_steps 20 --lr 0.1 \
    --arch "vgg9" \
    --dataset "cifar10" \
    --batch_size 128 \
    --leak_mem 0.95 \
    --num_workers 4  \
    --gpu_id 3 \
    --quant 8 \
    --no_var \
    --num_epochs 100 \
    --log_name %s \
    ' %(log_list[3])
     ) """
################ tfs ++++++++++++ var ##################
if args.task == 'tfs_zy':
    log_list = ['q1_tfs_zy_var','q3_tfs_zy_var','q4_tfs_zy_var', 'q8_tfs_zy_var']
    
# for i in log_list:
    if args.n ==1:
        os.system('CUDA_VISIBLE_DEVICES=0 python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant 1 \
            --num_epochs 100 \
            --log_name %s \
            ' %(log_list[0])
            )
    if args.n ==2:
     
            
        os.system('CUDA_VISIBLE_DEVICES=1 python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 1 \
            --quant 3 \
            --num_epochs 100 \
            --log_name %s \
            ' %(log_list[1])
            )
     
    if args.n ==3:
            
        os.system('CUDA_VISIBLE_DEVICES=2 python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 2 \
            --quant 4 \
            --num_epochs 100 \
            --log_name %s \
            ' %(log_list[2])
            )
    ### args.n == 8 is not supported yet
     
"""     
os.system('CUDA_VISIBLE_DEVICES=3 python train.py \
    --num_steps 20 --lr 0.1 \
    --arch "vgg9" \
    --dataset "cifar10" \
    --batch_size 128 \
    --leak_mem 0.95 \
    --num_workers 4  \
    --gpu_id 3 \
    --quant 8 \
    --num_epochs 100 \
    --log_name %s \
    ' %(log_list[3])
     )
"""   
######## above running ######

## q3 tfs
""" log_list = ['q8_tfs_novar']
for i in log_list:
    os.system('CUDA_VISIBLE_DEVICES=3 python train.py \
        --num_steps 20 --lr 0.1 \
        --arch "vgg9" \
        --dataset "cifar10" \
        --batch_size 128 \
        --leak_mem 0.95 \
        --num_workers 4  \
        --gpu_id 3 \
        --quant 8 \
        --no_var \
        --num_epochs 100 \
        --log_name %s \
        ' %(i)) """
## q8 tfs
""" log_list = ['q8_tfs']
for i in log_list:
    os.system('CUDA_VISIBLE_DEVICES=0 python train.py \
        --num_steps 20 --lr 0.1 \
        --arch "vgg9" \
        --dataset "cifar10" \
        --batch_size 128 \
        --leak_mem 0.95 \
        --num_workers 4  \
        --gpu_id 0 \
        --quant 8 \
        --num_epochs 120 \
        --log_name %s \
        ' %(i)) """
## q4 lr 0.1
""" log_list = ['q4_sram_init0']
for i in log_list:
    os.system('CUDA_VISIBLE_DEVICES=1 python train.py \
        --num_steps 20 --lr 0.1 \
        --arch "vgg9" \
        --dataset "cifar10" \
        --batch_size 128 \
        --leak_mem 0.95 \
        --num_workers 4  \
        --gpu_id 1 \
        --quant 4 \
        --ct \
        --num_epochs 100 \
        --sram \
        --sram_init0 \
        --log_name %s \
        ' %(i)) """
## qt XNOR lr 0.05 （2022-4-4）
""" log_list = ['q1_sram_xnor_lr005']
for i in log_list:
    os.system('CUDA_VISIBLE_DEVICES=0 python train.py \
        --num_steps 20 --lr 0.05 \
        --arch "vgg9" \
        --load_model "pretrained_model/cifar10vgg9_timestep20_lr0.1_epoch120_leak0.95_quant1_nonvar_False_bit_rran_bestmodel_test.pth.tar" \
        --dataset "cifar10" \
        --batch_size 128 \
        --leak_mem 0.95 \
        --num_workers 4  \
        --gpu_id 0 \
        --quant 1 \
        --ct \
        --num_epochs 50 \
        --sram \
        --log_name %s \
        ' %(i)) """
#        --sram_init0 \
""" log_list = ['q3_sram_xnor_lr005']
for i in log_list:
    os.system('CUDA_VISIBLE_DEVICES=1 python train.py \
        --num_steps 20 --lr 0.05 \
        --arch "vgg9" \
        --load_model "pretrained_model/cifar10vgg9_timestep20_lr0.1_epoch120_leak0.95_quant3_nonvar_False_sram_False_ct_False_bit_rran_bestmodel_test.pth.tar" \
        --dataset "cifar10" \
        --batch_size 128 \
        --leak_mem 0.95 \
        --num_workers 4  \
        --gpu_id 1 \
        --quant 3 \
        --ct \
        --num_epochs 50 \
        --sram \
        --log_name %s \
        ' %(i)) """
#        --sram_init0 \
""" log_list = ['q4_sram_xnor_lr005']
for i in log_list:
    os.system('CUDA_VISIBLE_DEVICES=2 python train.py \
        --num_steps 20 --lr 0.05 \
        --arch "vgg9" \
        --load_model "pretrained_model/cifar10vgg9_timestep20_lr0.1_epoch120_leak0.95_quant4_nonvar_False_sram_False_ct_False_bit_rran_bestmodel_test.pth.tar" \
        --dataset "cifar10" \
        --batch_size 128 \
        --leak_mem 0.95 \
        --num_workers 4  \
        --gpu_id 2 \
        --quant 4 \
        --ct \
        --num_epochs 50 \
        --sram \
        --log_name %s \
        ' %(i)) """
#        --sram_init0 \
""" log_list = ['q8_sram_xnor_lr005']
for i in log_list:
    os.system('CUDA_VISIBLE_DEVICES=3 python train.py \
        --num_steps 20 --lr 0.05 \
        --arch "vgg9" \
        --load_model "pretrained_model/cifar10vgg9_timestep20_lr0.1_epoch120_leak0.95_quant8_nonvar_False_sram_False_ct_False_bit_rran_bestmodel_test.pth.tar" \
        --dataset "cifar10" \
        --batch_size 128 \
        --leak_mem 0.95 \
        --num_workers 4  \
        --gpu_id 3 \
        --quant 8 \
        --ct \
        --num_epochs 50 \
        --sram \
        --log_name %s \
        ' %(i)) """
#        --sram_init0 \


##### -----------------
""" log_list = ['q3_sram_init0_qt_lr005']
for i in log_list:
    os.system('CUDA_VISIBLE_DEVICES=1 python train.py \
        --num_steps 20 --lr 0.05 \
        --arch "vgg9" \
        --load_model "/home2/rkuang1/SNN_tr/pretrained_model/cifar10vgg9_timestep20_lr0.1_epoch120_leak0.95_quant3_nonvar_False_sram_False_ct_False_bit_rran_bestmodel_test.pth.tar" \
        --dataset "cifar10" \
        --batch_size 128 \
        --leak_mem 0.95 \
        --num_workers 4  \
        --gpu_id 1 \
        --quant 3 \
        --ct \
        --num_epochs 60 \
        --sram \
        --sram_quant \
        --sram_init0 \
        --log_name %s \
        ' %(i))    """

""" log_list = ['q8_sram_init0_qt_lr001']
for i in log_list:
    os.system('CUDA_VISIBLE_DEVICES=0 python train.py \
        --num_steps 20 --lr 0.05 \
        --arch "vgg9" \
        --load_model "pretrained_model/cifar10vgg9_timestep20_lr0.1_epoch120_leak0.95_quant8_nonvar_False_sram_False_ct_False_bit_rran_bestmodel_test.pth.tar" \
        --dataset "cifar10" \
        --batch_size 128 \
        --leak_mem 0.95 \
        --num_workers 4  \
        --gpu_id 0 \
        --quant 8 \
        --ct \
        --num_epochs 60 \
        --sram \
        --sram_init0 \
        --sram_quant \
        --log_name %s \
        ' %(i)) """

""" ##### sram tern
log_list = ['q4_sram_tern']
for i in log_list:
    os.system('CUDA_VISIBLE_DEVICES=0 python train.py \
        --num_steps 20 --lr 0.08 \
        --arch "vgg9" \
        --load_model "/home2/rkuang1/SNN_tr/pretrained_model/cifar10vgg9_timestep20_lr0.1_epoch120_leak0.95_quant4_nonvar_False_sram_False_ct_False_bit_rran_bestmodel_test.pth.tar" \
        --dataset "cifar10" \
        --batch_size 128 \
        --leak_mem 0.95 \
        --num_workers 4  \
        --gpu_id 0 \
        --quant 4 \
        --ct \
        --num_epochs 80 \
        --sram \
        --sram_quant \
        --tern \
        --log_name %s \
        ' %(i))    """        
##### sram tern + shift
""" log_list = ['q4_sram_tern_shf']
for i in log_list:
    os.system('CUDA_VISIBLE_DEVICES=1 python train.py \
        --num_steps 20 --lr 0.08 \
        --arch "vgg9" \
        --load_model "/home2/rkuang1/SNN_tr/pretrained_model/cifar10vgg9_timestep20_lr0.1_epoch120_leak0.95_quant4_nonvar_False_sram_False_ct_False_bit_rran_bestmodel_test.pth.tar" \
        --dataset "cifar10" \
        --batch_size 128 \
        --leak_mem 0.95 \
        --num_workers 4  \
        --gpu_id 1 \
        --quant 4 \
        --ct \
        --num_epochs 80 \
        --sram \
        --sram_quant \
        --tern \
        --shift \
        --log_name %s \
        ' %(i)) """           

""" log_list = ['q3_sram_tern_shf']
for i in log_list:
    os.system('CUDA_VISIBLE_DEVICES=2 python train.py \
        --num_steps 20 --lr 0.08 \
        --arch "vgg9" \
        --load_model "/home2/rkuang1/SNN_tr/pretrained_model/cifar10vgg9_timestep20_lr0.1_epoch120_leak0.95_quant3_nonvar_False_sram_False_ct_False_bit_rran_bestmodel_test.pth.tar" \
        --dataset "cifar10" \
        --batch_size 128 \
        --leak_mem 0.95 \
        --num_workers 4  \
        --gpu_id 2 \
        --quant 3 \
        --ct \
        --num_epochs 80 \
        --sram \
        --sram_quant \
        --tern \
        --shift \
        --log_name %s \
        ' %(i))  """    