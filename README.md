# BNTT-Batch-Normalization-Through-Time + SRC-sram recovery

This repository contains the source code associated with [arXiv preprint arXiv:2010.01729][arXiv preprint arXiv:2010.01729]

[arXiv preprint arXiv:2010.01729]: https://arxiv.org/abs/2010.01729

## Introduction

Spiking Neural Networks (SNNs) have recently emerged as an alternative to deep learning owing to sparse, asynchronous and binary event (or spike) driven processing, that can yield huge energy efficiency benefits on neuromorphic hardware. However, training high-accuracy and low-latency SNNs from scratch suffers from non-differentiable nature of a spiking neuron. To address this training issue in SNNs, we revisit batch normalization and propose a temporal Batch Normalization Through Time (BNTT) technique. Most prior SNN works till now have disregarded batch normalization deeming it ineffective for training temporal SNNs. Different from previous works, our proposed BNTT decouples the parameters in a BNTT layer along the time axis to capture the temporal dynamics of spikes. The temporally evolving learnable parameters in BNTT allow a neuron to control its spike rate through different time-steps, enabling low-latency and low-energy training from scratch. We conduct experiments on CIFAR-10, CIFAR-100, Tiny-ImageNet and event-driven DVS-CIFAR10 datasets. BNTT allows us to train deep SNN architectures from scratch, for the first time, on complex datasets with just few 25-30 time-steps. We also propose an early exit algorithm using the distribution of parameters in BNTT to reduce the latency at inference, that further improves the energy-efficiency.


## Prerequisites
* Ubuntu 20.04    
* Python 3.7.11   
* PyTorch 1.10.2 
* NVIDIA GPU (>= 12GB)        

## Getting Started

### Installation
* Configure virtual (anaconda) environment
```
conda create -n env_name python=3.7
conda activate env_name
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

```


## Training

We provide VGG5/VGG9/VGG11/VGG19 architectures on CIFAR10/CIAR100 datasets
* ```train.py```: code for training with quantization and pruning 
* ```model.py```: code for VGG Spiking Neural Networks with BNTT  
* ```utill.py```: code for accuracy calculation / learning rate scheduler

Note that the training code will also do the testing for test set after each epoch.

### training step
1. baseline train from scratch
2. vat train from scracth
3. sram recovery fine tuning from 2. 
4. sram recovery with pruning fine tuning from 2. (not from 3. so as to save sram from the start)

### hyperparameters setting
#### baseline and vat training from scratch
* `time_step` = 20 suits VGG5 and VGG9, `time_step` = 30 suits VGG11 and VGG19. VGG19 may need more time_steps
* `leak_mem` should be fixed to 0.95
* `Epoches` >= 100 for “train from scratch”, >= 40 for “fine tuning” (sram recovery and sram pruning)
* `quant`: rram bit width
* `ct`: continued training or fine tuning
* `no_var`: no rram variance added
* `eva`: only inference
* `batch_size` can be lower if your gpu is not big enough
* `'var_scale`: variance scaler for vat or evaluation

#### sram recovery + bit shifting
* `sram_bit`: sram bit width
* `shift_bit`: shift sram bit position (left +; right -)
* `load_rram`: pretrained rram (vat) model to load to sram for fine tuning or rram for evaluation
* `load_sram`: pretrained sram (baseline) model to load to sram for fine tuning 

#### sram pruning
* `swp`: sram's weight pruning
* `th` : threshold for pruning -- max weight * th
* `lamda` : coefficient for regularization term
* `ratio`: min penalty ratio for intra group lasso
* `noprune1`: default not to prune 1st layer, can set False

#### others
* `tern`: ternary quantization
* `dorefa`: dorefa quantization **(haven't been implemented here)**

### arguments of interest
`log_name` is the name of output log file.
`model_name` is **not** name of the saved checkpoint. If it's empty, it will use default model names according to hyper parameter setting. If not, it will use the same name as log file.

### Running command

Run the following command for VGG9 SNN on CIFAR10.

* Run the baseline (train from scratch) (eg. quant to 1 bit)
no_var means it doesn't add rram's noise
```
python train.py \
    --num_steps 20 --lr 0.1 \
    --arch "vgg9" \
    --dataset "cifar10" \
    --batch_size 128 \
    --leak_mem 0.95 \
    --num_workers 4  \
    --gpu_id 0 \
    --no_var \
    --quant 1 \
    --num_epochs 100 \
    --log_name "r1_tfs"
```
* Run the vat
```
python train.py \
    --num_steps 20 --lr 0.1 \
    --arch "vgg9" \
    --dataset "cifar10" \
    --batch_size 128 \
    --leak_mem 0.95 \
    --num_workers 4  \
    --gpu_id 0 \
    --quant 1 \
    --num_epochs 100 \
    --log_name "r1_tfs_vat"
```
* Run the sram recover

Need to change quant [x] for rram bit, sram_bit [x] for sram bit, shift_bit [x] for shift left or right (left is + and right is -), load_rram [x] for vat trained model for the same rram bit
```
python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant 8 \
            --sram \
            --sram_bit 4 \
            --shift \
            --load_rram_w \
            --shift_bit 0 \
            --ct \
            --load_rram "vat/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant8_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 50 \
            --log_name "r8s4_LDrr_noSft"\
            --model_name '0'
```
* Run the sram pruning
```
python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 3 \
            --quant 8 \
            --sram \
            --sram_bit 2 \
            --shift \
            --load_rram_w \
            --shift_bit 0 \
            --ct \
            --swp \
            --load_rram "vat/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant8_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 40 \
            --log_name "r8s2_LDr_swp" \
            --model_name '0'
```

## Testing for a checkpoint
`eva` is for testing.
`load_rram` is the checkpoint you want to check.
```
python train.py \
        --num_steps 20 \
        --arch "vgg9" \
        --dataset "cifar10" \
        --batch_size 128 \
        --leak_mem 0.95 \
        --num_workers 4  \
        --gpu_id 0 \
        --quant ${qt} \
        --eva \
        --load_rram "baseline/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant${qt}_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
        --log_name "q${qt}vat_eva.log"
```


## Citation
Please consider citing ANN sram recovery paper:
```
To be published
```
[comment]:<A paper is under publishment in 2022>

Please consider citing BNTT's paper:
 ```
 @article{kim2020revisiting,
  title={Revisiting Batch Normalization for Training Low-latency Deep Spiking Neural Networks from Scratch},
  author={Kim, Youngeun and Panda, Priyadarshini},
  journal={arXiv preprint arXiv:2010.01729},
  year={2020}
}
 ```


### file organization
1. baseline models:
`checkpoint/SNN_final/baseline/`

1. vat models:
`checkpoint/SNN_final/vat/`

1. 2*var vat models:
`checkpoint/SNN_final/varvat/`

1. sram_recovered models:
named similar to 
"r(x)s(x)(\_LDrr)(\_VAR2)\_sram\_bestmodel.pth.tar"

1. sram_recovered with pruning models:
similar to sram_recovered models, but with 'swp' in the name.

1. Other running command examples: `*.sh`

1. BN layers' adaptation after fixing weights: `test_yale_recover.ipynb` 

