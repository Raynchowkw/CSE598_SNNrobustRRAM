### 6/21/2022 ###
## -- vgg9 cifar100 tfs vat gap -- ##
dateset_l="cifar100"
q_list="4"
for ds in ${dateset_l}
do
    for q in ${q_list}
    do
        python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset ${ds} \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant ${q} \
            --no_var \
            --num_epochs 100 \
            --log_name "vgg9-${ds}_r${q}_baseline" \
            ;
        python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset ${ds} \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant ${q} \
            --num_epochs 100 \
            --log_name "vgg9-${ds}_r${q}_vat" ;
    done
done
exit 0
### 6/19/2022 ###
## -- tfs vat gap -- ##
dateset_l="cifar10 cifar100"
q_list="1 8"
for ds in ${dateset_l}
do
    for q in ${q_list}
    do
        python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg5" \
            --dataset ${ds} \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant ${q} \
            --num_epochs 100 \
            --log_name "vgg5-${ds}_r${q}_vat"
    done
done
exit 0
### 6/13/2022 ###
## %% 6/13/2022 ##
## %% load sram w/ same_bit rram: 82 84
python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 1 \
            --quant 4 \
            --sram \
            --sram_bit 1 \
            --shift \
            --load_rram_w \
            --shift_bit 0 \
            --ct \
            --load_rram "baseline/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant4_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 50 \
            --log_name "r4_s1_LDrr" \
            --model_name '0'
python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 1 \
            --quant 4 \
            --sram \
            --sram_bit 1 \
            --shift \
            --load_rram_w \
            --shift_bit 0 \
            --ct \
            --swp \
            --load_rram "baseline/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant4_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 50 \
            --log_name "r4_s1_LDrr_swp" \
            --model_name '0'
exit 0            
## ===  sram recovery with shifter === ##
qlist="1 2 3 4"
q3_path="save_model/modelsave_2022-06-12/2022-06-12cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_r3_var2.0_nonvar_False_sram_False1_ct_False_tern_False_shift0_rram_bestmodel.pth.tar"
q2_path="save_model/modelsave_2022-06-13/2022-06-13cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_r2_var2.0_nonvar_False_sram_False1_ct_False_tern_False_shift0_rram_bestmodel.pth.tar"
q1_path="save_model/modelsave_2022-06-12/2022-06-12cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_r1_var2.0_nonvar_False_sram_False1_ct_False_tern_False_shift0_rram_bestmodel.pth.tar"
q4_path="save_model/modelsave_2022-06-13/2022-06-13cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_r4_var2.0_nonvar_False_sram_False1_ct_False_tern_False_shift0_rram_bestmodel.pth.tar"
### *** test eval var for bash ***
# var=q${q}_path
# path=var
# q=1
# eval path=$(echo \$$var)
# echo $path 
# exit 0
## -- q1 --
# echo $q3_path
for q in ${qlist}
do 
# echo ${q}
var=q${q}_path
# path=var
eval path=$(echo \$$var)
# echo ${path}
python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant ${q} \
            --sram \
            --sram_bit 1 \
            --shift \
            --load_rram_w \
            --shift_bit 0 \
            --ct \
            --load_rram ${path}  \
            --num_epochs 50 \
            --log_name "r${q}s1_Shf0" \
            --model_name '0'
done;
sbits="2 3 4"
for sb in $sbits
do
q=4
var=q4_path
# path=var
eval path=$(echo \$$var)
# echo ${path}
python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant ${q} \
            --sram \
            --sram_bit $sb \
            --shift \
            --load_rram_w \
            --shift_bit 0 \
            --ct \
            --load_rram ${path}  \
            --num_epochs 50 \
            --log_name "r${q}s${sb}_Shf0" \
            --model_name '0'
done
exit 0

### 6/12/2022 ###
#### 6/12 H_var tfs [[eng]]###
## ===== vgg9 high variance tfs ===== ##
var_list="2 3 4"
q_list="1 2"
for var in ${var_list}
do
    for q in ${q_list}
    do
        python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant ${q} \
            --var_scale ${var} \
            --num_epochs 100 \
            --log_name "r${q}_tfs_var${var}_vg9"
    done
done
exit 0
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
            --load_rram "baseline/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant8_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 40 \
            --log_name "r8s2_LDr_swp" \
            --model_name '0'
exit 0 
## For load different RRAM bit to SRAM: r3s2LD1, r3s1LD2 -- compare with r3s1LX, r3s2LX
python train.py \
            --num_steps 20 --lr 0.05 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant 3 \
            --sram \
            --sram_bit 1 \
            --shift \
            --load_sram_w \
            --shift_bit 0 \
            --ct \
            --load_sram "baseline/2022-05-01cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant2_nonvar_False_sram_False_ct_False_tern_False_shift_False_rran_bestmodel.pth.tar" \
            --load_rram "baseline/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant3_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 50 \
            --log_name "r3s1_LDrr2_noSft"\
            --model_name '0'
python train.py \
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
            --load_sram_w \
            --shift_bit 0 \
            --ct \
            --load_sram "baseline/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant1_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar" \
            --load_rram "baseline/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant3_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 50 \
            --log_name "r3s2_LDrr1_noSft"\
            --model_name '0'

exit 0

#### 6/9 eva ###
quant_list="1 3 4 8"
for qt in ${quant_list}
do
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
done


exit 0
### 6/7/2022 ###
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
            --load_rram "baseline/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant8_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 50 \
            --log_name "r8_s4_LDrr" \
            --model_name '0'
exit 0   
### 6/6/2022 ###
# swp for q4
python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant 4 \
            --sram \
            --sram_bit 2 \
            --shift \
            --load_rram_w \
            --shift_bit 0 \
            --ct \
            --swp \
            --load_rram "baseline/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant4_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 50 \
            --log_name "r4s2_LDrr_swp"\
            --model_name '0'
exit 0            
# 
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
            --sram_bit 2 \
            --shift \
            --load_rram_w \
            --shift_bit 0 \
            --ct \
            --load_rram "baseline/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant8_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 50 \
            --log_name "r8s2_LDrr_noSft"\
            --model_name '0'
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
            --load_rram "baseline/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant8_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 50 \
            --log_name "r8s4_LDrr_noSft"\
            --model_name '0'
exit 0            
### 6/2/2022 ###
python train.py \
            --num_steps 20 --lr 0.05 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant 2 \
            --sram \
            --sram_bit 1 \
            --shift \
            --load_sram_w \
            --shift_bit 0 \
            --ct \
            --load_sram "baseline/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant1_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar" \
            --load_rram "baseline/2022-05-01cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant2_nonvar_False_sram_False_ct_False_tern_False_shift_False_rran_bestmodel.pth.tar"  \
            --num_epochs 40 \
            --swp \
            --log_name "r2s1_LDrr2_swp_lr05"\
            --model_name '0'

exit 0
### 5/31/2022 ###
#### loading scheme: load rram_bit to sram ####

# # r2 s2, LDrr2
# python train.py \
#             --num_steps 20 --lr 0.1 \
#             --arch "vgg9" \
#             --dataset "cifar10" \
#             --batch_size 128 \
#             --leak_mem 0.95 \
#             --num_workers 4  \
#             --gpu_id 0 \
#             --quant 2 \
#             --sram \
#             --sram_bit 2 \
#             --shift \
#             --load_rram_w \
#             --shift_bit 0 \
#             --ct \
#             --load_rram "vatPreTrained/2022-05-01cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant2_nonvar_False_sram_False_ct_False_tern_False_shift_False_rran_bestmodel.pth.tar"  \
#             --num_epochs 40 \
#             --swp \
#             --log_name "r2s2_LDrr_noSft_swpNP1"\
#             --model_name '0'

### 5/30/2022 ###
###### --- sram + rram for recovery --- #####
## --- s{}r{}sht{}LDr{} --- ###
python train.py \
            --num_steps 20 --lr 0.1 \
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
            --load_sram_w \
            --shift_bit 0 \
            --ct \
            --load_sram "vatPreTrained/2022-05-01cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant2_nonvar_False_sram_False_ct_False_tern_False_shift_False_rran_bestmodel.pth.tar"  \
            --load_rram "SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant3_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 60 \
            --log_name "r3s2_LDrr2_noSft"\
            --model_name '0'

exit 0
### 5/29/2022 ###
### ------- pruning ------- ###
######## r1s1
python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant 1 \
            --sram \
            --sram_bit 1 \
            --shift \
            --load_rram_w \
            --shift_bit 0 \
            --swp \
            --ratio 0.5\
            --ct \
            --load_sram "SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant1_nonvar_True_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --load_rram "SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant1_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 60 \
            --log_name "r1s1_LDrr_noSft_swp_rt.5"
exit 0
### 5/25/2022 ###
## ===== vgg9 high variance recover w/ w/o XNORBP ===== ##
python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant 4 \
            --sram \
            --sram_bit 3 \
            --shift \
            --load_rram_w \
            --var_scale 2 \
            --shift_bit 0 \
            --ct \
            --load_rram "VarScaPreTrained/2022-05-24cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_r4_varscale2_nonvar_False_sbit1_sram_False1_ct_False_tern_False_shift_False_srinit0_False_rran_bestmodel.pth.tar"  \
            --num_epochs 60 \
            --log_name "vg9-var2-r4s3_LDr_NSf"

exit 0
## ===== vgg9 high variance tfs ===== ##
python train.py \
    --num_steps 20 --lr 0.1 \
    --arch "vgg9" \
    --dataset "cifar10" \
    --batch_size 128 \
    --leak_mem 0.95 \
    --num_workers 4  \
    --gpu_id 0 \
    --quant 4 \
    --var_scale 2 \
    --num_epochs 100 \
    --log_name "r_tfs_var2_vg9_q4"
exit 0
## ===== vgg9 high variance tfs ===== ##

###################################

###### --- sram + rram for recovery --- #####
## --- s{}r{}sht{} --- ###
### preload sram_trained w
python train.py \
            --num_steps 20 --lr 0.1 \
            --arch "vgg9" \
            --dataset "cifar10" \
            --batch_size 128 \
            --leak_mem 0.95 \
            --num_workers 4  \
            --gpu_id 0 \
            --quant 3 \
            --sram \
            --sram_bit 3 \
            --shift \
            --shift_bit 0\
            --ct \
            --load_sram "SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant3_nonvar_True_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --load_rram "SNN_modelsave_2022-04-12_tfs/cifar10vgg9_BNTT_timestep20_lr0.1_epoch100_leak0.95_quant3_nonvar_False_sram_False_ct_False_tern_False_shift_False_bit_rran_bestmodel_test.pth.tar"  \
            --num_epochs 60 \
            --log_name "r3_s3_LDsr_sft0"
