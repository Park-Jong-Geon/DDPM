#!/bin/bash

for steps in 769 768 767 766 765 764 763 762 761
do    
    python3 ddpm.py --mode sample --dataset cifar10 --sample_num 10000 --checkpoint newsave/230326_no_ema_copy --sample_dir newsample/noema_"$steps"k_eval --lr 2e-4 --random_seed 156800
    fidelity --fid --input1 cifar10-train --input2 newsample/noema_"$steps"k_eval >> fid.txt
    rm -f newsave/230326_no_ema_copy/checkpoint_"$steps"000
done

# for seed in 459200 438750
# do
#     python3 ddpm.py --mode sample --dataset cifar10 --sample_num 20000 --checkpoint newsave/trials --sample_dir newsample/230326_769k_eval --lr 2e-4 --random_seed "$seed"
# done