#!/bin/bash

#python train.py --dataroot ./datasets/idnet2sidtd1 --name idnet2sidtd1 --model cycle_gan --display_id -1
#python train.py --dataroot ./datasets/idnet2sidtd10 --name idnet2sidtd10 --model cycle_gan --display_id -1
#python train.py --dataroot ./datasets/idnet2sidtd20 --name idnet2sidtd20 --model cycle_gan --display_id -1
#python train.py --dataroot ./datasets/idnet2sidtd40 --name idnet2sidtd40 --model cycle_gan --display_id -1
#python train.py --dataroot ./datasets/idnet2sidtd100 --name idnet2sidtd100 --model cycle_gan --display_id -1
for i in {1..1}; do
    echo $i
    cd /scratch/luluxie/GAN/pytorch-CycleGAN-and-pix2pix
    rm -rf checkpoints/id*
    python train.py --dataroot ./datasets/idnet2sidtd1   --name idnet2sidtd1   --model cycle_gan --display_id -1 --n_epochs 5 --n_epochs_decay 1 
    python train.py --dataroot ./datasets/idnet2sidtd10  --name idnet2sidtd10  --model cycle_gan --display_id -1 --n_epochs 5 --n_epochs_decay 1
    python train.py --dataroot ./datasets/idnet2sidtd20  --name idnet2sidtd20  --model cycle_gan --display_id -1 --n_epochs 5 --n_epochs_decay 1

    cp checkpoints/iphone2dslr_flower_pretrained/latest_net_G.pth checkpoints/idnet2sidtd1/latest_net_G_A.pth
    python train.py --dataroot ./datasets/idnet2sidtd1   --name idnet2sidtd1   --model cycle_gan --display_id -1 --continue_train 
    cp checkpoints/iphone2dslr_flower_pretrained/latest_net_G.pth checkpoints/idnet2sidtd10/latest_net_G_A.pth
    python train.py --dataroot ./datasets/idnet2sidtd10  --name idnet2sidtd10  --model cycle_gan --display_id -1 --continue_train 
    cp checkpoints/iphone2dslr_flower_pretrained/latest_net_G.pth checkpoints/idnet2sidtd20/latest_net_G_A.pth
    python train.py --dataroot ./datasets/idnet2sidtd20  --name idnet2sidtd20  --model cycle_gan --display_id -1 --continue_train 

    rm -rf results/*
    cp checkpoints/idnet2sidtd1/latest_net_G_A.pth checkpoints/idnet2sidtd1/latest_net_G.pth
    python test.py --dataroot datasets/idnet2sidtdtest/testA --name idnet2sidtd1 --model test --no_dropout
    cp checkpoints/idnet2sidtd10/latest_net_G_A.pth checkpoints/idnet2sidtd10/latest_net_G.pth
    python test.py --dataroot datasets/idnet2sidtdtest/testA --name idnet2sidtd10 --model test --no_dropout
    cp checkpoints/idnet2sidtd20/latest_net_G_A.pth checkpoints/idnet2sidtd20/latest_net_G.pth
    python test.py --dataroot datasets/idnet2sidtdtest/testA --name idnet2sidtd20 --model test --no_dropout
    
    cd /home/luluxie/analysis/BO2
    python test_cyclegan.py idnet2sidtd1  >>logs/idnet2sidtd1.log
    python test_cyclegan.py idnet2sidtd10 >>logs/idnet2sidtd10.log
    python test_cyclegan.py idnet2sidtd20 >>logs/idnet2sidtd20.log
done
