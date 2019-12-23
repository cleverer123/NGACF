#!/bin/sh
# python run_Pair.py --model GACFMask --dataset ml100k --lr 0.003 --weight_decay 0.0000005 --parallel True --epochs 200 --droprate 0.2 --adj_type mean_adj

# python run_Pair.py --model NGCFMF --dataset ml1m --lr 0.001 --weight_decay 0.00001 --parallel True
#1 # python run_Pair.py --model GACFV2 --dataset ml1m --lr 0.001 --weight_decay 0.000001 --parallel True 
# python run_Pair.py --model GACFV2 --dataset ml1m --lr 0.001 --weight_decay 0.000001 --parallel True --embedSize 64 --layers [64,64,64]
# #1 python run_Pair.py --model GACFV2 --dataset ml1m --lr 0.001 --weight_decay 0.000001 --parallel True --embedSize 128 --layers [128,128]
# python run_Pair.py --model GACFV2 --dataset ml1m --lr 0.0005 --weight_decay 0.000001 --parallel True --embedSize 128 --layers [128,128]

# python run_Pair.py --model GACFV2 --dataset ml1m --lr 0.001 --weight_decay 0.000001 --parallel True --epochs 200 --droprate 0.0 --adj_type mean_adj
# python run_Pair.py --model GACFV2 --dataset ml1m --lr 0.002 --weight_decay 0.000001 --parallel True --epochs 100 --droprate 0.2 --adj_type mean_adj
# python run_Pair.py --model GACFV2 --dataset ml1m --lr 0.002 --weight_decay 0.000001 --parallel True --epochs 50 --droprate 0.4 --adj_type mean_adj
# python run_Pair.py --model GACFV2 --dataset ml1m --lr 0.001 --weight_decay 0.000001 --parallel True --epochs 100 --droprate 0.2 --adj_type mean_adj


# python run_Pair.py --model GACFV2 --dataset ml1m --lr 0.002 --weight_decay 0.000001 --parallel True --epochs 200 --droprate 0.2 --adj_type mean_adj
# python run_Pair.py --model GACFMask --dataset ml1m --lr 0.002 --weight_decay 0.000001 --parallel True --epochs 200 --droprate 0.2 --adj_type mean_adj
# 
# python run_Pair.py --model GACFV2 --dataset ml1m --lr 0.001 --weight_decay 0.000001 --parallel True --epochs 200 --droprate 0.2 --adj_type mean_adj --embedSize 128 --layers [128,128]

# python run_Mask.py --model SPGACF --dataset ml100k --lr 0.001 --weight_decay 0.000001 --parallel False --epochs 100 --droprate 0.2 --adj_type mean_adj


# python run_Mask.py --model SPGAMGP --dataset ml100k --lr 0.002 --weight_decay 0.000001 --parallel False --epochs 200 --droprate 0.2 --adj_type mean_adj --layers [64,64]
python run_Mask.py --model SPGAMGP --dataset ml1m --lr 0.002 --weight_decay 0.000001 --parallel False --epochs 200 --droprate 0.2 --adj_type mean_adj --layers [64]
