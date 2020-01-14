# NGACF
Implemention of 'Neural Graph Attention Collaborative Filtering'

## 环境配置

```conda create -n env python=3.6```

```source activate env```

```pip install -r requirements.txt```

## 实验
###  小数据集测试代码是否通过

```python run_Gowalla.py --parallel False --gpu_id 0 --model SPUIGACF --dataset ml100k --lr 0.002 --weight_decay 0.000001 --epochs 2 --droprate 0.2 --adj_type ui_mat --train_mode PairSampling --eval_mode AllNeg --eval_every 1```

### 实验1：Yelp 实验

内存占用约100G，显存占用12G。

```python run_Gowalla.py --parallel False --gpu_id 0 --model SPUIGACF --dataset Yelp --lr 0.01 --weight_decay 0.000001 --epochs 100 --droprate 0.2 --adj_type ui_mat --train_mode PairSampling --eval_mode AllNeg --eval_every 50```

### 实验2：Gowalla 实验

内存占用约60G，显存占用9G。

```python run_Gowalla.py --parallel False --gpu_id 0 --model SPUIGACF --dataset Gowalla --lr 0.01 --weight_decay 0.000001 --epochs 100 --droprate 0.2 --adj_type ui_mat --train_mode PairSampling --eval_mode AllNeg --eval_every 50```



