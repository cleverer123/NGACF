# GACF Model Details

## GACFV1: 在原文的图卷积层之前加入attention。

## GACFV2: 只在原始特征部分加入attention，去掉interactive Element-wise Product.

## *GACFV3*: 在GACFV1的基础上，去掉interactive部分的Element-wise Product. attention作用于原始feature和interative feature两部分。

## GACFV4: 在GACFV2的基础上，去掉attention作用于原始feature的部分，attention 只作用于interaction部分，结果做Element-wise乘积，形成interactive feature，然后Propagate

## *GACFV5*：在GACFV4的基础上，attention 只作用于interaction部分，并且去掉內积，attention输出的feature直接Propagate。

## GACFV6: 在GACFV5的基础上， attention 作用于interaction部分的结果，attention结果输入全连接。即不经过拉普拉斯Propagate。此方法不太合理。

