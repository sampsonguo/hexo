---
title: ctr_smooth
date: 2017-12-07 23:05:56
tags:
---

#### 最好的非个性化模型
最好的非个性化模型，即CTR倒序模型，那如何得到准确的统计CTR，是本文的关键。

#### 统计的小数效应（置信度和pCTR）
风控模型中，有两个值非常重要：
* confidence：用户的数据有多可靠，比如交易记录越多，越可靠
* score: 用户的资产有多少，比如交易额度越大，资产越多

两个case：
* 高confidence低score：卖矿泉水的小商贩，转账频繁，大多都是一两块钱
* 低confidence高score：偶尔用信用卡买了一辆车的大老板

同样的在统计CTR中，也有对应的两个概念：
* CTR可信不可信
* CTR是多少

同样的两个case：
* itemA：10次曝光5次点击，可能受到随机影响，所以confidence低，pCTR高（随机影响也可能导致pCTR偏低）
* itemB：10000次曝光1000次点击，大数效应，比较可信，confidence高，pCTR低

#### 三个变量的权衡
为了得到真实的CTR，可以从日志中统计得到：
exposure, click, ctr
三个变量

item | exposure | click | ctr
---|---|---|---
A | 100000 | 20000 | 0.2
B | 10000 | 1000 | 0.1
C | 10 | 5 | 0.5

那么哪款item最优先级被推荐？

理想情况下是：
* exposure和click越高越高，confidence越大
* ctr越高越好，score越大

但是当两者矛盾的时候，就需要平衡一下，综合来看A是最佳的。

#### 简单暴力的bayes平滑
根据贝叶斯有：先验+事件=后验，那么我们为模型增加人为的知识：

"所有样本，统一增加b个样本（其中a个正样本)"

a和b的相对值的确定，可以用a/b等于整体平均ctr*rate等来确定，rate常常略微小于1 

a和b的绝对值的确定，很有意思：

我常常用excel对优质数据进行标注，看如何设置可以使得优质数据上浮顶部。

#### 贝叶斯平滑物理意义 和 极大后验
贝叶斯平滑，相当于增加先验，即增加正则，先举一个L2正则的例子：
线性回归的loss function
（PS：loss function和cost function是不一样的，cost function是loss function在data上的累计总和）

loss function: (y - f(x))^2
maximum likelihood: guass_function(y-f(x))

L2正则相当于对参数的分布增加一个属于高斯分布的假设，
guass_function(theta)*guass_function(y-f(x))

将这个似然最大化，
argmax(guass_function(theta)*guass_function(y-f(x)))
=> argmax(log(guass_function(theta)*guass_function(y-f(x))))
=> argmax(log(guass_function(theta))+log(guass_function(y-f(x))))
=> argmin((theta)^2+(y_f(x))^2)

即极大后验。

因此，从参数估计的角度，贝叶斯平滑是将极大似然估计（直接除）变成极大后验估计（分子分母各加一个值）


#### REF
1. http://myslide.cn/slides/977
2. http://www.jianshu.com/p/a47c46153326