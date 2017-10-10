---
title: auc_n_logloss
date: 2017-10-06 15:31:48
tags:
categories: 机器学习
---

### 评估指标
评估指标大致分为两种，值评估和序评估。

<!-- more -->

| 值评估 | 序评估 |
|--- |---|
| MSE/RMSE | AUC/AUPR |
| R^2 | P@k/MAP/nDCG |
| logloss | Precision/Recall |
| MAE | TP/FP/TN/FN/F1 |

### 序准 VS 值准
* 指标和目的
    * 序评估目的是为了序准
    * 值评估目的是为了值准
* 应用场景
    * 序准适用于推荐系统，pCTR相对准确，目的是用户价值最大化
    * 值准适用于商业化系统，pCTR绝对准确，pCTR*cpc，目的是商业价值最大化
    * 综合公式 score=pCTR^a * cpc^b，进行调权重

### AUC的由来和计算
auc的一些基础知识，可以参考维基百科的解释:

https://en.wikipedia.org/wiki/Receiver_operating_characteristic

这里需要提到一些常见的错误：
* 错误1：auc是一条光滑曲线
auc是一条折线，如下图
 {% asset_img "1.gif" [1.gif] %}

* 错误2：auc是和预估值有关系的
auc只和序有关系，和值无关。

* 错误3：求auc需要画出roc曲线
auc计算部分，除了画出roc曲线，还可以直接计算：
 {% asset_img "2.png" [2.png] %}
其中,
M为正类样本的数目，N为负类样本的数目
rank是用的tiedrank

#### AUC的物理意义

和Wilcoxon-Mann-Witney Test有关，即:
auc=“测试任意给一个正类样本和一个负类样本，正类样本的score有多大的概率大于负类样本的score”，也即auc的物理意义。

#### AUC的计算
* spark
```
// Compute raw scores on the test set
val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

// Instantiate metrics object
val metrics = new BinaryClassificationMetrics(predictionAndLabels)

// AUROC
val auROC = metrics.areaUnderROC
println("Area under ROC = " + auROC)
```

* hivemall
 {% asset_img "3.png" [3.png] %}

* C语言
Ref: https://github.com/liuzhiqiangruc/dml/blob/master/regr/auc.c
```
double auc(int n, double *x, double *y) {
    if (!y || !x) return 0.0;
    double *rk = (double*) malloc(sizeof(double) * n);
    AucP *aucp = (AucP *)malloc(sizeof(AucP) * n);
    int i, tsum;
    double rksum, auc;
    for (i = 0; i < n; ++i) {
        aucp[i].x = x[i];
        aucp[i].id = i;
    }
    tiedrank(n, aucp, rk);
    for (rksum = 0., tsum = 0, i = 0; i < n; ++i) {
        if (y[i] >= 1. - 1e-10) {
            rksum += rk[i];
            tsum += 1;
        }
    }
    double mn, pst;
    mn = (double) (n - tsum);
    mn *= (double) tsum;
    pst = (double) tsum;
    pst *= (double) tsum + 1;
    auc = (rksum - pst / 2.) / mn;
    free(rk);
    free(aucp);
    return auc;
}
```

#### AUC的弊端和AUPR
 {% asset_img "4.png" [4.png] %}
 {% asset_img "6.png" [6.png] %}
 {% asset_img "7.png" [7.png] %}

### logloss的由来和计算

#### logloss由来
logloss是根据最大似然推导得到的，可参考：
http://www.csuldw.com/2016/03/26/2016-03-26-loss-function/

有些概念需要区分一下
* loss function: 样本粒度的函数，如logloss, hingeloss等。
引用一张名图：
{% asset_img "9.png" [9.png] %}

>Plot of various loss functions. Blue is the 0–1 indicator function. Green is the square loss function. Purple is the hinge loss function. Yellow is the logistic loss function. Note that all surrogates give a loss penalty of 1 for yf(x) = 0

* cost function: 集合粒度的函数，即 sum of loss.
 
#### logloss计算
logloss计算需要避免log0的情况，可以参考kaggle中的计算方式：
```
max(min(p,1−10^−15),10^−15)max(min(p,1−10^−15),10^−15).)
```
ref: https://www.kaggle.com/wiki/LogLoss

### AUC和logloss何时不一致
在样本不均衡的情况下，AUC和logloss会出现很大的偏差。
1. logloss低但是AUC也低
当负样本过多的时候，人为全部预测为负样本，可以实现低logloss，但是AUC=0.5，并不优秀。

2. AUC高但是logloss也高
负样本过多，当位置pCTR顺序不变，AUC不变，pCTR统一扩大到接近1时候，导致logloss会变得非常的高。

### 定向模式 VS 推荐模式
1. PUSH系统：广告为中心，为广告找用户，并push；展示可有可无。
2. 推荐系统：人为中心，为人找推荐项，并展示；用户来了必须展示。

### 线下AUC和线上不一致
有三种AUC，很多不一致是因为AUC的描述不同造成的
假设有user-item-pCTR矩阵，那么可以计算
* 横向AUC：每用户AUC，适用于推荐系统
* 纵向AUC：每广告AUC，适用于PUSH系统
* 全局AUC：统一大模型的AUC

存在很多种情况：
* 纵向AUC高，横向AUC不一定高
单广告训练做推荐的典型的问题，举一个例子

|  | Item1 | Item2 | Item3 | Item4 |
| --- | ---| --- | --- | --- |
| UserA | 0.7(0) | 0.7(1) | 0.7(1) | 0.7(1) |
| UserB | 0.6(0) | 0.6(0) | 0.6(1) | 0.6(1) |
| UserC | 0.5(0) | 0.5(0) | 0.5(0) | 0.5(1) |

从纵向来看，每个单Item模型的AUC=1.0，但是横向的AUC=0.5，因此纵向AUC高，并不代表横向AUC高。即：
从单广告训练的AUC，集合起来，变成真正用户X过来，对用户X进行广告排序，AUC不一定高。
这种不一致是由于基于Item的模型并没有发现用户的对比其他人“更”偏好什么。

* 横向AUC高，纵向AUC不一定高
上图翻转，同理。

### AUC@topk VS AUC人数加权
为了和线上的情况保持一致，最好的方式是：
* 用户来了必须展示，因此AUC的计算方式是横向AUC，即每个用户计算AUC，然后加权；
* 用户往往只看头部，因此只计算AUC@topk，衡量头部排序能力。


