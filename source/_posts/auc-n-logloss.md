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
logloss是根据


### AUC和logloss何时不一致

### 定向模式 VS 推荐模式

### 线下AUC和线上不一致

### AUC@topk VS AUC人数加权

