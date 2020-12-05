---
title: meituan-ml
date: 2018-11-22 10:29:13
tags:
---

* 线上线下指标一致性：t-test ( student-test )
* 检验指标：AUC, logloss, TF/TN/FP/FN, acc, F1 score, F_alpha score, MAE, WMAE, RMSE, MAP, nDCG
* 采样：加快训练速度，skip-above；交叉验证：时间侧hold-out
* 为什么不用树模型：无法增量训练
* EDA：
    * 数值：峰值截断，二值化，分桶，缩放（log），缺失值处理（补值，直接喂缺失值）。
    * 类型：自然数，分层，hash，计数，rank特征， 点击率统计特征+贝叶斯平滑，直接交叉 vs 更细致的分桶交叉
    * 类型 x 数值：类型内计算数值
    * 时间特征，空间特征
    * 过滤：类内部差异小，类间差异大
