---
title: MachineLearning_ZhouZhihua
date: 2017-11-27 11:10:36
tags:
---

### chapter 1

* No Free Lunch Theorem(NFL)

算法应用于问题要具体问题具体分析

### chapture 2: 模型评估和选择

#### Basic Concepts
* error
	* empirical error/training error 经验误差（训练误差）
	* generalization error 泛化误差（预测集误差）
* fitting
	* overfitting
	* underfitting
* set
	* train / dev / test set
* cross validation
	* k-fold
* bootstrapping
	* 适用小数据集
	* 会改变数据分布
* parameter tuning
* performance measure
	* MSE: mean squared error 
	* error rate
	* accuracy rate
	* Precision/Recall/F1/ROC/AUC
	
#### 常见错误
* 训练集测试集分布不一致(采样？Group By?)
* 数据穿越

#### 代码敏感错误率与代价曲线
* 代价曲线与期望总体代价
* 假设检验

#### 多分类
* Linear Regression/Logistic Regression/LDA(Linear Discriminant Analysis)
* OVO, OVR

#### DT
* entropy/IG/Gini
* 预减枝/后减枝

#### ANN
* Perceptron

