---
title: bayes
date: 2018-01-16 11:26:02
tags:
---

### Discriminative VS Generative
* Discriminative models 判别模型
直接建模P(c|x)
* Generative models 生成模型
P(c|x) = P(x, c) / P(x) = P(c)*P(x|c)/P(x)
其中，
	* P(c)可以通过统计各类样本比例频率来估计 --频率学派
	* P(x|c)因为样本x的数据量太小，很难估计准确
	
### 频率学派 VS 贝叶斯学派
* Frequentist 频率学派 
参数是一个未知但客观存在的固定值
* Bayesian 贝叶斯学派 
参数本身是一个分布 

### Naive Bayes 
* 属性条件独立性假设（假设每个属性独立地对分类结果发生影响）
* Smoothing: 拉普拉斯修正（Laplacian Correction）
* Lazy Learning

### semi-naive Bayes classifier
todo
