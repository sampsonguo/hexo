---
title: ctr_recalibration
date: 2018-08-27 20:21:40
tags: ML
---

### CTR为什么会不准？
在计算广告中，pCTR往往对比真实的CTR偏高或者偏低的现象，尤其在
1. 热门曝光广告和冷门曝光广告之间
2. 高CTR广告和低CTR广告之间

因此，CTR需要校准。

<!-- more -->

### CTR为什么要校准？
AUC体现序准；
logloss体现值准；
要计算商业价值最大化，因此需要值准；
因此需要校准；
校准之后，logloss会降低。

### CTR如何校准
CTR校准有很多方法，本质在于“拟合校准前和校准后”，即
f(pCTR校准前) = pCTR校准后
如何设计函数f，是校准的关键。

#### binning
binning就是样本等频分桶后，每个bin求平均，如下图：
 {% asset_img "图1.1.png" [图1.1.png] %}

#### Isotonic regression(保序回归）
保序回归，就是单调回归（保证按照自变量x和按照因变量y排序序不变，即成正比）
为何要保序？
为了保证不影响AUC，即默认原始CTR和校准后CTR的正相关性。
{% asset_img "图2.2.png" [图2.2.png] %}

### Best practice
#### 分解动作
* 将统计ctr加入特征中（最好做离散化处理）
* 建立f(pCTR)=统计CTR的函数
* 进行将f(pCTR)作为新的CTR
#### 小demo
假设训练数据集合为：
物品3：pCTR_统计=0.8
物品2：pCTR_统计=0.5
物品1：pCTR_统计=0.3

##### 原始LR
```
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
X = [
[0.8],
[0.8],
[0.8],
[0.8],
[0.8],
[0.8],
[0.8],
[0.8],
[0.8],
[0.8],
[0.5],
[0.5],
[0.5],
[0.5],
[0.5],
[0.5],
[0.5],
[0.5],
[0.5],
[0.5],
[0.3],
[0.3],
[0.3],
[0.3],
[0.3],
[0.3],
[0.3],
[0.3],
[0.3],
[0.3]
]
y = [
1,
1,
1,
1,
1,
1,
1,
1,
0,
0,
1,
1,
1,
1,
1,
0,
0,
0,
0,
0,
1,
1,
1,
0,
0,
0,
0,
0,
0,
0]
LR = LogisticRegression()
LR.fit(X, y)
y_p = LR.predict_proba(X)
score = log_loss(y, y_p)
```

##### LR+保序回归
```
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
X = [
[0.8],
[0.8],
[0.8],
[0.8],
[0.8],
[0.8],
[0.8],
[0.8],
[0.8],
[0.8],
[0.5],
[0.5],
[0.5],
[0.5],
[0.5],
[0.5],
[0.5],
[0.5],
[0.5],
[0.5],
[0.3],
[0.3],
[0.3],
[0.3],
[0.3],
[0.3],
[0.3],
[0.3],
[0.3],
[0.3]
]
y = [
1,
1,
1,
1,
1,
1,
1,
1,
0,
0,
1,
1,
1,
1,
1,
0,
0,
0,
0,
0,
1,
1,
1,
0,
0,
0,
0,
0,
0,
0]
LR = LogisticRegression()
LR.fit(X, y)
y_lr = LR.predict_proba(X)
ir = IsotonicRegression()
y_ir = ir.fit_transform(map(lambda x:x[1], y_lr), map(lambda x:x[0], X))
score = log_loss(y, y_ir)
```

##### itemID离散化LR
```
X = [
[0,0,1],
[0,0,1],
[0,0,1],
[0,0,1],
[0,0,1],
[0,0,1],
[0,0,1],
[0,0,1],
[0,0,1],
[0,0,1],
[0,1,0],
[0,1,0],
[0,1,0],
[0,1,0],
[0,1,0],
[0,1,0],
[0,1,0],
[0,1,0],
[0,1,0],
[0,1,0],
[1,0,0],
[1,0,0],
[1,0,0],
[1,0,0],
[1,0,0],
[1,0,0],
[1,0,0],
[1,0,0],
[1,0,0],
[1,0,0],
]
y = [
1,
1,
1,
1,
1,
1,
1,
1,
0,
0,
1,
1,
1,
1,
1,
0,
0,
0,
0,
0,
1,
1,
1,
0,
0,
0,
0,
0,
0,
0]
LR = LogisticRegression()
LR.fit(X, y)
y_p = LR.predict_proba(X)
score = log_loss(y, y_p)
```

### 采样的校准
由于负样本抽样后，会造成点击率偏高的假象，需要将预测值还原成真实的值。调整的公式如下：

#### 为什么要采样
1. 加快训练速度
2. 某些模型需要正负样本比例不要太不均匀，例如nn要每个batch里有正样本

#### 结论
{% asset_img "图4.4.png" [图4.4.png] %}

#### 推导1
* 从结果侧推导
{% asset_img "图3.3.png" [图3.3.png] %}

#### 推导2
* 证明采样之后只需要加一个bias即可校正
* 全量集合N，正样本a，负样本b，采样比例k，扩大比例为1/k
* p = a / (a + b) = 1 / (1+e^-wx)
* => b/a = e^-wx
* p' = a / (a + b/k) = 1 / (1+e^-wx-lnk) = 1 / e^-(wx+lnk)


##### 举一个例子
负样本采样10%
校准回去 q = p / (10 - 9p)
p: 模型预估的pCTR
q: 真正要输出的ctr


REF:
https://tech.meituan.com/mt_dsp.html
http://blog.csdn.net/lming_08/article/details/40214921

