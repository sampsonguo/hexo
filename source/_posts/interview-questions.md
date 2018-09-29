---
title: interview_questions
date: 2018-08-05 16:19:00
tags:
---


近期需要进行校招生内推面试，记录一下面试要点。

### 流程
内推面试时长30min，流程上是：
自我介绍（5min） -> 项目介绍+基础知识测试（10min） -> 思考能力测试（15min）

### 自我介绍
自我介绍环节，重点了解候选人的特点，有以下几种：
* 学霸：名校毕业，专业排名靠前，重点考察基础知识和学习能力
* 名企实习：在BAT等有过实习经历，重点考察实习项目和思考
* 热衷比赛：kaggle比赛，ACM比赛等有拿奖或排名靠前，重点考察对应比赛的经验并相应出题

### 项目介绍
* 适当打断，问为什么这么做，问实现的细节，看应变。

### 基础知识测试
在候选人介绍项目的时候，询问项目细节，适当的提出问题：
* 机器学习
  * 基础知识1：过拟合和欠拟合是什么？对应的模型如何解决？
    * 过拟合：训练误差小，预测误差大
    * 欠拟合：训练和预测误差都很大

| 过拟合 | 欠拟合 |
| ------------- | ------------- |
| 增加训练数据量 | -  |
| 做特征筛选 | 收集更多特征 |
| 用更简单模型 | 用复杂模型 |
| 增大模型正则 | 减少模型正则 |
| DT/RF/GBDT 减小树深度 | DT/RF/GBDT 增大树深度 |
| NN减少深/宽度，early stop，增加drop out | NN增加深/宽度，增大epoch |

  * 基础知识2：特征共线性是什么？如何解决？
    * 特征共线性是变量之间存在相关性，导致模型不稳定，泛化误差大。
    * 1. wrapper贪心添加特征，做特征筛选  2.增加正则
  * 基础知识3：lasso regression和ridge regression的区别是什么？
  * 模型评估1： precision和accuracy的区别是什么？
    * precision = TP/(TP+FP)
    * accuracy = (TP+TN)/(TP+TN+FP+FN)
  * 模型评估2： F1 和 AUC的区别是什么？
    * F1 是一个准召权衡的一个点，AUC是roc曲线下面积，F1上阈值移动过程中AUC上的一个点。
  * 模型评估3： AUC高但是logloss太大，是为什么？反过来，又是为什么？
    * AUC高但logloss大：序准值不准，可能是采样导致的，进行后处理校准。
    * AUC低但是logloss小：值准序不准，模型学到了平均值，欠拟合。
  * 树模型：比较下RF和GBDT
    * bagging vs boosting, 树越多越不过拟合 vs 越过拟合，训练快预测慢 vs 训练慢预测快，并行 vs 串行
  * 模型融合：模型融合有哪几种方法？
    * bagging（RF），boosting（GBDT），stacking
  * FM：FM的基本原理是什么？如何做embedding？
* 深度学习
  * 介绍下梯度消失和梯度爆炸的概念，产生原因和如何解决？
    * 梯度消失：sigmoid函数在链式求导的时候，梯度是不可能超过0.25的，同时w也比较小，因此层次越深梯度越来越小。解决方案：relu, batch normalization
    * 梯度爆炸：在网络比较深，权重初始值大的情况下，如果梯度大于1，链式求导很容易爆炸。解决方案：正则，lstm

### 基础知识
根据特点，选择基础知识题目。
熟悉数据库 -> Hive题目
数学好 -> 概率题
编程好 -> ACM题

* Hive（SQL）
有一张表t_action
字段: user_id, item_id, action(1=click, 2=buy), timestamp
要求：求购买次数最多的top10用户最早的购买时间，按照购买次数排序
```
select f2.user_id, f2.timestamp
from
(
  select user_id, row_number() over (order by buy_cnt desc) as rnk
  from
  (
    select user_id, count(1) as buy_cnt
    from t_action
    where action = 2
    group by user_id
  ) t
  where rnk <= 10
) f1
join
(
  select user_id, min(timestamp) as timestamp
  from t_action
  where action = 2
  group by user_id
) f2 on f1.user_id = f2.user_id
order by f1.rnk asc
```
  * 拓展：如果用户量巨大，order by全局排序时间过久怎么办？
  * 答案：分多个map来选出各map内的top10，在reduce里再汇总总排名top10.
* 概率题目1：token反拿问题
  * 题目：某员工使用token登陆系统（配token图），不小心把token拿反了，但是输入里面的数字竟然成功登陆，问此情况的概率是多少？
  * 数字对应：1-1,2-2,5-5,6-9,8-8,9-6,0-0
    总情况：10^6
    正反相等情况：7^3
    正确答案：7^3/10^6
* 概率题目2：最少晋级分数问题
  * 题目：有8支球队打循环赛，打C（3,2）场比赛，赢3分，平1分，负0分，4支球队可以晋级，问最少多少分可以保证一定出线？
  * 5支球队得分相同，3支球队没有得分，此种情况下是未晋级的最大分值；
    5支球队内部每支球队都2胜2负，此种情况得分大于4场全平，即（2×3>4）；
    未晋级的最大分值是3×3+ 2×3=15分，因此晋级的最少的分是16分。
* 概率题目3：三门问题
* 概率题目4：有偏好生育问题
* ACM题目1：二分查找
* ACM题目2：两颗树合并
* ACM题目3：最长公共子序列 vs 最长公共子串


