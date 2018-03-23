---
title: hivemall
date: 2018-03-19 10:57:35
tags:
---

##### Hivemall是什么
* apache主页：http://hivemall.incubator.apache.org/index.html
* 包含: regression, classification, recommendation, anomaly detection, k-nearest neighbor, and feature engineering
* 包含ML: Soft Confidence Weighted, Adaptive Regularization of Weight Vectors, Factorization Machines, and AdaDelta
* Support: Hive, Spark, Pig
* Architecture
{% asset_img "hivemall_1.PNG" [hivemall_architecture] %}


##### Hive on Spark VS Hive on MR
* configure: set hive.execution.engine=spark;
* tutorial: https://cwiki.apache.org/confluence/display/Hive/Hive+on+Spark%3A+Getting+Started
* 测试:
```
select max(a) as a, max(b) as b
from
(
select 1 as a, 2 as b
union all
select 2 as a, 3 as b
union all
select 3 as a, 4 as b
) t
```

##### Hive on Spark with Hivemall
* hive check version: hive --version
* download Hivemall: https://hivemall.incubator.apache.org/download.html
* UDF define: https://github.com/apache/incubator-hivemall/blob/master/resources/ddl/define-all.hive
* add jar
```
add jaradd jar hdfs://footstone/data/project/dataming/contentrec/hivemall/hivemall-all-0.5.0-incubating.jar;
source define-all.hive;
```
* setting queue: set mapred.job.queue.name=root.dataming.dev;
* set hive nonstrict mode: set hive.mapred.mode=nonstrict;

##### AUC calc
* Single node
```
with data as (
  select 0.5 as prob, 0 as label
  union all
  select 0.3 as prob, 1 as label
  union all
  select 0.2 as prob, 0 as label
  union all
  select 0.8 as prob, 1 as label
  union all
  select 0.7 as prob, 1 as label
)
select
  auc(prob, label) as auc
from (
  select prob, label
  from data
  ORDER BY prob DESC
) t;

```

* Parallel approximate
```
with data as (
  select 0.5 as prob, 0 as label
  union all
  select 0.3 as prob, 1 as label
  union all
  select 0.2 as prob, 0 as label
  union all
  select 0.8 as prob, 1 as label
  union all
  select 0.7 as prob, 1 as label
)
select
  auc(prob, label) as auc
from (
  select prob, label
  from data
  DISTRIBUTE BY floor(prob / 0.2)
  SORT BY prob DESC
) t;
```

##### Compile from source
* git clone https://github.com/sampsonguo/incubator-hivemall.git
* mvn clean package -Dmaven.test.skip=true
* create temporary function beta_dist_sample as 'com.sigmoidguo.math.BetaUDF';

##### Hive & Json
hive解析json:
* json_split: brickhouse split array
* get_json_object: hive udf
