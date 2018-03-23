---
title: lda
date: 2017-10-05 14:36:05
tags:
categories: 数据挖掘
---

#### 理论
* **痛点**<br>
“乔布斯离我们而去了” 和 “苹果什么时候降价”如何关联？

<!-- more -->

* **思路**
  * 将word映射到topic维度<br>
  {% asset_img "1.png" [图片1] %}
  * 概率表示<br>
  {% asset_img "2.png" [图片2] %}
  * 概率表示<br>
  {% asset_img "3.png" [图片3] %}
* **演进：Unigram Model**<br>
  {% asset_img "4.png" [图片4] %}
* **演进：Bayes Unigram Model**<br>
  {% asset_img "5.png" [图片5] %}
* **演进：PLSA**<br>
  {% asset_img "6.png" [图片6] %}
  {% asset_img "7.png" [图片7] %}
* **演进：LDA**<br>
  {% asset_img "8.png" [图片8] %}
  {% asset_img "9.png" [图片9] %}
* **参数估计：统计**<br>
  {% asset_img "100.png" [图片9] %}
* **参数估计：似然**<br>
  {% asset_img "101.png" [图片9] %}
* **参数估计：后验**<br>
  {% asset_img "102.png" [图片9] %}
* **参数估计：贝叶斯**<br>
  {% asset_img "103.png" [图片9] %}
* **参数估计：对比**<br>
  {% asset_img "104.png" [图片9] %}
* **马尔可夫链条**<br>
  {% asset_img "105.png" [图片9] %}
* **吉布斯采样**<br>
  {% asset_img "106.png" [图片9] %}
* **实现代码**<br>
  {% asset_img "201.png" [图片9] %}
* **Ref:**<br>
  * Parameter estimation for text analysis （http://www.arbylon.net/publications/text-est.pdf）
  * LDA数学八卦
  * LDA简介 http://blog.csdn.net/huagong_adu/article/details/7937616
  * Gibbs采样 https://www.youtube.com/watch?v=a_08GKWHFWo

#### 实践
* 基础数据
  * 豌豆荚软件的描述信息
  * 星级>3星
  * 下载数>100
  * 安装数>100
  * 用户数>100
* 目的
  * 得到基于内容（描述）的item2item
  * 得到“词--主题--包名” 的关系
* 代码
  * [lda_code](../NLP/LDA原理和实践/README.md)


* LDA工具<br>
  https://github.com/liuzhiqiangruc/dml/tree/master/tm
* 获取数据<br>
```
hive -e "
select a.user_id, a.item_id, a.preference
from
(
   ...
)
" > input_lda
```

* 数据概况
  * 基础数据获取：见hql
  * 数据整理：cat input_lda | awk -F"\t" '{ print $1"\t"$2 }' > input
  * 数据形式：user_id \t item_id （后期可考虑tf-idf优化）
  * 行数：1849296
  * 用户数：678588
  * 游戏数：3377
* 运行命令
```
./lda -a 0.2 -b 0.01 -k 50 -n 1000 -s 100 -d ./input -o ./output

    参数说明:
     --------------------------------------------
           -t               算法类型1:基本lda，2:lda-collective，3:lda_time
           -r               运行模式，1:建模，2:burn-in
           -a               p(z|d) 的 Dirichlet 参数
           -b               p(w|z) 的 Dirichlet 参数
           -k               Topic个数
           -n               迭代次数
           -s               每多少次迭代输出一次结果
           -d               输入数据
           -o               输出文件目录,实现需要存在

  运行时长：10分钟左右
```
* 关联名称<br>
  * 处理word_topic矩阵，将ID和名称关联起来<br>

```
Hql如下，
set hive.exec.compress.output=false;
create table xxxx
(
    id  int
) row format delimited
fields terminated by '\t';

load data local inpath '/output/f_word_topic' OVERWRITE  into table xxxx;
```

* Item2Item计算<br>

```
mport sys
import math
import heapq

items_D = {} ## key: id

def load_data():
    global items_D
    inFp = open("lda_norm_10.csv", 'r')
    while True:
        line = inFp.readline()
        if not line:
            break
        items = line.strip().split(',')
        if len(items) != 54:
            continue
        item_D = {}
        item_D['soft_package_name'] = items[0]
        item_D['name'] = items[1]
        item_D['id'] = int(items[2])
        item_D['topics'] = map(float, items[3:53])
        item_D['sum'] = float(items[53])
        items_D[item_D['id']] = item_D


def dis1(A, B):
    return sum( A['topics'][i] * B['topics'][i] for i in range(50))

def dis2(A, B):
    return sum( 100 - abs(A['topics'][i] - B['topics'][i]) for i in range(50))

def search_similar():
    while True:
        line = sys.stdin.readline()
        idx = int(line.strip())
        itemX = items_D[idx]
        sim = -1.0
        for idy, itemy in items_D.items():
            simy = dis1(items_D[idx], items_D[idy])
            if (simy > sim or sim < 0) and idx!=idy:
                sim = simy
                itemY = itemy
        print "%s\tass\t%s"%(itemX['name'], itemY['name'])

load_data()
search_similar()
```

* 效果展示<br>
{% asset_img "302.png" [图片1] %}
* doc2topic<br>
{% asset_img "401.png" [图片1] %}
* topic2word<br>
{% asset_img "402.png" [图片1] %}

* 矩阵分解图谱<br>
{% asset_img "501.png" [图片1] %}

* 生成模型 VS 判别模型<br>
  * 判别方法：由数据直接学习决策函数Y=f(X)或者条件概率分布P(Y|X)作为预测的模型，即判别模型。<br>
  * 生成方法：由数据学习联合概率密度分布P(X,Y)，然后求出条件概率分布P(Y|X)作为预测的模型，即生成模型：P(Y|X)= P(X,Y)/ P(X)<br>

#### 手写LDA
* code<br>

```
import sys
import random

t_c = {}
tw_c = {}
td_c = {}

d_w = {}
d_w_t = {}
w_S = set()

ITER_NUM = 10000
TOPIC_NUM = 2
ALPHA = 0.01
BETA = 0.01

p_k = [0] * TOPIC_NUM
print p_k

def input():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        items = line.strip().split('\t')
        doc = items[0]
        word_L = items[1:]
        for word in word_L:
            d_w.setdefault(doc, list())
            d_w[doc].append(word)
            w_S.add(word)

def init():
    for d, w_L in d_w.items():
        for w in w_L:
            for t in range(TOPIC_NUM):
                t_c.setdefault(t, 0)
                tw_c.setdefault(t, dict())
                tw_c[t].setdefault(w, 0)
                td_c.setdefault(t, dict())
                td_c[t].setdefault(d, 0)

    for d, w_L in d_w.items():
        for w in w_L:
            r = random.random()
            if r < 0.5:
                t = 0
            else:
                t = 1

            d_w_t.setdefault(d, dict())
            d_w_t[d].setdefault(w, t)

            t_c[t] += 1
            tw_c[t][w] += 1
            td_c[t][d] += 1

            print d_w_t[d][w]

def sampling():
    for iter in range(ITER_NUM):
        print "iters is %d" % iter
        for d, w_L in d_w.items():
            for w in w_L:
                t = d_w_t[d][w]
                t_c[t] -= 1
                tw_c[t][w] -= 1
                td_c[t][d] -= 1

                for k in range(TOPIC_NUM):
                    p_k[k] = (tw_c[k][w] + BETA) * (td_c[k][d] + ALPHA) * 1.0 / (t_c[k] + BETA*len(w_S))
                sum = 0
                for k in range(TOPIC_NUM):
                    sum += p_k[k]
                for k in range(TOPIC_NUM):
                    p_k[k] /= sum
                for k in range(1, TOPIC_NUM):
                    p_k[k] += p_k[k-1]
                r = random.random()
                for k in range(TOPIC_NUM):
                    if(r<=p_k[k]):
                        t = k
                        break
                d_w_t[d][w] = t
                t_c[t] += 1
                tw_c[t][w] += 1
                td_c[t][d] += 1

def output():
    for d, w_L in d_w.items():
        for w in w_L:
            print "%s\t%s\t%d" % (d, w, d_w_t[d][w])

if __name__ == "__main__":
    input()
    print "input end..."
    init()
    print "init end..."
    sampling()
    print "samplint end..."
    output()
    print "output end..."
```

* train corpus<br>
```
doc1    枪      游戏    计算机  dota    电脑
doc4    娃娃    美丽    面膜    高跟鞋  裙子
doc5    购物    娃娃    裙子    SPA     指甲
doc2    枪      帅      电脑    坦克    飞机
doc3    游戏    坦克    飞机    数学    美丽
doc7    计算机  帅      枪      dota
doc6    美丽    购物    面膜    SPA     飘柔
```

* result<br>
```
doc2    枪      1
doc2    帅      1
doc2    电脑    1
doc2    坦克    1
doc2    飞机    1
doc3    游戏    1
doc3    坦克    1
doc3    飞机    1
doc3    数学    1
doc3    美丽    0
doc1    枪      1
doc1    游戏    1
doc1    计算机  1
doc1    dota    1
doc1    电脑    1
doc6    美丽    0
doc6    购物    0
doc6    面膜    0
doc6    SPA     0
doc6    飘柔    0
doc7    计算机  1
doc7    帅      1
doc7    枪      1
doc7    dota    1
doc4    娃娃    0
doc4    美丽    0
doc4    面膜    0
doc4    高跟鞋  0
doc4    裙子    0
doc5    购物    0
doc5    娃娃    0
doc5    裙子    0
doc5    SPA     0
doc5    指甲    0
```

写的样例默认有2个主题，一个是男生主题，一个是女生主题，lda的结果是可以把两个topic分开的。1-男生，0-女生。