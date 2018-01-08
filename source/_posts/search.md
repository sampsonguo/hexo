---
title: search
date: 2018-01-08 16:05:05
tags:
---

### 垂直领域搜索
APP搜索引擎，属于垂直领域的搜索引擎，相对于泛需求的搜索引擎会简单很多。
泛搜索引擎 -> 意图识别（下APP，听歌，找wiki） -> 该意图垂直领域引擎 
即在垂直领域，不需要挂载意图识别模块。

<!-- more -->

### 相关性(relevance)和重要性(importance)
相关性：用户搜索“美”，京东和美团对比，美团的“相关性”更高，所以美团比京东排序高。
重要性：用户搜索“美”，美团和美丽说对比，美团的“重要性”更高（美团用户量更大，美丽说小众），所以美团比美丽说的排序高。

相关性和重要性，是搜索里需要tradeoff的两个指标，这里的tradeoff，往往是建立排序模型。
即：相关性召回+重要性召回 -> 排序 

### 相关性 
#### 文本相似性
* 标题精准匹配（Exact Match）
用户搜索“美团”，大概率是对“APP名称的搜索”，因此按照名称精准匹配，最简单最有效。
* 标题模糊匹配（Fuzzy Match）
用户会打错字，用户会query打不全，因此模糊匹配非常重要。
    * 拼音化
用户搜索“meituan”，也能够出来“美团”的结果，是因为将app名称进行拼音化再匹配。
    * query纠错/改写
用户搜索“没团”，也能够出来“美团”的结果，是将query纠错后再匹配。
    * 编辑距离
用户搜索“美”，也能够出来“美团”的结果，是因为“美”和“美团”的编辑距离为1，比较小。
【编辑距离ref】：https://zh.wikipedia.org/wiki/%E7%B7%A8%E8%BC%AF%E8%B7%9D%E9%9B%A2
* 内容TF-IDF 
    * TF(Term Frequency)
query分词后，词语在内容（描述，评论）中出现的次数，可以使0-1值，可以是次数，可以是log（次数）等等任意变种。
    * IDF(Inverse Document Frequency)
query分词后，词语在query中的重要性，可以是0-1值，可以是log(N/Nt)，可以是log(1+N/Nt)等等任意变种。
    * 组合
IDF将query分词后设置词的权重，TF将词去和文档匹配，TF-IDF就是加权的词和文档的匹配。
【TF-IDF ref】https://en.wikipedia.org/wiki/Tf%E2%80%93idf
* 内容BM25
BM25是一种TF-IDF-like retrieval functions，即原理相通。BM25假设有多个词，即sum(每个词的TF-IDF)。
公式为：
score(D, Q) = sum(TF-IDF) D stands for doc, Q stands for query.

TF: {% asset_img "TF.png" [TF] %}
* f(qi, D) is qi's term frequency in the document D
* |D| is the length of the document D in words
* avgdl is the average document length in the text collection from which documents are drawn.
* k1 and b are free parameters, usually chosen, in absence of an advanced optimization, as k1 in [1.2,2.0] and b = 0.75.

IDF: {% asset_img "IDF.png" [IDF] %}
* N is the total number of documents in the collection
* n(qi) is the number of documents containing qi.

【BM25 ref】https://en.wikipedia.org/wiki/Okapi_BM25

#### 语义相似性
* Topic Model
    乔布斯的苹果和水果店的苹果，识别语义。建议采用”维基预料训练“，而非query来训练，因为LDA对短文本效果不好。
* 类型相关
    识别query的类型，比如搜索”聊天“，那就从”聊天软件“类目中召回APP。

### 重要性
#### 属性重要性 
* 星级/评论挖掘
* 曝光量/点击量/下载量/安装量/使用量/注册量/付费量/留存量（率）
* 商业化价值

#### 图重要性
* Page Rank值

#### 其他召回
* x%用户下载召回
* 新品召回
* 运营召回
* 个性化召回

### 排序 
* 点击熵
    识别是精准需求（文本相关）还是泛需求（语义相关）的指标，可以用来确定召回比例，也可以用来作为排序特征（此时需要交叉两个召回来源）。
* 条件概率 p(ctr|query, user, scene)
给某人（对人的个性化），场景（对场景的把控，在机场推荐航班相关，晚饭时间推荐美食相关），query（用户对意图的主动描述）的最优化问题。

### 评估（IR evaluation）
* MAP: Mean Average Precision，不再赘述。
* nDCG: Normalized Discounted cumulative gain，按照位置加权。

【IR metrics ref】http://lixinzhang.github.io/xin-xi-jian-suo-zhong-de-ping-jie-zhi-biao-maphe-ndcg.htmll
【MAP vs NDCG ref】https://www.youtube.com/watch?v=qm1In7NH8WE

