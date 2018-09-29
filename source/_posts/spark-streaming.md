---
title: spark-streaming
date: 2017-11-17 14:38:36
tags:
---

### spark streaming k-means
* decay(forgetfulness)
* mini-batch k-means
    * c_t+1 = [(c_t * n_t * a) + (x_t * m_t)] / [n_t + m_t]
    * n_t+t = n_t * a + m_t


### Broadcast Variables

* ref
https://databricks.com/blog/2015/01/28/introducing-streaming-k-means-in-spark-1-2.html


