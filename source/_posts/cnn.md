---
title: cnn
date: 2017-11-17 13:51:41
tags:
---

* Detect vertical/horizontal edges

CNN kernal below can be used for detecting vertical edges

1 | 0 | -1
--- | --- | ---
1 | 0 | -1
1 | 0 | -1

* stride & padding

params | values
--- | ---
input volume size | W
stride | S
padding | P
filter size | F
output volume size | (W−F+2P)/S+1(W−F+2P)/S+1

* advantages
    * parameter sharing
    * sparsity of connections

* xavier_initializer
    * uniform distribution: x = sqrt(6. / (in + out)); [-x, x]
    * normal distribution: x = sqrt(2. / (in + out)); [-x, x]

*

