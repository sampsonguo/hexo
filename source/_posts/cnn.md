---
title: cnn
date: 2018-03-14 13:51:41
tags:
---

#### Detect vertical/horizontal edges

CNN kernal below can be used for detecting vertical edges

1 | 0 | -1
--- | --- | ---
1 | 0 | -1
1 | 0 | -1

#### stride & padding

params | values
--- | ---
input volume size | W
stride | S
padding | P
filter size | F
output volume size | (W−F+2P)/S+1(W−F+2P)/S+1

#### advantages
* parameter sharing
* sparsity of connections

#### xavier_initializer
* uniform distribution: x = sqrt(6. / (in + out)); [-x, x]
* normal distribution: x = sqrt(2. / (in + out)); [-x, x]

#### Convolution Demo
* W: width = 5
* H: Height = 5
* D: Depth = 3
* K: number of filters = 2
* F: Filter size = 3
* S: Stride = 2
* P: Padding = 1

 {% asset_img "cnn002.png" [cnn002.png] %}

#### Pooling Demo

 {% asset_img "cnn003.png" [cnn003.png] %}

#### LeNet
* http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

 {% asset_img "lenet.png" [lenet.png] %}

* CONV
* POOL
* FC

#### AlexNet
* http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

 {% asset_img "lenet.png" [lenet.png] %}

 * Bigger
 * Deeper

#### VGGNet

#### GoogleNet

#### ResNet

#### REF
* http://cs231n.github.io/convolutional-networks/
