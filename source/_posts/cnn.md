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

<<<<<<< HEAD
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
=======
#### LeNet

#### AlexNet

#### VGGNet

#### GoogleNet

#### ResNet
```aidl
import torchvision.models as models
import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

""" resnet.eval()
Yes, it hurts the testing accuracy. If you use resnet.eval(), batch normalization layer uses running average/variance instead of mini-batch statistics. You can improve the performance when using resnet.eval() by changing the momentum coefficient in batch normalization layer.
It is recommended to change nn.BatchNorm2d(16) to nn.BatchNorm2d(16, momentum=0.01). The default value of the momentum is 0.1.
"""
img_path="./dog.jpg"

image_transform = transforms.Compose([
    transforms.Scale([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])

resnet = models.resnet152(pretrained=True)
resnet = resnet.eval()
img_vec = image_transform(Image.open(img_path).convert('RGB')).unsqueeze(0)
img_vec = resnet(Variable(img_vec)).data.squeeze(0).cpu().numpy()
np.savetxt("vec.txt",img_vec)
```


#### REF
* http://cs231n.github.io/convolutional-networks/
