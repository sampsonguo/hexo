---
title: tf_wnd
date: 2018-01-05 11:46:39
tags:
---

#### wide_n_deep code

```

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import logging
import json
import math

from six.moves import urllib
import tensorflow as tf

# 读取文件
reader = tf.TextLineReader(skip_header_lines = 0)

# 文件列表
train_input_files = ["/root/xpguo/wnd/1.txt", "/root/xpguo/wnd/2.txt"]

input_file_list = []
for input_file in train_input_files:
    if len(input_file) > 0:
        input_file_list.append(tf.train.match_filenames_once(input_file))

filename_queue = tf.train.string_input_producer(
                tf.concat(input_file_list, axis = 0),
                num_epochs = 10,     # strings are repeated num_epochs
                shuffle = True,     # strings are randomly shuffled within each epoch
                capacity = 512)

batch_size = 3

(_, records) = reader.read_up_to(filename_queue, num_records = batch_size)
samples = tf.decode_csv(records, record_defaults = column_defaults, field_delim = ',')
label = tf.cast(samples[self.column_dict["label"]], dtype = tf.int32)
feature_dict = {}
for (key, value) in self.column_dict.items():
    if key == "label" or value < 0 or value >= len(samples):
        continue
    if key in ["user_features", "ads_features"]:
        feature_dict[key] = tf.string_split(samples[value], delimiter = ';')
    if key in ["user_weights", "ads_weights"]:
        feature_dict[key] = self.string_to_number(
                tf.string_split(samples[value], delimiter = ';'),
                dtype = tf.float32)
return feature_dict, label
```