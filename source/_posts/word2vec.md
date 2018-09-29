---
title: word2vec
date: 2018-06-13 22:34:28
tags:
---

##### 起源和工具包
word2vec是google在2013年的模型，可以有效的将词映射到某个向量空间，经典的demo是：
vector('Paris') - vector('France') + vector('Italy') = vector('Rome')
其开源地址是：https://code.google.com/archive/p/word2vec/
此工具简单好用，在很多项目上起到了很好的效果

##### word embedding的各种姿势
* Topic Model：词袋模型，不分先后顺序，代表是LDA模型
* Word2vec：等价于三层神经网络，能捕捉时序信息
* FM：通过FM的weight矩阵将原始向量做变换
* DSSM：通过点击数据把user和item的分别作Deep模型，取出来最顶层的做word representation

另外NLP领域最新的模型是LSTM和Attention，能否利用其模型做embedding还需调研下
* LSTM：能更好的捕捉时序信息，解决long dependence的问题
* Attention：通过注意力机制


##### word2vec的各种包
* google word2vec：效果非常好，单机
* gensim：经典的自然语言处理包，用的人很多，单机
* spark mllib：不确定效果如何，分布式
* tensorflow：未尝试，分布式
（因为各种姿势都可以通过tensorflow来实现，包括FM，Word2Vec，以及各种矩阵分解，因此tf是很值得学习的工具。）

##### Why word embedding？
* 降维：原始ID维度的特征太大了，需要过多的训练数据，word embedding之后不需要那么多数据，而且防止过拟合

##### embedding的两类方法
* count-based：lsa系列，通过矩阵分解来做embedding
* predictive-based：通过预测词可能出现的词来做embedding，如word2vec

##### gensim建模+tensorboard可视化
* 生成gensim训练数据，每行一个doc，空格分隔
```
select list
from
(
    select user_id, concat_ws(' ', collect_set(item_id)) as list
    from dm_music_prd.t_7d_imusic_iting_user_item_action
    where ds>=20180520 and action=1 and item_id != -1 and extend>1000*90
    group by user_id
) t1
join
(
    select user_id
    from
    (
        select user_id, count(distinct item_id) as cnt
        from dm_music_prd.t_7d_imusic_iting_user_item_action
        where ds>=20180520 and action=1 and item_id != -1 and extend>1000*90
        group by user_id
    ) f
    where cnt >= 5 and cnt <= 50
) t2 on t1.user_id=t2.user_id
```

* gensim训练word2vec模型
```
/root/xpguo/anaconda3/bin/python w2v_v2.py user_songlist.v2 ./model_v2/w2v20180625 song_vector_v2

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 4:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp1, outp2 = sys.argv[1:4]

    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    # w2v model
    model.save(outp1)
    # word vectors
    model.wv.save_word2vec_format(outp2, binary=False)
```

* Tensorboard 可视化
```
1.	拉取id2name
mysql -h10.20.125.43 -umyshuju_r -p3KAjvBHaDB{gLE9H -e "select third_id, name from music.t_song_info where third_id is not null and name is not null and length(third_id)>0 and length(name)>0" > t_song_info

2.	写tensorboard模型
import sys, os
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

id2name = {}

## 可视化函数
## model: gensim模型地址
## output_path: tf board转换模型地址
def visualize(model, output_path):
    ## tf board转换模型名称，自定义
    meta_file = "w2x_metadata.tsv"

    ## 词向量个数*词向量维数
    placeholder = np.zeros((len(model.wv.index2word), 400))

    ## 读取ID2name文件
    inFp = open("./t_song_info", 'r')
    while True:
        line = inFp.readline()
        if not line:
            break
        items = line.strip().split('\t')
        if len(items) != 2:
            continue
        id2name[items[0]] = items[1]
    inFp.close()

    ## 地址+名称拼接
    with open(os.path.join(output_path,meta_file), 'wb') as file_metadata:

        ## 对于每个词向量，写文件
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '':
                print("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
#                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')
                file_metadata.write("{0}".format(id2name.get(word, 'null')).encode('utf-8') + b'\n')


    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable = False, name = 'w2x_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path,'w2x_metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))

if __name__ == "__main__":
    model = Word2Vec.load("/home/xpguo/gensim/word2vec/song_vector_v1/a")
visualize(model,"/home/xpguo/gensim/word2vec/song_vector_v1_tf_board")

3.	tensorboard可视化
tensorboard --logdir=/home/xpguo/gensim/word2vec/song_vector_v2_tf_board --port=6607
```


##### word2vec源码细节
* 预计算sigmoid(x)，将[-6, 6]的区间划分成1000份来计算。
```
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
for (i = 0; i < EXP_TABLE_SIZE; i++) {
  expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
  expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
}
```
* hash函数，每个char+原hash值*257，很常见的做法
```
// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}
```
* 线性探测法，hash后线性地去找词，要么找到返回词的位置，要么找到-1
```
// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}
```
* 将词添加到词典中， 有两个地方要加，第一是hash映射表，第二是词对应的count表
```
// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}
```
* 将词典排序，目的是为了建立哈弗曼树，有几个地方要注意：一是出现频率比较低的词，会被过滤掉；二是过滤掉之后需要讲hashtable重新hash，因为去掉了一些词，table不能用了。感慨一下，c语言要自己实现hash_dict，真心麻烦。
```
// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}
```
* 初始化网络参数，首先要搞清楚网络是什么样子的：
    * syn0: 存放从词典到第一层的参数，其实就是embedding_look_up表，模型去训练这个表，这个表就是word Embedding表。
    * syn1: 存放从第一层到第二层的参数。
    * syn1neg: 
    * hs: 
    * vocab_size:词典大小。
    * layer1_size:第一层大小。
    * 末尾还会创建一棵哈弗曼树
 
```
void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  CreateBinaryTree();
}
```
* 初始化采样表，注意不是直接用次数，而是用pow(次数, 0.75)来做，降低一下高频词的采到的概率。
```
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}
```
* 构建哈弗曼树，自底向上构建，code指的是，point指的是
```
// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}
```
* 