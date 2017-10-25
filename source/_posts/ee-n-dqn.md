---
title: ee-n-dqn
date: 2017-10-24 23:59:18
tags:
---

### 先有鸡还是先有蛋？

#### 数据闭环
推荐系统根据用户日志来进行建模推荐，即：
日志 -> 推荐算法 -> 用户

<!-- more -->

日志也是由用户产生的，即：
用户 -> 日志

两者拼成一个环状，我们称之为"数据闭环"，即：
{% asset_img "1.png" [1.png] %}

#### "数据闭环"和"越推越窄"
这是一个"先有鸡还是先有蛋？"的问题
> 问：为什么给A推荐"摇滚"歌曲？
> 答：因为A过去听的都是"摇滚"歌曲，所以A喜欢"摇滚"。
> 问：推荐系统不给A用户推"非摇滚"，用户怎么能听到"非摇滚"？

在数据闭环中流转的都是"老Item"，新"Item"并没有多少展现机会，推荐变得越来越窄

#### "越推越窄"解决方案
越推越窄是典型的EE问题(explore & exploit)
解决方案有两类：
1. Bandit: epsilon-greedy, thompson sampling, UCB, linUCB
2. RL

#### Bandit的方案
bandit方案可以参考 http://banditalgs.com/ ，此处不做详细解释, 常见有以下方法：
* epsilon-greedy
* Thompson Sampling
* UCB
* linUCB

### RL的方案
RL解决了ML解决不了的两大问题：
* 延迟reward问题
* 数据缺失问题（EE问题，先有鸡先有单
RL有两大实体：
* agent
	* agent可以从environment中得到reward
	* agent需要知道自己的state, agent可以选择自己的action，即是一个p(action|state)的求解过程
* environment
	* environment需提供一个reward函数（往往自定义设计）
	* environment需进行state的状态转移（往往是黑盒子）
	* environment需接收agent的action

两大实体互相作用，有几大重要的元素:
* action: 动作，由agent产生，作用于environment
* reward: 奖赏，environment针对agent的state+action产生的奖赏or惩罚
* state: agent的状态，由action实现状态转移，即p(state_x+1|state_x, action_x)的马尔科夫转移过程
* observation: 即state的外在表现

用图可视化即
{% asset_img "2.png" [2.png] %}

### 两种observation
observation是state的外在表现，那么observation也有两种：
1. state space: 直接表达state的空间
	比如cartpole中的observation(state)的定义是[position of cart, velocity of cart, angle of pole, rotation rate of pole]
	有意思的是，并不需要（往往也不知道）其具体的含义，只知道是一个四维数组
2. pixels: 
	直接从像素级别（声音，嗅觉，味觉，触觉）等得到observation
	有意思的是，某时刻的图片不一定能够表达全部信息（比如速度），因此可能用图片串表示observation
	p(action_t|pixel_t, pixel_t-1, pixel_t-2, ..., pixel_1)
	
### RL
reinforcement learning有两个比较通用的算法
* Q learning 
* policy gradients

### Q-learning
Q-learning的核心是计算Q值，那么Q值的定义是：
Q value =  what our return would be, if we were to take an action in a given state
即Q是一个两维空间[observation, action]，表示在某个observation时执行某个action的总的reward和（立即的reward和之后的reward的discount）

#### Q值 -> action 
假设已经有了Q值，那么如何sample出一个action，可以简单用目前observation下的最大的Q，顺便加一些随机性来探索。

#### Q值更新
Q值的更新需要用到Bellman equation，即：
Q(s,a) = r + γ(max(Q(s’,a’))
其中,
s表示state，也即observation
a表示action
r表示current reward
s’表示next state，即state下做出action之后到达的new state
a’表示next state后的策略，max(Q(s’,a’)表示s’后的最佳策略的Q值
γ表示future reward的一个discount

有意思的是，我们用差分，设置步长，确定方向，来逼近这个值：
```
Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
```

OpenAI的FrozenLake-v0完整的code如下：
```
import gym
import numpy as np
env = gym.make('FrozenLake-v0')
#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    #jList.append(j)
    rList.append(rAll)
```


### DQN(Deep Q Network)
比如利用CNN来做observation来表达state，即是DQN，后续再更新。










