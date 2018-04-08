---
title: elastic-search.md
date: 2018-04-02 19:21:02
tags: Search
---

#### Install
* 配置公网访问
```
修改配置文件 config/elasticsearch.yml
network.host: 0.0.0.0
```
* 扩大vm.max_map_count
```
sysctl -w vm.max_map_count=655360
```
* node 安装
```
http.cors.enabled: true
http.cors.allow-origin: "*"
```
* kibana install
```
kibana安装后外网无法访问：
修改config/kibaba.yml下的server.host为0.0.0.0
```
* cluster config
```
cluster.name: xxx-search

node.name: node-2
node.master: false
node.data: true

network.host: 0.0.0.0

http.cors.enabled: true
http.cors.allow-origin: "*"

discovery.zen.ping.unicast.hosts: ["xxx.xxx.xxx.xxx"]
```
*