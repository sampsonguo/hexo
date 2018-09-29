---
title: hive-param-tune
date: 2018-06-21 15:06:21
tags:
---
遇到一个hive调优的case，运行时长从原来的2h缩短到了10min，现总结下经验。

* hive运行时长太长的问题
先看源码：
```
insert overwrite table dm_music_prd.t_7d_imusic_iting_user2item_reclist_redis partition(ds='$DATE', item_type=1, scene='daily_rec', algo='arcf_001')
select concat('arcf_001_iting_', user_id) as user_id,  --step 5
  concat_ws(',', collect_list(item_id)) as reclist
from
(
  select user_id, --step4
    item_id,
    rank2
  from
  (
      select user_id, item_id, row_number() over (partition by user_id order by rank asc) as rank2  --step 3
      from
      (
          select user_id, item_id, rank   --step 1
          from dm_music_prd.t_7d_imusic_iting_user2item_reclist
          where ds='$DATE' and item_type=1 and scene='daily_rec' and rank<=$REC_MAX

          union all

          select /*+mapjoin(b)*/ user_id, item_id, rank   --step 2
          from
          (
            select distinct user_id from dm_music_prd.t_7d_imusic_iting_user2item_reclist_default  where ds='$DATE' and item_type=1 and scene='daily_rec'
          ) a
          join
          (
            select item_id, rank+1000 as rank from dm_music_prd.t_7d_imusic_iting_random_hot where ds='$DATE' and item_type=1 and rank<=$REC_MAX
          ) b
      ) tt
  ) f
  where rank2<=$REC_MAX
  distribute by user_id
  sort by rank2 asc
) t
where rank2<=$REC_MAX
group by user_id;
```
上述代码中标注了一些step，总结下经验：
1. step1：3kw用户*300item，只有选择操作，速度很快
2. step2：3kw用户和300item做笛卡尔积，用mapjoin把300item放到内存，速度很快
3. step3：3kw用户，每个用户内的600个item排序，3kw*log(600)的复杂度，耗时巨大
4. step4：按照user_id分桶，桶内进行排序，复杂度是reduce个数*log（reduce内数据量），耗时不确定
5. step5：按照user_id做group by的操作，速度很快

* 代码分析
核心的step3，按照上述代码运行的reduce个数是：157个
因为reduce是根据数据量来确定个数的，因此我们需要通过改变参数，增大reduce的个数

* 改进方案
```
set hive.map.aggr=false;
set mapreduce.input.fileinputformat.split.minsize=8000000;
set mapreduce.input.fileinputformat.split.minsize.per.node=8000000;
set mapreduce.input.fileinputformat.split.minsize.per.rack=8000000;
set mapreduce.input.fileinputformat.split.maxsize=16000000;
set hive.exec.reducers.bytes.per.reducer=67108864;
```
通过设置hive.exec.reducers.bytes.per.reducer为一个较小的值（上述代码是67M，默认是256M），来增多reduce个数，增加并行度。
最终reduce个数为600+个，10min跑完step3.
