#!/bin/bash

#==============================================================================
#author      guoxinpeng
#date        2018-06-09
#usage       sh imusic_rec.sh ($DATE)
#==============================================================================

### Params
TIME_MIN=60*1000
TIME_MAX=10*60*1000
PRIOR_MAX=100
AR_MAX=10
REC_MAX=300
PRIOR_DAYS=7

# cd to current folder
cd "$(dirname "$0")"

# get date
if [ $# -eq 0 ]
then
    DATE=$(date +"%Y%m%d" --date="1 days ago")
    YYYY_MM_DD=$(date +"%Y-%m-%d" --date="1 days ago")
    DEL_DATE=$(date +"%Y%m%d" --date="8 days ago")
    PRIOR_MIN_DATE=$(date +"%Y%m%d" --date="$PRIOR_DAYS days ago")
else
    DATE=$1
    YYYY_MM_DD=$(date -d "$DATE" +'%Y-%m-%d')
    DEL_DATE=$(date -d "$DATE -7 days" +'%Y%m%d')
    PRIOR_MIN_DATE=$(date -d "$DATE -$PRIOR_DAYS days" +'%Y%m%d')
fi
echo "The date is $DATE"
echo "The deleted date is $DEL_DATE"

function hive_generate_candidate_user_prior(){
    echo "Generate candidate user begin..."
    hive -e"
    set mapred.job.priority=VERY_HIGH;
    alter table dm_music_prd.t_7d_imusic_iting_candidate_user_prior drop if exists partition(ds='$DEL_DATE', item_type=1, scene='daily_rec');
    with song_info as
    (
      select song_id, song_name, t1.singer_id, singer_name
      from
      (select song_id, name as song_name, singer_id from dm_music_prd.t_0d_imusic_iting_song) t1
      join
      (select singer_id, name as singer_name from dm_music_prd.t_0d_imusic_iting_singer) t2
      on t1.singer_id=t2.singer_id
    )
    insert overwrite table dm_music_prd.t_7d_imusic_iting_candidate_user_prior partition(ds='$DATE', item_type=1, scene='daily_rec')
    select user_id,
      item_id,
      prior,
      debug_info
    from
    (
        select user_id,
          item_id,
          prior,
          row_number() over (partition by user_id order by prior desc) as rank,
          concat(nvl(t2.song_id,''), '|',
            nvl(t2.song_name,''), '|',
            nvl(t2.singer_id,''), '|',
            nvl(t2.singer_name,'')) as debug_info
        from
        (
            select user_id,
              item_id,
              count(1) as prior -- can be improved
            from dm_music_prd.t_7d_imusic_iting_user_item_action
            where ds<='$DATE'
              and ds>='$PRIOR_MIN_DATE'
              and item_type=1
              and action=1
              and item_id>0
              and user_id>0
              and cast(extend as bigint)>=$TIME_MIN
              and cast(extend as bigint)<=$TIME_MAX
            group by user_id, item_id
        ) t1
        left outer join song_info t2 on t1.item_id=t2.song_id
    ) f
    where rank<=$PRIOR_MAX;
    "
    echo "Generate candidate user finished"
}

function hive_calculate_candidate_posterior(){
    echo "Calculate candidate posterior begin..."
    hive -e "
    set mapred.job.priority=VERY_HIGH;
    alter table dm_music_prd.t_7d_imusic_iting_user2item_reclist drop if exists partition(ds='$DEL_DATE', item_type=1, scene='daily_rec');
    insert overwrite table dm_music_prd.t_7d_imusic_iting_user2item_reclist partition(ds='$DATE', item_type=1, scene='daily_rec')
    select user_id,
        item_id,
        score,
        row_number() over(partition by user_id order by rank asc) as rank,
        rec_reason,
        debug_info,
        singer_id
    from
    (
        select user_id,
          item_posterior as item_id,
          0 as score,
          rank,
          item_prior as rec_reason,
          debug_info,
          singer_posterior as singer_id,
          row_number() over(partition by user_id, singer_posterior order by rank asc) as singer_rank
        from
        (
            select user_id,
              max(item_prior) as item_prior,
              item_posterior,
              min(rank) as rank,
              max(debug_info) as debug_info,
              max(singer_posterior) as singer_posterior
            from
            (
              select user_id,
                item_id as item_prior,
                item_posterior,
                row_number() over(partition by user_id order by rank*1.0/prior asc) as rank, -- can be improved
                debug_info,
                singer_posterior
              from
              (
                select user_id,
                  item_id,
                  prior
                from dm_music_prd.t_7d_imusic_iting_candidate_user_prior
                where ds='$DATE'
                  and item_type=1
                  and scene='daily_rec'
              ) t1
              join
              (
                select item_prior,
                  item_posterior,
                  rank,
                  debug_info,
                  singer_posterior
                from dm_music_prd.t_7d_imusic_iting_item2item_bool_matrix_evaluation_metrics
                where ds='$DATE'
                  and item_type=1
                  and rank<=$AR_MAX
              ) t2 on t1.item_id=t2.item_prior
            ) t
            where rank<=$REC_MAX
            group by user_id, item_posterior
        ) tt
    ) ttt
    where singer_rank<=3;
    "
    echo "Calculate candidate posterior finished"
}
 
function  hive_construct_default_reclist(){
hive -e"
insert overwrite table dm_music_prd.t_7d_imusic_iting_default  partition(ds='$DATE', item_type=1, scene='daily_rec')
select 'default' as user_id, song_id as item_id ,0.0 as score, row_number() over (partition by 1 order by hot  asc) as rank , ''
as rec_reason,''  as debug_info,singer_id  from(
select max(song_id) as song_id ,song_name,max(singername) ,max(singer_id) as singer_id,count(song_id) as hot from
 (select item_id from dm_music_prd.t_7d_imusic_iting_user_item_action where  ds=$DATE )  t1
join 
(select song_id,a.name as song_name,b.name as singername ,b.singer_id 
from dm_music_prd.t_0d_imusic_song a join dm_music_prd.t_0d_imusic_iting_singer b on a.singer_id=b.singer_id) t2
on t1.item_id=t2.song_id where singername!='儿歌'   group by song_name )t3 where hot>300   order by rand() limit 300;

insert overwrite table dm_music_prd.t_7d_imusic_iting_user2item_reclist_default partition(ds='$DATE', item_type=1, scene='daily_rec')
select user_id,item_id,score,rank,rec_reason,debug_info,singer_id  from dm_music_prd.t_7d_imusic_iting_default  where ds='$DATE' union all
select user_id,item_id,score,rank,rec_reason,debug_info,singer_id from   dm_music_prd.t_7d_imusic_iting_user2item_reclist
where ds=$DATE;
"
}


function hive_calculate_random_hot(){
    echo "Calculate random hot begin..."
    hive -e "
    set mapred.job.priority=VERY_HIGH;
    alter table dm_music_prd.t_7d_imusic_iting_random_hot drop if exists partition(ds='$DEL_DATE', item_type=1);
    with song_info as
    (
      select song_id, song_name, t1.singer_id, singer_name
      from
      (select song_id, name as song_name, singer_id from dm_music_prd.t_0d_imusic_iting_song) t1
      join
      (select singer_id, name as singer_name from dm_music_prd.t_0d_imusic_iting_singer) t2
      on t1.singer_id=t2.singer_id
    )
    insert overwrite table dm_music_prd.t_7d_imusic_iting_random_hot partition(ds='$DATE', item_type=1)
    select item_id,
      row_number() over(order by rank1*rand() desc) as rank2,
      concat(nvl(t2.song_id, ''), '|',
        nvl(t2.song_name, ''), '|',
        nvl(t2.singer_name, ''), '|') as debug_info
    from
    (
      select item_id,
        row_number() over(order by hot_rate desc) as rank1
      from
      (
        select item_id,
          count(1) as hot_rate
        from dm_music_prd.t_7d_imusic_iting_user_item_action
        where ds='$DATE'
            and item_type=1
            and action=1
            and item_id>0
            and user_id>0
            and cast(extend as bigint)>60*1000
        group by item_id
      ) f
    ) t1
    left outer join song_info t2 on t1.item_id=t2.song_id
    where rank1<=10000;
    "
}

function hive_post_processing(){
    echo "Post processing begin..."
    hive -e "

    "
    echo "Post processing finished"
}

function hive_transform_reclist2redisformat(){
    echo "Transform reclist to redis format begin..."
    hive -e "
    set mapred.job.priority=VERY_HIGH;
    set hive.mapred.mode=nonstrict;
    alter table dm_music_prd.t_7d_imusic_iting_user2item_reclist_redis drop if exists partition(ds='$DEL_DATE', item_type=1, scene='daily_rec', algo='arcf_001');
    insert overwrite table dm_music_prd.t_7d_imusic_iting_user2item_reclist_redis partition(ds='$DATE', item_type=1, scene='daily_rec', algo='arcf_001')
    select concat('arcf_001_iting_', user_id) as user_id,
      concat_ws(',', collect_list(item_id)) as reclist
    from
    (
      select user_id,
        item_id,
        rank2
      from
      (
          select user_id, item_id, row_number() over (partition by user_id order by rank asc) as rank2
          from
          (
              select user_id, item_id, rank
              from dm_music_prd.t_7d_imusic_iting_user2item_reclist
              where ds='$DATE' and item_type=1 and scene='daily_rec' and rank<=$REC_MAX

              union all

              select /*+mapjoin(b)*/ user_id, item_id, rank
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
    "
    echo "Transform reclist to redis format finished"
}

function load_hive2redis(){
    echo "load hive2redis begin..."
#    curl  "http://bdsp-manager.api.prd.bj01.vivo.lan:8080/excution/taskCode/VDLQCZGC/add.json?jdbc.reader.sqlArgs=$DATE,arcf_001&redis.writer.keyTimeout=1000000000"
#    curl  "http://bdsp-facade.api.prd.bj01.vivo.lan:8080/bdsp/api.json?taskCode=WHUNHFTD&methodType=addExecution&ds=${ds}"
    echo "load hive2redis finished"

    echo "Update music algo version begin..."
    hive -e"
    insert overwrite table dm_music_prd.music_algo_version
    select 'arcf_001_iting' as versionname,
        concat(1, ',', '$DATE', ',', '$DATE') as content,
        1 as item_type,
        'daily_rec' as scene;
    "
    echo "Update music algo version finished"

    echo "load redis update status begin..."
    curl "http://bdsp-manager.api.prd.bj01.vivo.lan:8080/excution/taskCode/ASHFAJBA/add.json?redis.writer.keyTimeout=1000000000"
    echo "load redis update status finished"
}

##Main
#hive_generate_candidate_user_prior
#hive_calculate_candidate_posterior
#hive_construct_default_reclist
#hive_calculate_random_hot
#hive_post_processing
#hive_transform_reclist2redisformat
load_hive2redis
