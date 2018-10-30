#!/bin/bash

#==============================================================================
#author      guoxinpeng
#date        2017-09-21
#usage       sh imusic-itemcf.sh ($DATE)
#==============================================================================

### Params
ACTION_DAYS=30
LIST_MIN=2
LIST_MAX=100
ITEM_CNT_MIN=2
ITEM_CNT_MAX=1000000
PAIR_MIN=2
PAIR_MAX=1000000
AR_MAX=20

### cd to current folder
cd "$(dirname "$0")"

### get date
if [ $# -eq 0 ]
then
    DATE=$(date +"%Y%m%d")
    DEL_DATE=$(date +"%Y%m%d" --date="7 days ago")
    ACTION_MIN_DATE=$(date +"%Y%m%d" --date="$ACTION_DAYS days ago")
else
    DATE=$1
    DEL_DATE=$(date -d "$DATE -7 days" +'%Y%m%d')
    ACTION_MIN_DATE=$(date -d "$DATE -$ACTION_DAYS days" +'%Y%m%d')
fi
echo "The date is $DATE"
echo "The deleted date is $DEL_DATE"

# song list from mysql to file and delete the file out of date
function load_songlist_mysql2file(){
    echo "Song list from mysql to file begin..."
    mysql -h10.20.125.43 -umyshuju_r -p3KAjvBHaDB{gLE9H -e "
    select t2.id,
    t2.playlist_id as songlist_id,
    t1.third_id as item_id,
    '-1' as sq,
    t2.create_time,
    t2.update_time
    from
    (
        select id as t_song_id,
            third_id
        from music.t_song_info
    ) t1
    join
    (
        select id,
            playlist_id,
            create_time,
            song_id,
            update_time
        from music.t_playlist_song_relation
    ) t2 on t1.t_song_id = t2.song_id;
    " > songlist."$DATE"
    echo "Song list from mysql to file finished"

    echo "Delete song list file at $DEL_DATE begin..."
    if [ -f songlist."$DEL_DATE" ]
    then
      rm songlist."$DEL_DATE"
      echo "Delete song list file at $DEL_DATE finished"
    else
      echo "Warning: Songlist.$DEL_DATE does not exist!"
    fi
}

# song list from file to hive
function load_songlist_file2hive(){
    echo "Load song list file to hive begin..."
    hive -e "
    alter table dm_music_prd.t_7d_imusic_iting_song_list drop if exists partition(ds='$DEL_DATE', item_type=1);
    load data local inpath './songlist.$DATE' overwrite into table dm_music_prd.t_7d_imusic_iting_song_list partition(ds='$DATE', item_type=1);
    "
    echo "Load song list file to hive finished"
}

# song list to bool matrix
function hive_transform_songlist2boolmatrix(){
    echo "Transform t_7d_imusic_song_list to t_7d_imusic_user_item_preference_bool_matrix begin..."
    hive -e "
    set hive.merge.mapredfiles = true;
    alter table dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix drop if exists partition(ds='$DEL_DATE', item_type=1);
    insert overwrite table dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix partition(ds='$DATE', item_type=1)
    select user_id,
        item_id,
        preference
    from
    (   
        -- source 1: song list
        select concat('sl_', songlist_id) as user_id,
          item_id,
          1 as preference
        from dm_music_prd.t_7d_imusic_iting_song_list
        where ds='$DATE' and item_type=1
        group by songlist_id, item_id

        union all

--        -- source 2: play song
--        select concat('s_', user_id) as user_id,
--          item_id,
--          preference
--        from dm_music_prd.t_7d_imusic_iting_user_item_play_threshold
--        where ds='$DATE' and item_type=1

        -- source 3: favor, download, share
        select concat('s_', user_id) as user_id,
            item_id,
            1 as preference
        from dm_music_prd.t_7d_imusic_iting_user_item_action
        where ds<='$DATE'
            and ds>='$ACTION_MIN_DATE'
            and item_type=1
            and action in (2, 3, 5)
            and item_id!='-1'
    ) t
    group by user_id, item_id, preference
    "
    echo "Transform t_7d_imusic_song_list to t_7d_imusic_iting_user_item_preference_bool_matrix finished"
}

# filter song list
function hive_filter_songlist(){
    echo "Filter song list begin..."
    hive -e "
    alter table dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix_list_filtered drop if exists partition(ds='$DEL_DATE', item_type=1);
    insert overwrite table dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix_list_filtered partition(ds='$DATE', item_type=1)
    select t1.user_id,
        t1.item_id,
        t1.preference
    from
    (
        select user_id,
            item_id,
            preference
        from dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix
        where ds='$DATE' and item_type=1
    ) t1
    join
    (
        select user_id
        from
        (
            select user_id,
                count(1) as item_cnt
            from dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix
            where ds='$DATE' and item_type=1
            group by user_id
        ) f
        where item_cnt >= $LIST_MIN and item_cnt <= $LIST_MAX
    ) t2 on t1.user_id = t2.user_id;
    "
    echo "Filter song list finished"
}

# frequent item filter
function hive_filter_frequent_item(){
    echo "Filter item begin..."
    hive -e "
    alter table dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix_frequent_item_filtered drop if exists partition(ds='$DEL_DATE', item_type=1);
    insert overwrite table dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix_frequent_item_filtered partition(ds='$DATE', item_type=1)
    select t1.user_id,
        t1.item_id,
        t1.preference
    from
    (
        select user_id,
            item_id,
            preference
        from dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix_list_filtered 
        where ds='$DATE' and item_type=1
    ) t1
    join
    (
        select item_id
        from
        (
            select item_id,
                count(1) as user_cnt
            from dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix_list_filtered 
            where ds='$DATE' and item_type=1
            group by item_id
        ) f
        where user_cnt > $ITEM_CNT_MIN and user_cnt < $ITEM_CNT_MAX
    ) t2 on t1.item_id=t2.item_id;
    "
    echo "Filter item finished"
}

# frequent pair filter
function hive_filter_pair(){
    echo "Filter item pair begin..."
    hive -e "
    alter table dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix_frequent_pair_filtered drop if exists partition(ds='$DEL_DATE', item_type=1);
    insert overwrite table dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix_frequent_pair_filtered partition(ds='$DATE', item_type=1)
    select item_prior,
        item_posterior,
        pair_cnt
    from
    (
        select t1.item_id as item_prior,
            t2.item_id as item_posterior,
            count(1) as pair_cnt
        from
        (
            select user_id,
                item_id,
                preference
            from dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix_frequent_item_filtered
            where ds='$DATE' and item_type=1
        ) t1
        join
        (
            select user_id,
                item_id,
                preference
            from dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix_frequent_item_filtered
            where ds='$DATE' and item_type=1
        ) t2 on t1.user_id=t2.user_id
        where t1.item_id != t2.item_id
        group by t1.item_id, t2.item_id
    ) t
    where pair_cnt >= $PAIR_MIN
        and pair_cnt <= $PAIR_MAX
    "
    echo "Filter item pair finished"
}

# conf, lift, kulr, ir
function hive_calculate_evaluation_metrics(){
    echo "Calculate evaluation metrics begin..."
    hive -e "
    alter table dm_music_prd.t_7d_imusic_iting_item2item_bool_matrix_evaluation_metrics drop if exists partition(ds='$DEL_DATE', item_type=1);
    set mapred.job.priority=VERY_HIGH;
    set hive.mapred.mode=nonstrict;
    with t_item_cnt as
    (
        select item_id,
            count(1) as item_frequent_cnt
        from dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix_frequent_item_filtered
        where ds='$DATE' and item_type=1
        group by item_id
    ),
    t_all as
    (
      select count(distinct user_id) as all_cnt
      from dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix_frequent_item_filtered
      where ds='$DATE' and item_type=1
    )
    insert overwrite table dm_music_prd.t_7d_imusic_iting_item2item_bool_matrix_evaluation_metrics partition(ds='$DATE', item_type=1)
    select item_prior,
      item_posterior,
      prior_cnt,
      posterior_cnt,
      pair_cnt,
      confidence,
      max_conf,
      min_conf,
      lift,
      kulc,
      ir,
      rank,
      '' as debug_info,
      '' as singer_prior,
      '' as singer_posterior
    from
    (
        select item_prior,
          item_posterior,
          a as prior_cnt,
          b as posterior_cnt,
          ab as pair_cnt,
          ab*1.0/a as confidence,
          greatest(ab*1.0/a, ab*1.0/b) as max_conf,
          least(ab*1.0/a, ab*1.0/b) as min_conf,
          ab*1.0*n/a/b as lift,
          0.5*ab*1.0/a + 0.5*ab*1.0/b as kulc,
          abs(a-b)*1.0/(a+b-ab) as ir,
          row_number() over (partition by item_prior order by ab*1.0/a*log(1.0+ab*1.0*n/a/b) desc) as rank
        from
        (
          select /*+mapjoin(t_prior, t_posterior, t_all)*/
            item_prior,
            item_posterior,
            t_prior.item_frequent_cnt as a,
            t_posterior.item_frequent_cnt as b,
            t1.pair_cnt as ab,
            t_all.all_cnt as n
          from
          (
              select item_prior,
                  item_posterior,
                  pair_cnt
              from dm_music_prd.t_7d_imusic_iting_user_item_preference_bool_matrix_frequent_pair_filtered
              where ds='$DATE' and item_type=1
          ) t1
          join t_item_cnt t_prior on t1.item_prior=t_prior.item_id
          join t_item_cnt t_posterior on t1.item_posterior=t_posterior.item_id
          join t_all t_all
        ) t
    ) f 
    where rank<=$AR_MAX;
    "
    echo "Calculate evaluation metrics finished"
}


function hive_calculate_evaluation_metrics_debug_info(){
    echo "hive calculate evaluation metrics debug info begin..."
    hive -e "
    with song_info as
    (
      select song_id, song_name, t1.singer_id, singer_name
      from
      (select song_id, name as song_name, singer_id from dm_music_prd.t_0d_imusic_iting_song) t1
      join
      (select singer_id, name as singer_name from dm_music_prd.t_0d_imusic_iting_singer) t2
      on t1.singer_id=t2.singer_id
    )
    insert overwrite table dm_music_prd.t_7d_imusic_iting_item2item_bool_matrix_evaluation_metrics partition(ds='$DATE', item_type=1)
     select item_prior    ,
      item_posterior,
      prior_cnt     ,
      posterior_cnt ,
      pair_cnt      ,
      confidence    ,
      max_conf      ,
      min_conf      ,
      lift          ,
      kulc          ,
      ir            ,
      row_number() over (partition by item_prior order by old_rank asc) as rank,
      debug_info    ,
      singer_prior  ,
      singer_posterior
    from
    (
        select /*+mapjoin(t2, t3)*/
          item_prior    ,
          item_posterior,
          prior_cnt     ,
          posterior_cnt ,
          pair_cnt      ,
          confidence    ,
          max_conf      ,
          min_conf      ,
          lift          ,
          kulc          ,
          ir            ,
          rank as old_rank,
          concat(nvl(t2.song_id,''), '|',
            nvl(t2.song_name,''), '|',
            nvl(t2.singer_id,''), '|',
            nvl(t2.singer_name,''), ',',
            nvl(t3.song_id,''), '|',
            nvl(t3.song_name,''), '|',
            nvl(t3.singer_id,''), '|',
            nvl(t3.singer_name,'')) as debug_info,
          nvl(t2.singer_id, '') as singer_prior,
          nvl(t3.singer_id, '') as singer_posterior,
          nvl(t2.song_name, '') as song_prior,
          nvl(t3.song_name, '') as song_posterior,
          row_number() over (partition by item_prior, nvl(t3.singer_id, '') order by rank asc) as song_in_singer_rank
        from
        (
          select *
          from dm_music_prd.t_7d_imusic_iting_item2item_bool_matrix_evaluation_metrics
          where ds='$DATE'
            and item_type=1
        ) t1
        left outer join song_info t2 on t1.item_prior=t2.song_id
        left outer join song_info t3 on t1.item_posterior=t3.song_id
    ) tt
    where song_in_singer_rank<=1 
        and ((singer_prior='') or
            (singer_posterior='') or
            (singer_prior!=singer_posterior))
        and ((song_prior='') or
            (song_posterior='') or
            (song_prior!=song_posterior))
    "
    echo "hive calculate evaluation metrics debug info finished"
}

## Main
load_songlist_mysql2file
load_songlist_file2hive
hive_transform_songlist2boolmatrix
hive_filter_songlist
hive_filter_frequent_item
hive_filter_pair
hive_calculate_evaluation_metrics
hive_calculate_evaluation_metrics_debug_info
