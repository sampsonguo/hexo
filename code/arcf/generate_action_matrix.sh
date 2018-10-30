#!/bin/bash

#==============================================================================
#author      guoxinpeng
#date        2017-09-26
#usage       sh user_item_matrix.sh ($DATE)
#==============================================================================

# cd to current folder
cd "$(dirname "$0")"

# get date
if [ $# -eq 0 ]
then
    DATE=$(date +"%Y%m%d" --date="1 days ago")
    YYYY_MM_DD=$(date +"%Y-%m-%d" --date="1 days ago")
    DEL_DATE=$(date +"%Y%m%d" --date="8 days ago")
else
    DATE=$1
    YYYY_MM_DD=$(date -d "$DATE" +'%Y-%m-%d')
    DEL_DATE=$(date -d "$DATE -7 days" +'%Y%m%d')
fi
echo "The date is $DATE"
echo "The deleted date is $DEL_DATE"

function hive_generate_user_item_action(){
    echo "Generate user item action matrix begin..."
    hive -e "
    alter table dm_music_prd.t_7d_imusic_iting_user_item_action drop if exists partition(ds='$DEL_DATE', item_type=1);
    insert overwrite table dm_music_prd.t_7d_imusic_iting_user_item_action partition(ds='$DATE', item_type=1)
    select user_id,
        item_id,
        action,
        location,
        tmstamp,
        extend
    from
    (
        select nvl(imei, '-1') as user_id,
          nvl(params['songid'], '-1') as item_id,
          case when event_id = '122|001|05|007' then 1 --play song
            when event_id = '080|001|03|007' then 2 -- download song
            when event_id = '083|001|01|007' then 3 -- favor song
            when event_id in ('099|001|01|007') then 4 -- switch song
            when event_id in ('073|001|01|007', '073|002|01|007', '073|003|01|007', '073|004|01|007') then 5 -- share song
            else -1 end as action,
          -1 as location,
          cast(event_time/100 as bigint) as tmstamp,
          case when event_id = '122|001|05|007' then params['songtime'] --play song
              when event_id = '080|001|03|007' then '' --download song
              when event_id = '083|001|01|007' then ''  -- favor song
              when event_id in ('099|001|01|007') then '' -- switch song
              when event_id in ('073|001|01|007', '073|002|01|007', '073|003|01|007', '073|004|01|007') then '' -- share song
              else '-1' end as extend
        from bi_music_prd.dw_music_pstsdk
        where day='$YYYY_MM_DD'
            and app_version_name>='6.0'
            and event_id in ('122|001|05|007', -- play song
              '080|001|03|007', --download song
              '083|001|01|007', -- favor song
              '099|001|01|007', -- switch song
              '073|001|01|007', '073|002|01|007', '073|003|01|007', '073|004|01|007' --share
            )
    ) f1
    join
    (
        select distinct song_id
        from dm_music_prd.t_0d_imusic_iting_song
    ) f2 on f1.item_id=f2.song_id
    "
    echo "Generate user item action matrix finished"
}

function hive_generate_user_item_play_threshold(){
    hive -e "
    alter table dm_music_prd.t_7d_imusic_iting_user_item_play_threshold drop if exists partition(ds='$DEL_DATE', item_type=1);
    insert overwrite table dm_music_prd.t_7d_imusic_iting_user_item_play_threshold partition(ds='$DATE', item_type=1)
    select user_id,
      item_id,
      1 as preference
    from dm_music_prd.t_7d_imusic_iting_user_item_action
    where ds='$DATE'
      and item_type=1
      and action=1
      and item_id!='-1'
      and cast(extend as bigint)>60*1000
    group by user_id, item_id
    "
}


## Main
hive_generate_user_item_action
hive_generate_user_item_play_threshold

