#!/bin/bash

#==============================================================================
#author      guoxinpeng
#date        2017-09-27
#usage       sh run.sh ($DATE)
#==============================================================================

# cd to current folder
cd "$(dirname "$0")"

# get date
if [ $# -eq 0 ]
then
    DATE=$(date +"%Y%m%d" --date="1 days ago")
else
    DATE=$1
fi
echo "The date is $DATE"
echo "The deleted date is $DEL_DATE"

sh -x load_song_singer_mysql2hive.sh
sh -x generate_action_matrix.sh $DATE
sh -x item2item.sh $DATE
sh -x imusic_rec.sh $DATE

