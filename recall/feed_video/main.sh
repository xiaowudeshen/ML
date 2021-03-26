#!/bin/bash

#day=$(date -d "1 day ago" +%Y-%m-%d)
day=$1
mkdir -p ../data/${day}/two_dnn

function fetch_uid_data()
{

  yesterday='2021-03-30'
  ago_15='2021-03-15'
    hive -e "
    select uid, concat_ws("\t",collect_set(concat(dt,'_',gid_event))) day_gid_event_info from
(select dt, if(mid!='0' and mid!="null",mid,tuid) uid,concat_ws('#',collect_set(concat(a.gid,"@",max_event))) gid_event from
(
  select mid,tuid,device,extend_info['gid'] gid, max(cast(event as int)) max_event,dt from log_feed where dt<='${yesterday}' and dt>='${ago_15}' and app=29 and  event in (1002,1301) and extend_info ['feed'] is null group by mid,tuid,device,extend_info['gid'],dt
  ) a
  group by mid,tuid,dt
  ) b
  group by uid
limit 200000000
    " >  ../data/${day}/two_dnn/uid_gid_day_${day}.csv
    python3 fetch_uid_gid.py ../data/${day}/two_dnn/uid_gid_day_${day}.csv > ../data/${day}/two_dnn/uid_15_gid_${day}
}

function fetch_feed_data()
{
  yesterday='2021-03-30'

    hive -e "select gid,video_duration,category_id,video_width,video_height,round(video_width/video_height,2) rate,video_size,description FROM feed_short_video WHERE status = 2 and feed_content_from = 'qtt' and comment_count > 0
limit 200000000" >../data/${day}/two_dnn/all_feed_info_${day}.csv
    python3 fetch_gid_code.py ../data/${day}/two_dnn/all_feed_info_${day}.csv >  ../data/${day}/two_dnn/all_feed_vec_${day}

    hive -e "select short_video.gid gid from
(select gid FROM feed_short_video WHERE status = 2 and feed_content_from = 'qtt' and comment_count > 0)  short_video
join
(select gid from feed_top_content lateral view explode(split(top_category, ',')) tb as category
where id > 3719 and to_date(update_time) >= '${yesterday}'  and category in ('top_pv', 'top_ctr', 'top_durtion') group by gid ) top_info
on short_video.gid = top_info.gid limit 200000000" > ../data/${day}/two_dnn/top_feed_info_${day}.csv

    awk 'BEGIN{FS="\t";OFS="\t"}NR==FNR{a[$1]=0;next} {if($1 in a){print $0}}' ../data/${day}/two_dnn/top_feed_info_${day}.csv  ../data/${day}/two_dnn/all_feed_vec_${day} > ../data/${day}/two_dnn/top_feed_vec_${day}
    
    python3 merge_col_info.py ../data/${day}/two_dnn/top_feed_vec_${day} > ../data/${day}/two_dnn/merge_col_feed_vec_${day} 
    python3 predict_query_feature_net.py ../data/${day}/two_dnn/merge_col_feed_vec_${day} > ../data/${day}/two_dnn/feed_feature_${day}
}
fetch_uid_data &
fetch_feed_data &
wait

python3 fetch_uid_vec_fast.py ../data/${day}/two_dnn/all_feed_vec_${day} ../data/${day}/two_dnn/uid_15_gid_${day} > ../data/${day}/two_dnn/uid_vec_${day}
awk 'BEGIN{FS="\t";OFS=","}{print $2,$3}' ../data/${day}/two_dnn/uid_vec_${day} > ../data/${day}/two_dnn/merge_col_uid_vec_${day}

python3 predict_user_feature_net.py ../data/${day}/two_dnn/merge_col_uid_vec_${day} > ../data/${day}/two_dnn/uidindex_feature_${day}
awk 'BEGIN{FS="\t";OFS="\t"}NR==FNR{a[$2]=$1;next} {if($1 in a){print a[$1],$2}}'  ../data/${day}/two_dnn/uid_vec_${day} ../data/${day}/two_dnn/uidindex_feature_${day} > ../data/${day}/two_dnn/uid_feature_${day}

python3 fetch_recommend_data.py ../data/${day}/two_dnn/all_feed_info_${day}.csv   ../data/${day}/two_dnn/feed_feature_${day}  ../data/${day}/two_dnn/uid_feature_${day}  ../data/${day}/aiqingli/frequence_control_info_${day} > ../data/${day}/two_dnn/uid_feed_corr_score_${day} 

python3 fetch_recommend_result.py ../data/${day}/two_dnn/uid_feed_corr_score_${day} > ../data/${day}/two_dnn/uid_feed_data_${day}


