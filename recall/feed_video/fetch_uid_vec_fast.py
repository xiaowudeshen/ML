#!/bin/python
import sys
import numpy as np
import multiprocessing as mul


def read_feed_vec(filename):
    feed_vec_dict = {}
    with open(filename) as f:
        for line in f:
            line_list = line.strip().split("\t")
            if len(line_list) != 8:
                continue
            gid,video_duration_code,video_width_code, video_height_code, video_rate_code, video_size_code,category_id, vec_str = line_list
            vec = list(map(float, vec_str.split(",")))
            feed_vec_dict[gid] = [vec, category_id]
    return feed_vec_dict

feed_vec_file = sys.argv[1]
feed_vec_dict = read_feed_vec(feed_vec_file)

def fetch_uid_vec(content_info):
    global feed_vec_dict
    cate_dict = {}
    uid,index, avg_gid_num, activate_day_rate, content = content_info
    content_list = content.split(",")
    vec_list = []
    for contentid in content_list:
        if contentid not in feed_vec_dict:
            continue
        vec, category = feed_vec_dict[contentid]
        vec_list.append(vec)
        cate_dict.setdefault(category, 0)
        cate_dict[category] += 1

    cate_items = sorted(cate_dict.items(), key=lambda d:d[1], reverse = True)
    cate_select = [ele[0] for ele in cate_items[:3]]
    cate_select_len = len(cate_select)
    if cate_select_len >=3:
        cate_num_list = [cate_dict[cate] for cate in cate_select[:3]]
        cate_num_sum = sum(cate_num_list)
        cate_num_rate = [round(num/cate_num_sum, 3) for num in cate_num_list]
        cate_str = "%s,%s,%s,%.3f,%.3f,%.3f"%(cate_select[0],cate_select[1],cate_select[2],cate_num_rate[0],cate_num_rate[1],cate_num_rate[2])
    elif cate_select_len == 2:
        cate_num_list = [cate_dict[cate] for cate in cate_select[:2]]
        cate_num_sum = sum(cate_num_list)
        cate_num_rate = [round(num/cate_num_sum, 3) for num in cate_num_list]
        cate_str = "%s,%s,%d,%.3f,%.3f,%d"%(cate_select[0],cate_select[1],0,cate_num_rate[0],cate_num_rate[1],0)
    elif cate_select_len == 1:
        cate_str = "%s,%d,%d,%d,%d,%d"%(cate_select[0],0,0,1,0,0)
    else:
        return '0'
    feed_num = len(vec_list)
    if feed_num < 1:
        return '0'
    result_vec = np.mean(np.array(vec_list), axis=0).tolist()
    result_str = list(map(str, result_vec))
    avg_gid_num = int(avg_gid_num)
    if avg_gid_num >= 90:
        avg_gid_num = 90
    activate_rate = round(avg_gid_num/90,3)
    activate_rate_2 = round(activate_rate**2,3)
    activate_rate_sqrt = round(activate_rate**0.5,3)

    activate_day_rate = float(activate_day_rate)
    activate_day_rate_2 = round(activate_day_rate**2, 3)
    activate_day_rate_sqrt = round(activate_day_rate**0.5, 3)
    return "%s\t%d\t%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s"%(uid,index,avg_gid_num,activate_rate,activate_rate_2,activate_rate_sqrt,activate_day_rate,activate_day_rate_2, activate_day_rate_sqrt, cate_str, ",".join(result_str))

def read_uid_info(filename):
    content_list = []
    flag = 0
    with open(filename) as f:
        for line in f:
            line_list = line.strip().split("\t")
            if len(line_list) != 4:
                continue
            uid,avg_gid_num, activate_day_rate,gid_str = line_list
            content_list.append([uid,flag, avg_gid_num, activate_day_rate, gid_str])
            flag += 1
    return content_list

def process():
    filename = sys.argv[2]
    content_list = read_uid_info(filename)
    cores = mul.cpu_count()
    cores_num = int(cores*0.7)
    with mul.Pool(cores_num) as pool:
        iter_test = pool.imap(fetch_uid_vec, content_list)
        for ele in iter_test:
            if ele == '0':
                continue
            else:
                print(ele)


if __name__ == '__main__':
    process()
