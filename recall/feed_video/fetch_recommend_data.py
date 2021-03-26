#!/bin/python3
import numpy as np
import multiprocessing as mul
import sys

def fetch_uid_feature(filename):
    uid_feature_list = []
    block_size = 100000
    with open(filename) as f:
        for index, line in enumerate(f):
            line_list = line.strip().split("\t")
            if len(line_list) != 2:
                continue
            else:
                uid, feature = line_list
                feature_list = list(map(float, feature.split(",")))
                feature_arr = np.array(feature_list)
                uid_feature_list.append([uid, feature_arr])
                if (index + 1) % block_size == 0:
                    yield uid_feature_list
                    uid_feature_list = []
    if len(uid_feature_list) > 0:
        yield uid_feature_list

def fetch_feed_feature(filename):
    feed_feature_list = []
    index2feed = {}
    index = 0
    with open(filename) as f:
        for line in f:
            line_list = line.strip().split("\t")
            if len(line_list) != 2:
                continue
            else:
                feed, feature = line_list
                feature_list = list(map(float, feature.split(",")))
                feed_feature_list.append(feature_list)
                index2feed[index] = feed
                index += 1
    return index2feed, np.array(feed_feature_list).T

def fetch_feed_video_len(filename):
    feed_videolen = {}
    with open(filename) as f:
        for index, line in enumerate(f):
            if index == 0:
                continue
            
            line_list = line.strip().split(",")
            if len(line_list) < 8:
                continue
            feed = line_list[0]
            video_duration = line_list[1]
            feed_videolen[feed] = int(video_duration)
    return feed_videolen

def fetch_filter_info(filename):
    uid_dict = {}
    with open(filename) as f:
        for line in f:
            line_list = line.strip().split("\t")
            if len(line_list) < 2:
                continue
            else:
                uid, viewed_gid = line_list
                gid_list = viewed_gid.split(",")
                uid_dict[uid] = {}
                for gid in gid_list:
                    uid_dict[uid][gid] = 0
    return uid_dict

feed_videolen_file = sys.argv[1]
feed_feature_file = sys.argv[2]
filter_uid_file = sys.argv[4]

feed_videolen = fetch_feed_video_len(feed_videolen_file)
index2feed, feed_arr  = fetch_feed_feature(feed_feature_file)
filter_uid_dict = fetch_filter_info(filter_uid_file)

def fetch_feed_data(content_list):
    global feed_videolen
    global index2feed
    global feed_arr
    global filter_uid_dict 
    uid, feature = content_list
    dot_value = np.dot(feature, feed_arr)
    predict_arr = 1/(1 + np.exp(-1*dot_value))
    expect_video_len = []
    if uid in filter_uid_dict:
        for i in range(predict_arr.size):
            feed = index2feed[i]
            if feed in filter_uid_dict[uid]:
                continue
            video_len = feed_videolen[feed]
            score = predict_arr[i]
            temp = "%s:%d:%f"%(feed,video_len, score)
            expect_video_len.append(temp)
    else:        
        for i in range(predict_arr.size):
            feed = index2feed[i]
            video_len = feed_videolen[feed]
            score = predict_arr[i]
            temp = "%s:%d:%f"%(feed,video_len, score)
            expect_video_len.append(temp)
    return "%s\t%s"%(uid, ",".join(expect_video_len))

def process():
    uid_feature_file = sys.argv[3]
    uid_feature_iter = fetch_uid_feature(uid_feature_file) 
    cores = mul.cpu_count()
    cores_num = int(cores*0.8)
    for block in uid_feature_iter:
        with mul.Pool(cores_num) as pool:
            iter_result = pool.map(fetch_feed_data, block)
            for ele in iter_result:
                print(ele)

if __name__ == '__main__':
    process() 
