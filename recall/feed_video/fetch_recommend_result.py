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
                uid, feed_videolen_score = line_list
                uid_feature_list.append([uid, feed_videolen_score])
                if (index + 1) % block_size == 0:
                    yield uid_feature_list
                    uid_feature_list = []
    if len(uid_feature_list) > 0:
        yield uid_feature_list


def fetch_feed_data(content_list):
    uid, feed_videolen_score = content_list
    feed_videolen_score_list = feed_videolen_score.split(",") 
    expect_video_len = {}
    for ele in feed_videolen_score_list:
        ele_list = ele.split(":")
        feed, video_len, score = ele_list
        video_len = int(video_len)
        score = float(score)
        if video_len > 200:
            video_len = 200
        expect_video_len[feed] = score * np.log10(video_len)
        #expect_video_len[feed] = score
    item_sort = sorted(expect_video_len.items(), key=lambda d:d[1], reverse = True)
    sort_list = [ele[0] for ele in item_sort]
    #sort_list = ["%s_%s"%(ele[0], ele[1]) for ele in item_sort]
    return "%s\t%s"%(uid, ",".join(sort_list))

def process():
    uid_feature_file = sys.argv[1]
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
