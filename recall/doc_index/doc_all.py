# -*- coding: utf-8 -*-
"""
@Author: yangxiaohan
@Date:   2019-04-26
@Desc:   Create the inverted index file to hash entity to article
"""
import os
import io
import sys
import json
import time
import datetime
import math
import redis
import joblib
import csv

GLOBAL_HOT_FLAG = 'GLOBAL_HOT_FLAG'
def readfile(filename):
    keyword_doc_dict = {}
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        birth_header = next(csv_reader)
        for row in csv_reader:
            docid,keywords_v4,create_time,rec_validshow,rec_read_pv,rec_validread,hot_status = row
            info_list = get_docscore(row)
            docid, keywords_v4_list, docscore = info_list
            for word in keywords_v4_list:
                keyword_doc_dict.setdefault(word, [])
                keyword_doc_dict[word].append([docid, docscore])
            if hot_status != '':
                keyword_doc_dict.setdefault(GLOBAL_HOT_FLAG, [])
                keyword_doc_dict[GLOBAL_HOT_FLAG].append([docid, docscore])
    return keyword_doc_dict

# score threshold
def get_docscore(tokens):
    docid, keywords_v4, create_time, rec_validshow, rec_read_pv, rec_validread, hot_status = tokens
    if rec_validshow == "None":
        rec_validshow = 0
    if rec_read_pv == "None":
        rec_read_pv = 0
    if rec_validread == "None":
        rec_validread = 0
    validshowpv = int(rec_validshow)
    readpv = int(rec_read_pv)
    validreadpv = int(rec_validread)
    if validshowpv != 0:
        ctr = min(float(readpv)/validshowpv, 0.3)
        validctr = min(float(validreadpv)/validshowpv, 0.25)
    else:
        ctr = 0
        validctr = 0
    pvscore = math.log(float(validreadpv)) if validreadpv > 0 else 0.
    pvscore = min(pvscore, 10.) / 20
    docscore = 0.3*ctr + 0.6*validctr + 0.1*pvscore
    # time decay for score
    d1 = datetime.datetime.today()
    d2 = datetime.datetime.strptime(create_time,"%Y-%m-%d")
    dgap = int((d1-d2).days)
    docscore = docscore * (0.98**dgap)
    keywords_v4_list = [ele.split(":")[0] for ele in keywords_v4.split(",")]
    return docid, keywords_v4_list, docscore

def fetch_conn():
    detail_redis = {"hostname":"", "port":, "password":"", "db":}
    redis_conn = redis.Redis(host=detail_redis["hostname"], port=detail_redis["port"], password = detail_redis["password"], db=detail_redis["db"])
    return redis_conn


def write_redis(keyword_doc_dict):
    redis_conn  = fetch_conn()
    pipe = redis_conn.pipeline(transaction=False)
    print("writing redis ...")
    index = 0
    for key in keyword_doc_dict:
        score_docid_list = keyword_doc_dict[key]
        try:
            redis_conn.ltrim(key, 1,0)
        except:
            pass
        for score_docid in score_docid_list:
            pipe.rpush(key, score_docid)
            pipe.expire(key, 7*24*60*60)
            if (index+1) % 1000 == 0:
                pipe.execute()
                print("index=",index)
                time.sleep(0.001)
            index += 1
    pipe.execute()
    print("written redis success!")
    
def process():
    filename = sys.argv[1]
    keyword_doc_dict = readfile(filename)
    result_dict = {}
    score_thr = 0.03
    for word in keyword_doc_dict:
        result_dict[word] = []
        doc_score_list = keyword_doc_dict[word]
        doc_score_list_sort = sorted(doc_score_list ,key=(lambda x:x[1]),reverse=True)
        for ele in doc_score_list_sort[:1000]:
            docid, score = ele
            """
            if score < score_thr:
                continue
            """
            value = "%f:%s"%(score, docid)
            result_dict[word].append(value)
    #today = datetime.datetime.today().strftime("%Y-%d-%m_%H")
    #save_file = "data/keyword_score_doc_%s"%today
    save_file = sys.argv[2]
    with open(save_file, 'w') as f:
        for key in result_dict:
            content =  "%s\t%s\n"%(key, ",".join(result_dict[key]))
            f.write(content)
if __name__ == '__main__':
    process()
