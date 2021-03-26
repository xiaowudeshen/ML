#!/usr/bin/env python
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
import pymysql
import traceback
import datetime
import math
import redis
from sklearn.externals import joblib


class DocIndex(object):
    def __init__(self, document_data_file, run_time):
        self.run_time = run_time
        self.document_data_file = document_data_file
        self.time_str = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
        self.score_thr = 0.0

        # parameters
        self.redis_dict = {
                }

        # load_data
        self.document_data = None
        self.load_document_data_file()



    def load_document_data_file(self):
        print("getting entity_doc_dict ...")
        self.document_data = list()
        with open(self.document_data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue

                try:
                    tokens = line.split("\t")

                    item = {
                        "content_id": tokens[0].strip(),
                        "content_type": tokens[1].strip(),
                        "type": tokens[2].strip(),
                        "create_time": tokens[3].strip(),
                        "publish_time": tokens[4].strip(),
                        "show_pv": int(tokens[5].strip()),
                        "valide_read_pv": int(tokens[6].strip()),
                        "read_pv": int(tokens[7].strip()),
                        "rec_readlong": int(tokens[8].strip()),
                    }
                    
                    # 最近48h内的doc取film_per_merage，最近24hdoc取其他的
                    now_ts = datetime.datetime.today()
                    now_ts = int(now_ts.timestamp())
                    publish_time = int(tokens[4].strip())
                    if now_ts - publish_time > 86400 * 2:
                        continue

                    # features
                    film_per_merage = tokens[14].strip()
                    film_per_merage = [item.split("###")[0] for item in film_per_merage.split("^")]
                    item['film_per_merage'] = [item for item in film_per_merage if item]
                    item['film_per_merage_all'] = item['film_per_merage']
                    
                    if now_ts - publish_time < 86400:
                        d_cate_id_lv1 = [tokens[2].strip()]
                        item["d_cate_id_lv1_all"] = [item for item in d_cate_id_lv1 if item]
                    
                        d_entities_v4 = tokens[10].strip()
                        d_entities_v4 = [item.split("#")[0] for item in d_entities_v4.split("^")]
                        item["d_entities_v4"] = [item for item in d_entities_v4 if item]
                        item["d_entities_v4_all"] = item["d_entities_v4"]

                        d_keywords_v4 = tokens[12].strip()
                        d_keywords_v4 = [item.split(":")[0] for item in d_keywords_v4.split(",")]
                        item["d_keywords_v4"] = [item for item in d_keywords_v4 if item]
                        item["d_keywords_v4_all"] = item["d_keywords_v4"]
                    
                        zhihu_tag_system = tokens[13].strip()
                        zhihu_tag_system = [item.split(":")[0] for item in zhihu_tag_system.split(",")]
                        item["zhihu_tag_system"] = [item for item in zhihu_tag_system if item]
                        item["zhihu_tag_system_all"] = item["zhihu_tag_system"]

                    
                    # calc article score
                    docscore, docscore_cate = self.get_docscore(item)
                    # score filter
                    if docscore < self.score_thr:
                        continue
                    item["score"] = docscore
                    item["score_cate"] = docscore_cate
                    
                except:
                    print("[ERROR] wrong line: %s" % line)
                    continue

                self.document_data.append(item)
        print("get entity_doc_dict success!")

    def get_docscore(self, data):
        content_id = data["content_id"]

        content_type = data["content_type"]
        type = data["type"]
        create_time = data["create_time"]
        publish_time = data["publish_time"]

        show_pv = data["show_pv"]
        valid_read_pv = data["valide_read_pv"]
        read_pv = data["read_pv"]
        rec_readlong = data["rec_readlong"]

        # ctr base score
        if show_pv == 0:
            ctr, validctr = 0.5,0.5
        else: 
            ctr = min((float(read_pv) + 100)  / (show_pv + 200), 1)
            validctr = min((float(valid_read_pv) + 100) / (show_pv + 200), 1)
        docscore = 0.6 * validctr + 0.4 * ctr # [0, 0.5]
   
        # 有效展示pv衰减
        show_pv_score = math.log(float(show_pv), 10) if show_pv > 0 else 0.  # [0, 8]
        show_pv_score = min(show_pv_score, 10) # [0,10]
        
        # 时间衰减
        now_ts = datetime.datetime.today()
        now_ts = int(now_ts.timestamp())
        publish_time = int(publish_time)
        time_diff = math.log(float(now_ts - publish_time), 10) # [0,5]
        time_diff = max(time_diff, 1)  # [1,5]

        docscore = (1 + 2 * docscore) * (0.90 ** time_diff) + 0.95 ** show_pv_score 
        docscore_cate = 0.75 * (0.2 * validctr + 0.1 * ctr + 0.1 * show_pv_score) +  0.90 ** time_diff + 0.95 ** show_pv_score 

        return docscore, docscore_cate

    def gen_inverted_index(self, schema):
        print("\n******** %s process start ********" % schema)

        doc_dict = self.get_doc_dict(schema)

        self.sort_doc_dict(doc_dict)

        self.save_doc_data(schema, doc_dict)

        self.write_redis(schema)

        print("******** %s process end ********" % schema)

    def get_doc_dict(self, schema):
        print("getting doc_dict ...")
        doc_dict = dict()
        for item in self.document_data:
            feature = item.get(schema,[])
            score = item["score"]
            if 'cate_id' in schema or 'zhihu' in schema or 'film' in schema:
                score = item["score_cate"]
            content_id = item["content_id"]
            content_type = item["content_type"]

            for fea in feature:
                key = fea + "_" + str(content_type)
                # 增加规则 不区分图文和视频 
                if '_all' in schema:
                    key = fea

                if key not in doc_dict:
                    doc_dict[key] = []
                doc_dict[key].append([score, content_id])
        print("getting doc_dict success!")

        return doc_dict

    def sort_doc_dict(self, doc_dict):
        print("sorting docs ...")
        for key, value in doc_dict.items():
            value.sort(key=lambda x: x[0], reverse=True)
            doc_dict[key] = value[0:500]
        print("sorting docs success!")

    def save_doc_data(self, schema, doc_dict):
        print("saving data ...")
        data_file = "data/%s_%s" % (schema, self.run_time)
        with open(data_file, "w") as f:
            for key, value in doc_dict.items():
                f.write(key + "\t" + ",".join([str(item[0]) + ":" + str(item[1]) for item in value]))
                f.write("\n")
        print("save data success!")

    def write_redis(self, schema):
        print("writing redis ...")
        redis_conn = redis.from_url(self.redis_dict[schema])
        data_file = "data/%s_%s" % (schema, self.run_time)

        pipe = redis_conn.pipeline()
        index = 0
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    key, value = line.split("\t")
                except:
                    print("line split error, schema=%s, line=[%s]" % (schema, line))
                    continue

                index += 1
                pipe.set(key, value, 3600 * 4)

                if index % 1000 == 0:
                    pipe.execute()
            pipe.execute()

        print("write redis success!")


def main():
    document_data_file = sys.argv[1]
    run_time = sys.argv[2]
    doc_index = DocIndex(document_data_file, run_time)

    doc_index.gen_inverted_index("d_cate_id_lv1_all")
    doc_index.gen_inverted_index("d_entities_v4_all")
    doc_index.gen_inverted_index("d_keywords_v4_all")
    doc_index.gen_inverted_index("zhihu_tag_system_all")
    doc_index.gen_inverted_index("film_per_merage_all")



if __name__ == "__main__":
    main()

