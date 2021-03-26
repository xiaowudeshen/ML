from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import jieba
import gensim
import multiprocessing as mul
import sys

def fetch_video_duration_code(video_duration):
    #the video_duration code len is 13
    value = int(video_duration/5)

    if video_duration <=1:
        return 0
    elif video_duration >= 2 and video_duration <= 12:
        return video_duration - 1
    else:
        return 12

def fetch_video_width_code(video_width):
    #the video_width code len is 6

    if video_width <540:
        return 0
    elif video_width == 540:
        return 1
    elif video_width > 540 and video_width < 576:
        return 2
    elif video_width == 576:
        return 3
    elif video_width > 576 and video_width< 720:
        return 4
    else:
        return 5

def fetch_video_height_code(video_height):
    #the video_width code len is 4

    if video_height <1024:
        return 0
    elif video_height == 1024:
        return 1
    elif video_height > 1024 and video_height < 1280:
        return 2
    else:
        return 3

def fetch_rate_code(video_rate):
    #the video_width code len is 4
    if video_rate <0.56:
        return 0
    elif video_rate == 0.56:
        return 1
    elif video_rate > 0.56 and video_rate < 0.58:
        return 2
    else:
        return 3

def fetch_size_code(video_size):
    #the video_width code len is 13
    num = int(video_size/500000)
    if num <10:
        return num
    elif num in (10,11):
        return 10
    elif num in (12,13,14):
        return 11
    else:
        return 12



def read_corpus_dict(fname, block_size=10000):
    block_list = []
    with open(fname) as f:
        for index, line in enumerate(f):
            if index == 0:
                continue
            line_list = line.strip().split(",")
            if len(line_list) < 8:
                continue
            gid,video_duration,category_id,video_width,video_height,rate,video_size = line_list[:7]
            description = ",".join(line_list[7:])
            video_duration = int(video_duration)
            category_id = int(category_id)
            video_width = int(video_width)
            video_height = int(video_height)
            rate = float(rate)
            video_size = int(video_size)
            video_duration_code = fetch_video_duration_code(video_duration)
            video_width_code = fetch_video_width_code(video_width)
            video_height_code = fetch_video_height_code(video_height)
            video_rate_code = fetch_rate_code(rate)
            video_size_code = fetch_size_code(rate)
            description_list = jieba.lcut(description)
            if int(category_id) >=90:
                category_id = '0'
            gid_base_code = "%s\t%s\t%s\t%s\t%s\t%s\t%s"%(gid,video_duration_code,video_width_code, video_height_code, video_rate_code, video_size_code,category_id)
            block_list.append([gid_base_code, description_list])
            if (index+1)%block_size == 0:
                yield block_list
                block_list = []
    if len(block_list) > 0:
        yield block_list

model_file = "doc2vec/curpus_350w_epoch_30_doc2vec.model.bin"
model = Doc2Vec.load(model_file,mmap='r')

def fetch_vec(ele):
    global model
    gid_base_code, title_cut_list = ele
    inferred_vector = model.infer_vector(title_cut_list)
    inferred_vector_list = inferred_vector.tolist()
    inferred_vector_str = ",".join(list(map(str, inferred_vector_list)))
    return "%s\t%s"%(gid_base_code, inferred_vector_str)

def process():
    feed_info_file = sys.argv[1]
    docid_iter = read_corpus_dict(feed_info_file)
    
    cores = mul.cpu_count()
    cores_num = int(cores*0.5)
    for block in docid_iter:
        with mul.Pool(cores_num) as pool:
            iter_result = pool.imap(fetch_vec, block)
            for ele in iter_result:
                print(ele)
 
if __name__ == '__main__':
    process()
