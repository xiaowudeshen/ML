import multiprocessing as mul

def fetch_test():

if __name__ == '__main__':
    data = 
    cores = mul.cpu_count()
    cores_num = int(cores*0.5)
    with mul.Pool(cores_num) as pool:
        iter_result = pool.imap(fetch_test, data)
        for ele in iter_result:
            print(ele)
