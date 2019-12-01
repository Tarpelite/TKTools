from transformers import BertTokenizer
from multiprocessing import Pool, cpu_count
import argparse
import gc
import sys
import os
import psutil
from tqdm import *
import fileinput
import time

# initialize the ray tools
num_cpus = psutil.cpu_count(logical=False)


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

output_dir = ("tk_pieces")

mem_cnt = 0
max_mem = 1024*1024*500
global_cnt = 0
piece_size = 10
data = []


if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def tokenize(idx):
    # idx, data = idx
    if idx+ piece_size > len(data):
        lim = len(data)
    else:
        lim = idx + piece_size
    results = []
    for i in range(idx, lim):
        results.append([tokenizer.tokenize(x) for x in data[i]])
    return results

def write(data, cnt):
    dst = os.path.join(output_dir, str(cnt))
    print("write exapmles to {}".format(dst))
    with open(dst, "w+", encoding="utf-8") as dst_f:
        for line in data:
            line_a = " ".join(line[0]) + "\n"
            line_b = " ".join(line[1]) + "\n"
            dst_f.write(line_a)
            dst_f.write(line_b)
            dst_f.write("\n")

def process(data, tokenizer):
    id_list = []
    id_s = 0
    while id_s + piece_size < len(data):
        id_list.append(id_s)
        id_s += piece_size
    id_list.append(id_s)

    with Pool(cpu_count()) as p:
        results = list(tqdm(p.imap(tokenize, id_list), total=len(id_list)))
    results_collect = []
    for res in results:
        results_collect.extend(res)
    return results_collect

def load(file_iterator):
    mem_cnt = 0
    data = []
    for line in file_iterator:
        line = line.strip().split("\t")
        data.append(line)
        mem_cnt = sys.getsize(data)
        if mem_cnt >= max_mem: 
            return data
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--piece_size", type=int, default=10)
    parser.add_argument("--max_mem", type=int, default=1)

    args = parser.parse_args()
    piece_size = args.piece_size

    max_mem = args.max_mem*1024*1024
    file_path = "2019-10-25_AllMatchType_AllTime_ClicksBigger0_ExceptQEqualK_Normalized_DocLengthBigger10.txt"
    file_iterator = fileinput.input(file_path)
    global_mem = 0
    mem_cnt = 0
    data = []
    cnt = 0
    for line in file_iterator:
        line = line.strip().split("\t")
        data.append(line)
        mem_cnt = sys.getsizeof(data)
        if mem_cnt >= max_mem: 
            global_mem += mem_cnt
            print("processing {}  GB of 675.6 GB of file".format( global_mem*1.0000/(1024*1024*1024)))
            data = process(data, tokenizer)
            write(data, cnt)
            data.clear()
            gc.collect()
            cnt += 1
    print("finished")
            

    


