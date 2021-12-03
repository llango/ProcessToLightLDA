# -*- coding:utf8 -*-
import functools
import mmap
from collections import Counter
import time
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
# import jieba.posseg as psg

from config import NOT_USE_fLAG

# 加载停用词
from text2uci import read_stopWords
stopWordsList = read_stopWords('/home/lcl/LightLDA/ProcessForLightLDA/docs/stopWords')

def read_vocab(vocab_file_path):
    word2idx = dict()
    with open(vocab_file_path,'r') as vocab_file:
        idx = 1
        for line in vocab_file:
            line = line.strip()
            word2idx[line] = idx
            idx += 1
    return word2idx

def write_docWord(docIndex, doc_counter, f_docWord):
    for key in doc_counter.keys():
        line = str(docIndex) + " " + str(key) + " " + str(doc_counter[key])
        f_docWord.write(line + '\n')

# def token_docs(docs_list):
#     # 结巴分词时 docs_list不能太长
#     t1 = time.time()
#     # 使用paddle模式会降低效率
#     tokens_list = list(psg.cut("".join(docs_list), use_paddle=False))
#     # 根据词性进行筛选
#     tokens_list = [x.word for x in tokens_list if x.flag not in NOT_USE_fLAG or '\n' in x.word]
#     # 删除停用词
#     tokens_list = [token for token in tokens_list if token not in stopWordsList]
#     tokens_list = "\ ".join(tokens_list)
#     t2 = time.time()
#     # print("One process token's time is :{}".format(t2-t1))
#     return tokens_list

def get_vocab(docs_list):
    counter = Counter()
    t1 = time.time()
    for doc in docs_list:
        doc_tokens = doc.strip().split("\ ")
        doc_tokens = [token for token in doc_tokens if " " not in token and '\\' not in token]
        counter.update(doc_tokens)
    t2 = time.time()
    print("One process get vocab's time is :{}".format(t2 - t1))
    return counter

def get_uci(docs_list, word2idx=None):
    assert word2idx is not None
    doc_counter_list = []
    t1 = time.time()
    for doc in docs_list:
        doc_counter = Counter()
        doc_tokens = doc.strip().split("\ ")
        doc_tokens = [word2idx[token.strip()] for token in doc_tokens if token in word2idx.keys()]
        doc_counter.update(doc_tokens)
        if len(doc_counter)>0:
            doc_counter_list.append(doc_counter)
    t2 = time.time()
    print("UCI data preprocess:{}".format(t2-t1))
    return doc_counter_list


def docs_supplier(filename, nums_split=500000, seek_position = None):
    """A generator for frames"""
    f = open(filename)
    fsize = Path(filename).stat().st_size
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    docs_list = []
    tot = 0
    tot_line = 0
    if seek_position is not None:
        mm.seek(seek_position)

    for line in iter(mm.readline, b""):
        if len(line) < 5:
            tot += len(line)
            continue
        try:
            docs_list.append(line.decode('utf-8'))
        except:
            tot += len(line)
            continue
        # update pbar
        tot += len(line)
        tot_line += 1
        if tot_line % nums_split == 0:
            #print("Read rate {}".format(tot/fsize))
            yield docs_list
            docs_list = []
    mm.close()
    print("Read rate {}".format(tot / fsize))
    yield docs_list


# def get_tokenized_dataset_mp(nums_workers, raw_url, token_out_url):
#     """
#     将原始文本转成分词之后的文本
#     :param nums_workers: 进程数
#     :param raw_url: 放原文本的文件路径
#     :param token_out_url: 输出分词之后的文件路径
#     :return:
#     """
#     seek_p = 3245630
#     docs = docs_supplier(raw_url, nums_split=10, seek_position=seek_p)
#     log_f = open(token_out_url, "a+")
#     t1 = time.time()
#     with Pool(nums_workers) as processes:
#         # result = processes.starmap(token_docs, zip(docs))
#         for result in tqdm(processes.imap_unordered(token_docs, docs, chunksize=30), total=(5601551 - seek_p)/10):
#             log_f.writelines("".join(result))
#     # log_f.writelines("".join(result))
#     t2 = time.time()
#     tm_cost = t2 - t1
#     print("Time is :{}".format(tm_cost))


def build_vocab_dataset_mp(num_workers,tokenized_url,vocab_url):
    """

    :param num_workers:进程数
    :param tokenized_url: 分词之后的文本路径
    :param vocab_url: 输出的词典文件
    :return:
    """
    docs = docs_supplier(tokenized_url, nums_split=10000)
    vocab_all = Counter()
    t1 = time.time()
    with Pool(num_workers) as processes:
        for vocab in tqdm(processes.imap_unordered(get_vocab,docs,chunksize=30), total=5601551/10000):
            vocab_all = vocab_all + vocab
    t2 = time.time()
    tm_cost = t2 - t1
    print("Get Vocab time is :{}".format(tm_cost))
    # 根据单词的次数过滤
    vocab_all = {k: v for k, v in vocab_all.items() if v > 5}
    # 根据单词出现次数排序
    vocab_all = sorted(vocab_all.items(), key= lambda x:x[0], reverse=True)
    sorted_words = [x[0] for x in vocab_all]

    vocab_file = open(vocab_url, 'w')
    vocab_file.writelines("\n".join(sorted_words))
    vocab_file.close()

def get_UCI_dataset_mp(num_workers,tokenized_url,uci_url,vocab_url):
    word2idx = read_vocab(vocab_url)
    p_seek = 4639998
    nums_split = 1000
    docs = docs_supplier(tokenized_url, nums_split=nums_split,seek_position=p_seek)
    uci_out = open(uci_url, 'a+')
    get_uci_ = functools.partial(get_uci, word2idx=word2idx)
    doc_index = p_seek+1
    t1 = time.time()
    with Pool(num_workers) as processes:
        for doc_counter_list in tqdm(processes.imap_unordered(get_uci_,docs,chunksize=10), total=(5601551-p_seek)/nums_split):
            t_w = time.time()
            for doc_counter in doc_counter_list:
                write_docWord(doc_index, doc_counter, uci_out)
                doc_index += 1
            print("write file time {}".format(time.time()-t_w))
    uci_out.close()
    t2 = time.time()
    tm_cost = t2 - t1
    print("Get uci time is :{}".format(tm_cost))

def get_UCI_dataset_single(tokenized_url,uci_url,vocab_url):
    word2idx = read_vocab(vocab_url)
    total_row = 6145660
    doc_index = 1
    pbar = tqdm(total=total_row)
    uci_file = open(uci_url, 'a+')

    tokenized_file = open(tokenized_url)
    mm = mmap.mmap(tokenized_file.fileno(), 0, access=mmap.ACCESS_READ)

    for doc in iter(mm.readline, b""):
        doc_counter = Counter()
        doc = doc.decode("utf-8")
        doc_tokens = doc.strip().split("\ ")
        if len(doc_tokens) < 5:
            pbar.update(1)
            continue
        doc_tokens = [word2idx[token.strip()] for token in doc_tokens if token in word2idx.keys()]
        if len(doc_tokens) < 5:
            pbar.update(1)
            continue
        doc_counter.update(doc_tokens)
        write_docWord(doc_index,doc_counter,uci_file)
        doc_index += 1
        pbar.update(1)
    pbar.close()
    uci_file.close()
    print("Total doc is {}".format(doc_index))

def pipeline_preprocess_raw(num_worker):
    raw_data_url = '/home/lcl/LightLDA/military_20g/20g/all_docs.txt'
    tokenized_data_url = '/home/lcl/LightLDA/military_20g/20g/all_docs_tokenized.txt'
    vocab_url = '/home/lcl/LightLDA/military_20g/20g/vocab.military20.txt'
    uci_url = '/home/lcl/LightLDA/military_20g/20g/docword.military20_new.txt'
    # 得到分词后的数据集
    #get_tokenized_dataset_mp(num_worker, raw_url=raw_data_url, token_out_url=tokenized_data_url)
    # 得到词表
    #build_vocab_dataset_mp(num_worker,tokenized_url=tokenized_data_url,vocab_url=vocab_url)
    # 得到UCI数据格式数据
    # 6086741
    get_UCI_dataset_single(tokenized_url=tokenized_data_url,uci_url=uci_url,vocab_url=vocab_url)

if __name__ == '__main__':
    pipeline_preprocess_raw(10)

