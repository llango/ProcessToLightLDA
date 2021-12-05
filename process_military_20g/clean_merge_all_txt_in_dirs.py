import jieba.posseg as psg
import re
import os
import tqdm
import time
from multiprocessing import Pool
from config import NOT_USE_fLAG, WORD_LABEL
import copy

regix_nosiy_seg = [
    # 去除原创图片信息
    re.compile(r"[\[【]*(.{1,5}图|.{1,5}原创)[\]】]+"),
    # 去除网址
    re.compile(
        r"[https:\/\/|http:\/\/]*(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)"),
    # 特殊字符
    re.compile('[#$%&*@★、…【】]+')
]


def preprocess_articles(doc_str):
    """
    TODO:还需要完善该方法
    使用正则处理清洗一篇文章
    :param doc_str:
    :return:
    """
    for re_nosiy_word in regix_nosiy_seg:
        doc_str = re.sub(re_nosiy_word, '', doc_str)
    return doc_str


def get_all_articles_in_one_txt(txt_file) -> object:
    """
    格式为每一篇文章像个一个空行
    :param txt_file:txt路径
    :return:返回文章列表["sjaojx","sahxoias",...,"sada"]
    """
    docs_list = []
    txt_file = open(txt_file, 'r', encoding='utf-8')
    tmp = ""
    try:
        for line in txt_file:
            line = line.strip().replace('\n', '')
            if len(line) > 0:
                tmp += line
            else:
                docs_list.append(preprocess_articles(tmp))
                tmp = ""
        txt_file.close()
    except:
        txt_file.close()
    return docs_list


def get_all_article_in_one_dir(dir_path, dir_docs):
    """
    获取一个文件下的所有文章
    :param dir_path:
    :return:
    """
    file_list = os.listdir(dir_path)
    file_list = [os.path.join(dir_path, filename) for filename in file_list if filename.endswith("txt")]
    for file_path in tqdm.tqdm(file_list):
        dir_docs += get_all_articles_in_one_txt(file_path)
    print("完成{}!".format(dir_path))


def write_docs_to_file(out_file, docs):
    """
    将一个文章写入文件
    :param out_file:
    :param docs:
    :return:
    """
    out_file.write("\n".join(docs))


def get_all_article_in_all_dirs(num_workers):
    base_dir = '/home/lcl/LightLDA/military_20g/20g/'
    dir_all_list = [
        base_dir + '5k', base_dir + '10k', base_dir + '15k', base_dir + '20k', base_dir + '25k', base_dir + '30k',
        base_dir + '35k', base_dir + 'new'
    ]
    # 一次只能处理三个文件夹
    for s_index in range(0, len(dir_all_list), 3):
        e_index = s_index + 3
        if e_index >= len(dir_all_list):
            e_index = len(dir_all_list)
        dir_list = dir_all_list[s_index:e_index]
        all_file_list = []
        for dir_ in dir_list:
            dir_file_list = os.listdir(dir_)
            dir_file_list = [os.path.join(dir_, filename) for filename in dir_file_list]
            all_file_list += dir_file_list

        s_time = time.time()
        with Pool(num_workers) as processes:
            result = processes.starmap(get_all_articles_in_one_txt, zip(all_file_list))
        e_time = time.time()
        print("Time for precessing:{}".format(e_time - s_time))

        out_all_file = open(base_dir + 'all_docs.txt', 'a+')
        for docs in result:
            write_docs_to_file(out_all_file, docs)
        ee_time = time.time()
        out_all_file.close()
        print("Time for write file:{}".format(ee_time - e_time))


if __name__ == '__main__':
    get_all_article_in_all_dirs(10)
