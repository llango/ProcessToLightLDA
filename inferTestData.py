'''
Author:YXB
Time:2021
本文件将输入的文件（格式为每一行是一篇文章）
转化成libsvm文件的形式得到XXXX.libsvm和一个词典文件
'''
import jieba.posseg as psg
from config import NOT_USE_fLAG, LightLdaBinPath, BinaryOutPath, TopicK, TestDocWordLibsvmPath, \
    TestDatapath, TestVocabLibsvmPath, TrainVocabPath, LibSvmVocabPath
import os
import numpy as np
# TODO:1.消去一些无用的词 2.消去标签 （做好数据处理） 2.想好模型训练的方法
from processRrsultForLightLDA import LDAResult

word_list = []
# key:count count is in all doc
word_dict = {}
# 存放doc的word key:word value:word count in doc
doc_word_dict = {}


def statistics_to_dict(tokenList):
    """
    将单词列表放到全局词典与文章词典中进行统计
    :param tokenList:文章分词后的列表
    :return:无
    """
    for word in tokenList:
        # 全局的字典
        if word not in word_list:
            print(word)
            # word_list.append(word)
        # 文章字典
        if word not in doc_word_dict.keys() and word in word_list:
            doc_word_dict[word] = tokenList.count(word)
            if word not in word_dict.keys():
                word_dict[word] = doc_word_dict[word]
            else:
                word_dict[word] += doc_word_dict[word]


def write_docWord(docIndex, wordDict):
    """
    将序号为docIndex的文章的wordID写入输入文件中
    :param docIndex: 文章序号
    :param wordDict: 文章词典词频统计
    :return:
    """
    f_docWord = open(TestDocWordLibsvmPath, "a+")
    doc = str(docIndex) + '\t'
    for key in wordDict.keys():
        doc += str(word_list.index(key)) + ':' + str(wordDict[key]) + ' '
    f_docWord.write(doc.strip() + '\n')
    f_docWord.close()


def write_vocab(wordDict):
    '''
    输出一个词典每一行为"wordIndex 词 词频"
    :param wordList:
    :return:
    '''
    f_vocab = open(TestVocabLibsvmPath, "w")
    for key in wordDict:
        line = '\t'.join([str(word_list.index(key)), key, str(wordDict[key])]) + "\n"
        f_vocab.write(line)
    f_vocab.close()


def read_dict(dict_path):
    '''
    将词典读入变成list
    :param dict_path: 词典路径
    :return:
    '''
    f_dict = open(dict_path)
    line = f_dict.readline().strip()
    while line:
        word_list.append(line)
        line = f_dict.readline().strip()
    print("获取词库" + str(len(word_list)) + "单词")


def transforLIBSVM():
    # 获取全局词典
    read_dict(TrainVocabPath)
    # 读取停用词

    with open(TestDatapath) as f:
        # 文章的数目,从1开始计数
        doc_index = 0
        read_data = f.readline().strip()
        while read_data:
            if len(read_data):
                token_list = list(psg.cut(read_data))
                # 筛选出有用的词性的词语
                token_list = [x.word for x in token_list if x.flag not in NOT_USE_fLAG]
                # 遍历分词的结果
                # 统计
                statistics_to_dict(token_list)
                # 写入文件docWord
                write_docWord(doc_index, doc_word_dict)
                doc_index += 1
            doc_word_dict.clear()
            read_data = f.readline().strip()
        f.close()

    write_vocab(word_dict)
    return doc_index, len(word_list)


def libsvmTOBinary(exePath, outdir, numblocks=0):
    a = os.system('{dumpBinaryPath} {libsvmPath}   {libsvmVocabPath} {outDir} {blockNum}'.format(dumpBinaryPath=exePath,
                                                                                                 libsvmPath=TestDocWordLibsvmPath,
                                                                                                 libsvmVocabPath=LibSvmVocabPath,
                                                                                                 outDir=outdir,
                                                                                                 blockNum=numblocks))
    return a


def inferByLightLDA(infer_path, block_path, vocab_num, topic_num=378):
    # num_vocabs为训练集的单词
    a = os.system(
        '{inferPath}  -num_vocabs {num_vocabs}  -num_topics {K} -num_iterations 100 -alpha 0.1 -beta 0.01 -mh_steps 2 -num_local_workers 1 -num_blocks 1 -max_num_document 1300000 -input_dir {blockPath}  -data_capacity 8000'.format(
            inferPath=infer_path, blockPath=block_path, K=topic_num, num_vocabs=vocab_num))
    return a


def getTop5(outputs):
    outputs = np.argsort(outputs).tolist()
    # 取后面五个最大的类别
    res = [l[-5:] for l in outputs]
    # 翻转列表
    res = [l[::-1] for l in res]
    return res


if __name__ == '__main__':
    # 将输入转成libsvm格式
    docNum, vocab_num = transforLIBSVM()

    # libsvm转成LightLDA 的block
    libsvmTOBinary(LightLdaBinPath + 'dump_binary', BinaryOutPath)
    # 推理
    inferByLightLDA(LightLdaBinPath + "infer", BinaryOutPath, vocab_num=vocab_num, topic_num=TopicK)
    doc_topic_path = "./doc_topic.0"
    ldaResult = LDAResult(0.1, 0.01, TopicK, vocab_num, docNum)
    result = ldaResult.LoadDocTopicModel(doc_topic_path)
    for i in range(docNum):
        print(ldaResult.get_max_sim(i))
    print(getTop5(result))
