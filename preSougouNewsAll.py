import jieba.posseg as psg
import re
from config import NOT_USE_fLAG
import copy

data_path = "./dataset/military_news.txt"
out_path_docWord = "./dataset/docword.military.txt"
out_path_vocab = "./dataset/vocab.military.txt"

title_pattern = re.compile("<contenttitle>(.*)</contenttitle>")
context_pattern = re.compile("<content>(.*)</content>")
# 存放所有的词语的词典 word:word_cout
Word_Dict = {}
# 存放文章所有的词，下标为词序号
Word_List = []
# 存放doc的word key:word value:word count in doc
Doc_Word_Dict = {}
# 存放所有文章的dco_dict
Doc_Dict_List = []


def statistics_to_dict(wordList):
    for word in wordList:
        # 过滤掉单个字
        if len(word) >= 1:
            # 文章字典
            word_count = wordList.count(word)
            # 全局的字典
            if word not in Word_Dict.keys():
                Word_Dict[word] = word_count
                Word_List.append(word)
            else:
                Word_Dict[word] += word_count
            # 去掉词频小于等于2的词汇
            if word not in Doc_Word_Dict.keys():
                Doc_Word_Dict[word] = word_count


def write_docWord(docIndex, wordDict, f_docWord=None, delList=None):
    for key in wordDict.keys():
        if key not in delList:
            line = str(docIndex) + " " + str(Word_List.index(key)+1) + " " + str(wordDict[key])
            f_docWord.write(line + '\n')


def write_vocab(wordDict):
    f_vocab = open(out_path_vocab, "w")
    # 根据value排序 按照word的序号输出
    wordDict = sorted(wordDict.items(), key=lambda item: item[1])
    for key in wordDict:
        f_vocab.write(key[0] + '\n')
    f_vocab.close()


def write_vocab_docWord(min_frequency):
    # 从全局词典中删除低频word
    del_word = []
    for word in Word_Dict.keys():
        if Word_Dict[word] < min_frequency:
            del_word.append(word)

    for dword in del_word:
        Word_Dict.pop(dword)
        Word_List.remove(dword)
    print("删除后个数：" + str(len(Word_List)))
    f = open(out_path_docWord, 'w')
    for doc_index, doc_dict in enumerate(Doc_Dict_List):
        write_docWord(doc_index+1, doc_dict, f, del_word)
    f.close()

    f = open(out_path_vocab, 'w')
    for word in Word_List:
        f.write(word + '\n')
    f.close()

def preSoGouNews():
    with open(data_path) as f:
        # 存放标题+内容
        content = ''
        # 文章的数目
        doc_index = 0
        read_data = f.readline()
        while read_data:
            print(str(doc_index) + ":")
            read_data = f.readline()
            title_result = title_pattern.findall(read_data)
            if len(title_result):
                # 存放doc的word key:word value:word count in doc
                doc_word_dict = {}
                # 清空
                content = ''
                doc_index += 1
                content = content + title_result[0]
                continue
            context_result = context_pattern.findall(read_data)
            if len(context_result):
                # 将文章的标题和内容连接
                content = content + ":" + context_result[0]
                word_list = list(psg.cut(content))
                # 筛选出有用的词性的词语
                word_list = [x.word for x in word_list if x.flag not in NOT_USE_fLAG]
                print(word_list)
                # 遍历分词的结果
                # 统计
                statistics_to_dict(word_list)
                # 写入文件docWord
                write_docWord(doc_index, doc_word_dict)
                print(word_list)
        f.close()

    print(doc_index)
    print(len(Word_Dict))
    write_vocab(Word_Dict)


def preMillitaryNews():
    with open(data_path) as f:
        # 文章的数目
        doc_index = 1
        read_data = f.readline()
        while read_data:
            if len(read_data):
                word_list = list(psg.cut(read_data))
                # 筛选出有用的词性的词语
                word_list = [x.word for x in word_list if x.flag not in NOT_USE_fLAG]
                statistics_to_dict(word_list)
                # write_docWord(doc_index, doc_word_dict)
                tmp = copy.deepcopy(Doc_Word_Dict)
                Doc_Dict_List.append(tmp)
                # 存放doc的word key:word value:word count in doc
                Doc_Word_Dict.clear()
                # 清空
                doc_index += 1
            read_data = f.readline()
        f.close()
    print(doc_index)
    print(len(Word_List))
    # print(Doc_Dict_List)
    write_vocab_docWord(2)


if __name__ == '__main__':
    preMillitaryNews()
