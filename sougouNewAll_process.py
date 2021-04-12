import jieba
import jieba.posseg as psg
import re

data_path = "./dataset/sougou_news_utf8.txt"
out_path_docWord = "./dataset/docword.sougou.txt"
out_path_vocab = "./dataset/vocab.sougou.txt"

title_pattern = re.compile("<contenttitle>(.*)</contenttitle>")
context_pattern = re.compile("<content>(.*)</content>")
# 存放所有的词语的词典 key:word value:word_index
word_dict = {}
# 存放doc的word key:word value:word count in doc
doc_word_dict = {}

# x表示符号 u助词 w符号 p介词 q量词
not_use_flAG = ['x', 'uj', 'd', 'ul', 'w', 'p', 'q']


def statistics_to_dict(wordList):
    for word in wordList:
        # 全局的字典
        if word not in word_dict.keys():
            word_dict[word] = len(word_dict) + 1
        # 文章字典
        if word not in doc_word_dict.keys():
            doc_word_dict[word] = wordList.count(word)


def write_docWord(docIndex, wordDict):
    f_docWord = open(out_path_docWord, "a+")
    for key in wordDict.keys():
        line = str(docIndex) + " " + str(word_dict[key]) + " " + str(wordDict[key])
        f_docWord.write(line + '\n')
    f_docWord.close()


def write_vocab(wordDict):
    f_vocab = open(out_path_vocab, "w")
    # 根据value排序 按照word的序号输出
    wordDict = sorted(wordDict.items(), key=lambda item: item[1])
    print(wordDict)
    for key in wordDict:
        f_vocab.write(key[0] + '\n')
    f_vocab.close()


with open(data_path) as f:
    # 存放标题+内容
    content = ''
    # 文章的数目
    doc_index = 0
    read_data = f.readline()
    while read_data:
        print(str(doc_index)+":")
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
            content = content + ":" + context_result[0]
            word_list = list(psg.cut(content))
            # 筛选出有用的词性的词语
            word_list = [x.word for x in word_list if x.flag not in not_use_flAG]
            # 遍历分词的结果
            # 统计
            statistics_to_dict(word_list)
            # 写入文件docWord
            write_docWord(doc_index, doc_word_dict)
            print(word_list)
    f.close()

print(doc_index)
print(len(word_dict))
write_vocab(word_dict)
