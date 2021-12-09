import os

import numpy as np
import copy
from config import TopicK


class LDAResult(object):

    def __init__(self, alpha, beta, topic_num, vocab_num, doc_num):
        # 每篇文档词（主题）出现的次数
        self.topic_cnt = []
        self._doc_cnts = []
        self._alpha = alpha
        self._beta = beta
        self._topic_num = topic_num
        self._vocab_num = vocab_num
        self._doc_num = doc_num
        self._doc_topic_mat = np.zeros((self._doc_num, self._topic_num))
        self._word_topic_mat = np.zeros((self._vocab_num, self._topic_num))
        self.vocabs_list = []

    def LoadDocTopicModel(self, model_path):
        f = open(model_path, 'r')
        read_data = f.readline()
        while read_data:
            doc_info = read_data.split()
            # 文档号
            doc_index = doc_info[0]
            doc_index = int(doc_index)
            # 处理后面的主题与词频
            for i in range(1, len(doc_info)):
                topic_index, topic_count = tuple(doc_info[i].split(':'))
                topic_index = int(topic_index)
                topic_count = int(topic_count)
                self._doc_topic_mat[doc_index, topic_index] = topic_count
            read_data = f.readline()
        f.close()
        # 记录每篇文档主题出现的总数，既包含词的数目
        for i in range(self._doc_num):
            self._doc_cnts.append(np.sum(self._doc_topic_mat[i]))
        # 计算概率
        factor = self._topic_num * self._alpha
        for i in range(self._doc_num):
            for j in range(self._topic_num):
                self._doc_topic_mat[i][j] = (self._doc_topic_mat[i][j] + self._alpha) / (self._doc_cnts[i] + factor)
        return self._doc_topic_mat

    def LoadWordTopicModel(self, model_dir, summary_path, server_nums=1):
        """
        处理词-主题矩阵
        :param server_nums: 模型的数量 分布式最后形成模型的数量
        :param model_dir: 模型路径
        :param summary_path: 记录主题总次数的文件
        :return:
        """
        model_path_list = []
        for index in range(server_nums):
            single_model = os.path.join(model_dir, 'server_{}_table_0.model'.format(index))
            model_path_list.append(single_model)

        for model_path in model_path_list:
            f = open(model_path, 'r')
            read_data = f.readline()
            while read_data:
                word_info = read_data.split()
                # 文档号
                word_index = word_info[0]
                word_index = int(word_index)
                # 处理后面的主题与词频
                for i in range(1, len(word_info)):
                    topic_index, topic_count = tuple(word_info[i].split(':'))
                    topic_index = int(topic_index)
                    topic_count = int(topic_count)
                    self._word_topic_mat[word_index, topic_index] = topic_count
                read_data = f.readline()
            f.close()

        self.origin_word_topic = copy.deepcopy(self._word_topic_mat)
        # 读入每个主题出现的总次数
        model_summary_path_list = []
        for index in range(server_nums):
            single_model_summary = os.path.join(model_dir, 'server_{}_table_1.model'.format(index))
            model_summary_path_list.append(single_model_summary)

        for summary_path in model_summary_path_list:
            f = open(summary_path, 'r')
            read_data = f.readline()
            summary_info = read_data.split()
            for i in range(1, len(summary_info)):
                _, topic_count = tuple(summary_info[i].split(':'))
                topic_count = int(topic_count)
                self.topic_cnt.append(topic_count)
            f.close()
        # 计算概率
        self.count_word_topic_p()

    def count_word_topic_p(self):
        # 计算概率
        factor = self._vocab_num * self._beta
        for i in range(self._vocab_num):
            for j in range(self._topic_num):
                self._word_topic_mat[i][j] = (self._word_topic_mat[i][j] + self._beta) / (self.topic_cnt[j] + factor)

    def getTopicTopWordN(self, n):
        topic_nWordList = []
        for j in range(self._topic_num):
            max_list = np.argsort(self._word_topic_mat[:, j])[::-1][:n]
            topic_nWordList.append(max_list)
        return topic_nWordList

    def getVocabList(self, vocab_path):
        f = open(vocab_path, 'r', encoding='utf-8')
        read_data = f.readline()
        while read_data:
            self.vocabs_list.append(read_data.split('\n')[0])
            read_data = f.readline()
        f.close()
        return self.vocabs_list

    def dumpTopicWord(self, vocab_path, output_path, topNum):
        word_list = self.getVocabList(vocab_path)
        topic_nWordList = self.getTopicTopWordN(topNum)
        # topic_nWordList =[[1,2,3],[3,2,1],[50,7,99],[4,55,77]]
        print("writing....")
        f = open(output_path, 'a', encoding='utf-8')
        for i in range(len(topic_nWordList)):
            write_line = str(i) + "  "
            for j in range(len(topic_nWordList[i])):
                write_line = write_line + " " + word_list[topic_nWordList[i][j]]
            print(write_line)
            f.write(write_line + '\n')
        f.close()

    def getTopicWord(self, vocab_path, topNum):
        word_list = self.getVocabList(vocab_path)
        topic_nWordList = self.getTopicTopWordN(topNum)
        # topic_nWordList =[[1,2,3],[3,2,1],[50,7,99],[4,55,77]]
        topic_TopWords = {}
        for i in range(len(topic_nWordList)):
            topic_TopWords[i] = []
            for j in range(len(topic_nWordList[i])):
                topic_TopWords[i].append(word_list[topic_nWordList[i][j]])
        return topic_TopWords

    def get_word_topic(self, query):
        if len(self.vocabs_list) == 0:
            return -1
        if query in self.vocabs_list:
            query_index = self.vocabs_list.index(query)
            query_topic = self.origin_word_topic[query_index, :]
            return np.argsort(query_topic)[-1]
        else:
            return -1

    def cos_sim(self, x, y):
        """
        calulate two doc x and y 's cosine similarity
        :param x:doc x's index
        :param y:doc y's index
        :return:cosine similarity
        """
        x_topic = self._doc_topic_mat[x, :]
        y_topic = self._doc_topic_mat[y, :]
        return np.dot(x_topic, y_topic) / np.linalg.norm(x_topic) * np.linalg.norm(y_topic)

    def get_max_sim(self, x):
        """
        calulate for x search max similarity doc y
        :param x:query
        :return:the max sim index
        """
        index = -1
        max_value = -1.0
        for i in range(self._doc_num):
            if i != x:
                sim = self.cos_sim(x, i)
                if sim > max_value:
                    max_value = sim
                    index = i
        return index


if __name__ == '__main__':
    doc_topic_path = "millitaryNesResult/result_K_20_New2/doc_topic.0"
    topic_summary = "millitaryNesResult/result_K_20_New2/server_0_table_1.model"
    ori_word_path = "dataset/vocab.military.txt"
    output = "millitaryNesResult/result_K_20_New2/res_100.txt"
    ldaResult = LDAResult(0.1, 0.01, TopicK, 4297, 1141)
    # ldaResult.LoadDocTopicModel(doc_topic_path)
    # TODO: 这里需要传入存放所有模型文件的文件夹路径
    model_dir = "millitaryNesResult/result_K_20_New2/"
    print("Loading DocTopic finished!")
    ldaResult.LoadWordTopicModel(model_dir, topic_summary, server_nums=1)
    print("Loading WordTopic finished!")
    TopNum = 100
    ldaResult.dumpTopicWord(ori_word_path, output, TopNum)
