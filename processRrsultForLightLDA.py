import numpy as np


class LDAResult(object):

    def __init__(self, alpha, beta, topic_num, vocab_num, doc_num):
        # 每篇文档词（主题）出现的次数
        self._doc_cnts = []
        self._alpha = alpha
        self._beta = beta
        self._topic_num = topic_num
        self._vocab_num = vocab_num
        self._doc_num = doc_num
        self._doc_topic_mat = np.zeros((self._doc_num, self._topic_num))
        self._word_topic_mat = np.zeros((self._vocab_num, self._topic_num))

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

    def LoadWordTopicModel(self, model_path, summary_path):
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
        # 读入每个主题出现的总次数
        topic_cnt = []
        f = open(summary_path, 'r')
        read_data = f.readline()
        summary_info = read_data.split()
        for i in range(1, len(summary_info)):
            _, topic_count = tuple(summary_info[i].split(':'))
            topic_count = int(topic_count)
            topic_cnt.append(topic_count)
        f.close()
        # 计算概率
        factor = self._vocab_num * self._beta
        for i in range(self._vocab_num):
            for j in range(self._topic_num):
                self._word_topic_mat[i][j] = (self._word_topic_mat[i][j] + self._beta) / (topic_cnt[j] + factor)

    def getTopicTopWordN(self, n):
        topic_nWordList = []
        for j in range(self._topic_num):
            max_list = np.argsort(self._word_topic_mat[:, j])[::-1][:n]
            topic_nWordList.append(max_list)
        return topic_nWordList

    def getVocabList(self, vocab_path):
        f = open(vocab_path, 'r')
        word_list = []
        read_data = f.readline()
        while read_data:
            word_list.append(read_data.split('\n')[0])
            read_data = f.readline()
        f.close()
        return word_list

    def dumpTopicWord(self, vocab_path, output_path,topNum):
        word_list = self.getVocabList(vocab_path)
        topic_nWordList = self.getTopicTopWordN(topNum)
        # topic_nWordList =[[1,2,3],[3,2,1],[50,7,99],[4,55,77]]
        print("writing....")
        f = open(output_path, 'a')
        for i in range(len(topic_nWordList)):
            write_line = str(i) + "  "
            for j in range(len(topic_nWordList[i])):
                write_line = write_line + " " + word_list[topic_nWordList[i][j]]
            print(write_line)
            f.write(write_line + '\n')
        f.close()


if __name__ == '__main__':
    doc_topic_path = "millitaryNesResult/result_K_4_Less/doc_topic.0"
    topic_word_path = "millitaryNesResult/result_K_4_Less/server_0_table_0.model"
    topic_summary = "millitaryNesResult/result_K_4_Less/server_0_table_1.model"
    ori_word_path = "./dataset/vocab.military.txt"
    output = "millitaryNesResult/result_K_4_Less/res_100.txt"
    ldaResult = LDAResult(0.1, 0.01, 4, 4339, 1140)
    # ldaResult.LoadDocTopicModel(doc_topic_path)

    print("Loading DocTopic finished!")
    ldaResult.LoadWordTopicModel(topic_word_path, topic_summary)
    print("Loading WordTopic finished!")
    TopNum =100
    ldaResult.dumpTopicWord(ori_word_path, output, TopNum)
