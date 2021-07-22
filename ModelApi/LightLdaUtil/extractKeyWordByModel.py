import jieba.analyse as analyse
from ModelApi.LightLdaUtil.processRrsultForLightLDA import LDAResult
from ModelApi.LightLdaUtil.config import BinaryOutPath, NOT_USE_fLAG, TrainVocabPath, TopicK, StopWordsPath


def read_stopWords(stopWordsPath=StopWordsPath):
    f = open(stopWordsPath, encoding='utf-8')
    stop_list = []
    line = f.readline().strip()
    while line:
        stop_list.append(line)
        line = f.readline().strip()
    f.close()
    return set(stop_list)


def get_token_list_jieba(context, stopWords=None):
    result = analyse.extract_tags(context, topK=70, withWeight=True, allowPOS=())
    # filter by part of speech and importance
    below_weight = result[len(result) // 2][1] / 1.5
    # print(below_weight)
    result = [word for word, weight in result if weight > below_weight]
    # del stopWords and len(word)==1
    token_list = [word for word in result if word not in stopWords and len(word) > 1]
    return token_list


def extract_doc_word_topic(word_list, lda_result, topics_word_set):
    words_topic = {}
    for word in word_list:
        topic_index = lda_result.get_word_topic(word)
        if topic_index == -1 or word not in topics_word_set:
            continue
        if topic_index in words_topic.keys():
            words_topic[topic_index].add(word)
        else:
            words_topic[topic_index] = set([word])
    result = sorted(words_topic.items(), key=lambda kv: len(kv[1]), reverse=True)
    return result


stop_list = read_stopWords()


def run(corpus_dict, vob_num=4297, doc_num=1141, K=TopicK, stop_list_=stop_list, model_path=BinaryOutPath):
    lda_result = LDAResult(0.1, 0.01, K, vob_num, doc_num)
    lda_result.LoadWordTopicModel(model_path + '/server_0_table_0.model', model_path + '/server_0_table_1.model')
    # get topic:top 100 word
    topic_top_word = lda_result.getTopicWord(TrainVocabPath, 100)
    list_tmp = []
    for topic_index in topic_top_word.keys():
        list_tmp += topic_top_word[topic_index]
    docs_word_topic_set = set(list_tmp)
    docs_result = {}
    for key in corpus_dict.keys():
        im_world_list = get_token_list_jieba(corpus_dict[key], stop_list_)
        result = extract_doc_word_topic(im_world_list, lda_result, docs_word_topic_set)
        if key not in docs_result.keys():
            docs_result[key] = ""
        docs_result[key] = (str(result))
    return docs_result


if __name__ == '__main__':
    text_list = {index: line for index, line in enumerate(open('new_text.txt', 'r', encoding='utf-8'))}
    print(type(text_list))
    # getTokenList("根据报道，二战结束后，美国在几年时间内陆续派德特里克堡基地细菌战专家前往日本，向“731部队”主要成员了解日本细菌战情况。为了得到“731部队”细菌战数据资料，美支付了25"
    #             "万日元，甚至向世界隐瞒“731部队”主要头目的滔天罪行，并让其成为德特里克堡的生物武器顾问。德特里克堡基地正是在此基础上快速发展成为美国生物武器研发基地", stop_list)
    print(run(text_list))
