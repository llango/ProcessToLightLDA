# 根据词性筛选词
import time

NOT_USE_fLAG = ['x', 'xc', 'uj', 'd', 'ul', 'w', 'p', 'q', 'c', 'r', 'k', 'm', 'y', 'un', 't', "TIME"]
# 去除标签
WORD_LABEL = ["战略态势 外交舆论", "战略态势 政治", "战略态势 军事科技", "战略态势 经济金融"]
PROJECT_BASE_PATH = '/home/yxb/MyCoding/WebApp/ModelApi/ModelApi/LightLdaUtil'


# Test Doc Path
TestDatapath = PROJECT_BASE_PATH+"/new_text.txt"
# Libsvm file OutPath
date = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
LibSvmDir = PROJECT_BASE_PATH + f'/testOut/lib_{date}'
TestDocWordLibsvmPath = LibSvmDir + "/docword.libsvm"
TestVocabLibsvmPath = LibSvmDir + "/vocab.word_id.dict"
InferResultDir = PROJECT_BASE_PATH+f'/inferResult/{date}'


# 训练好的模型文件以及dump_binary(LightLDA Block) out path
BinaryOutPath = PROJECT_BASE_PATH+"/modelBlockPath"
LibSvmVocabPath = PROJECT_BASE_PATH+"/docs/millitary.word_id.dict"

StopWordsPath = PROJECT_BASE_PATH+"/dataset/stopWords"

# LightLDA BIN Files PATH
LightLdaBinPath = PROJECT_BASE_PATH+'/lightLDABin/'
# TrainDataSet Vocab File
TrainVocabPath = PROJECT_BASE_PATH+"/dataset/vocab.millitaryNesResult.txt"
# super parameters
TopicK = 20
