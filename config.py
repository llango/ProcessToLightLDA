# 根据词性筛选词
NOT_USE_fLAG = ['x', 'xc', 'u', 'uj', 'd', 'ul', 'w', 'p', 'q', 'c', 'r', 'k', 'm', 'y', 'un', 't', "TIME"]
# 去除标签
WORD_LABEL = ["战略态势 外交舆论", "战略态势 政治", "战略态势 军事科技", "战略态势 经济金融"]

# Test Doc Path
TestDatapath = "./new_text.txt"
# Libsvm file OutPath
TestDocWordLibsvmPath = "./test_Out/docword.libsvm"
TestVocabLibsvmPath = "./test_Out/vocab.word_id.dict"
# 训练好的模型文件以及dump_binary(LightLDA Block) out path
BinaryOutPath = "./model_block_path"
LibSvmVocabPath = "docs/millitary.word_id.dict"

# LightLDA BIN Files PATH
LightLdaBinPath = './lightLDABin/'
# TrainDataSet Vocab File
TrainVocabPath = "dataset/vocab.millitaryNesResult.txt"
# super parameters
TopicK = 20
