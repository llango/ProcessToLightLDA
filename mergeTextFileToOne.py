"""
AUthor: Yu Xiubin
Time: 2021-6-8
"""
import os

OutPutPath = "./dataset/military_news.txt"
DatasetDirectory = "/home/yxb/Documents/NLP/DataSet/pdl_qingbao/data/data/"
fileList = [DatasetDirectory + filename for filename in os.listdir(DatasetDirectory)]
txtFileList = [file for file in fileList if file.split('.')[-1] == "txt"]
allArticle = []
sorted(txtFileList)
for file in txtFileList:
    f = open(file, 'r')
    context = ""
    line = f.readline().strip()
    while line:
        context += line
        line = f.readline().strip()
    f.close()
    allArticle.append(context)


# 写文件
f = open(OutPutPath,'w')
for article in allArticle:
    f.write(str(article)+'\n')
f.close()