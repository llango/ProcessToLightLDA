#### 1.数据集格式

![img.png](img.png)

clean_merge_all_txt_in_dirs.py 并行清洗各个文件夹下的数据，并将数据集合到一个文件中

所有数据的存放文件为all_docs.txt
共有：5601551 数据行

tokrnizer_by_jieba_mp.py 使用jieba对总的数据进行并行分词