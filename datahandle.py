# -*- coding: utf-8 -*-

import gensim
import os
import shutil
from sys import platform


# 计算行数，就是单词数
import pandas as pd
from sklearn.model_selection import train_test_split


def getFileLineNums(filename):
    f = open(filename, 'r', encoding='UTF-8')
    count = 0
    for line in f:
        count += 1
    return count


# Linux或者Windows下打开词向量文件，在开始增加一行
def prepend_line(infile, outfile, line):
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    with open(infile, 'r', encoding='UTF-8') as fin:
        with open(outfile, 'w', encoding='UTF-8') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)


def load(filename):
    num_lines = getFileLineNums(filename)
    gensim_file = './model/gloveModel/glove_model.txt'
    gensim_first_line = "{} {}".format(num_lines, 300)
    #     Prepends the line.
    if platform == "linux" or platform == "linux2":
        prepend_line(filename, gensim_file, gensim_first_line)
    else:
        prepend_slow(filename, gensim_file, gensim_first_line)

    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)


#    model.save('./model')
#    print(model['unk'])

# load('./model/gloveModel/glove.6B.300d.txt')
data_train = pd.read_csv('./data/curTrain.csv')
data_test = pd.read_csv('./data/curTest.csv')
def getTrainData():
    sentences = data_train['Phrase']
    labels = data_train['Sentiment']
    return sentences,labels
def getTestData():
    sentences = data_test['Phrase'].values
    labels = data_test['Sentiment'].values
    df = pd.DataFrame(labels)
    df.to_csv('./data/realResult.csv')
    return sentences

def embeddingSeq(model, sentences):
    all_sentences = []
    #    sentence_embedding = []
    for words in sentences:
        words = words.lower()
        #        print(words)
        word = words.split(' ')
        sentence_embedding = []
        for key in word:
            if key not in model:
                emword = model.get_vector('unk')  # 若没有这个词则使用unk的向量，一开始想过置为全0的但又感觉不合适
            else:
                emword = model.get_vector(key)
            sentence_embedding.append(emword)

        all_sentences.append(sentence_embedding)

    return all_sentences
#    print(np.shape(all_sentences[0][:][:]))


