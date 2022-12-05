# -*- coding: utf-8 -*-
import joblib
import torch
from torch import nn
import random
import datahandle
import gensim
import math


import os


class SeqRNN(nn.Module):
    '''
    vocab_size:词向量维度
    hidden_size:隐藏单元数量决定输出长度
    output_size:输出类别为5，维数为1
    '''

    def __init__(self, vocab_size, hidden_size, output_size):
        super(SeqRNN, self).__init__()
        self.vocab_size = vocab_size  # 这个为词向量的维数，GLove中为300维
        self.hidden_size = hidden_size  # 隐藏单元数
        self.output_size = output_size  # 最后要输出的

        self.rnn = nn.RNN(self.vocab_size, self.hidden_size, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        h0 = torch.zeros(1, 1, self.hidden_size)
        output, hidden = self.rnn(input, h0)
        output = output[:, -1, :]
        output = self.linear(output)
        output = torch.nn.functional.softmax(output, dim=1)
        return output


class RNNClassificationModel:
    def __init__(self, epoches=100):
        self.model = SeqRNN(300, 128, 5)
        self.epoches = epoches
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.0003)

    def fit(self, trainSet, labels):

        for epoch in range(self.epoches):
            for i in range(200):
                index = random.randint(0, len(labels) - 1)
                sentence = trainSet[index][:][:]
                label = labels[labels.index[index]]
                sentence_tensor = torch.tensor([sentence], dtype=torch.float)
                label_tensor = torch.tensor([label], dtype=torch.long)

                self.optimizer.zero_grad()
                pred = self.model(sentence_tensor)
                loss = self.loss_func(pred, label_tensor)
                loss.backward()

                self.optimizer.step()

    def predict_single(self, sentence):
        sentence_tensor = torch.tensor([sentence], dtype=torch.float)
        with torch.no_grad():
            out = self.model(sentence_tensor)
            out = torch.argmax(out).item()
            return out

    def predict(self, sentences):
        results = []
        for sentence in sentences:
            result = self.predict_single(sentence)
            results.append(result)

        return results

    def scores(self, train, label):
        results = self.predict(train)
        t = 0
        for i in range(len(label)):
            if int(label[i]) == int(results[i]):
                t += 1

        return t / len(label)


if __name__ == '__main__':
    gensim_file = './model/gloveModel/glove_model.txt'
    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)
    sentences, labels = datahandle.getTrainData()
    # param: epoches
    rnn = RNNClassificationModel(100)
    sentences = list(sentences)
    length = len(sentences)
    N = 20
    dataset = []
    labelset = []
    '''
    将数据集分为二十份，然后进行训练
    '''
    for i in range(N):
        onedata = sentences[math.floor(i / N * length):math.floor((i + 1) / N * length)]
        onelabel = labels[math.floor(i / N * length):math.floor((i + 1) / N * length)]
        dataset.append(onedata)
        labelset.append(onelabel)


    for i in range(N):
        print('第' + str(i) + '次')
        train = datahandle.embeddingSeq(model, dataset[i])
        rnn.fit(train, labelset[i])
    dirs = './model'
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    # 保存模型
    joblib.dump(rnn, dirs + '/RNN.pkl')
