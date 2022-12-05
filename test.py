import random

import joblib
import os

import torch
from torch import nn

import datahandle
import gensim
import math
import pandas as pd


def test():
    dirs = './model'
    rnn = joblib.load(dirs + '/RNN.pkl')
    gensim_file = './model/gloveModel/glove_model.txt'
    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)
    testData = datahandle.getTestData()
    lenOftest = len(testData)
    testResult = []
    for i in range(10):
        onetest = testData[math.floor(i / 10 * lenOftest):math.floor((i + 1) / 10 * lenOftest)]
        test1 = datahandle.embeddingSeq(model, onetest)
        result = rnn.predict(test1)
        testResult = testResult + result

    df = pd.DataFrame(testResult)
    df.to_csv("./data/testResult.csv")
    print('over')

def accuracy():
    real_result = pd.read_csv('./data/realResult.csv')
    test_result = pd.read_csv('./data/testResult.csv')
    real_result = real_result['0'].values
    test_result = test_result['0'].values
    length = len(real_result)
    sum = 0
    diff = 0
    for i in range(length):
        if real_result[i] == test_result[i]:
            sum += 1
        diff += math.fabs(real_result[i]-test_result[i])
    print("accuracy: "+str(sum/length))
    # difference(abs(real-test)/len) in wrong prediction
    print("difference: " + str(diff/(length*(1-sum/length))))

# model definition
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


test()
accuracy()