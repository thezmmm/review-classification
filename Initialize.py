import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("./data/train.tsv",sep='\t')
data_train, data_test = train_test_split(data, test_size=0.1)
data_train.to_csv('./data/curTrain.csv')
data_test.to_csv('./data/curTest.csv')