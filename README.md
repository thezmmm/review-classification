## Directory
- RNN
  - data
    - curTest
    
      此次测试所使用的数据
    
    - curTrain
    
      此次训练所使用的数据
    
    - realResult
    
      此次测试所使用数据的真实结果
    
    - testResult
    
      模型预测curTest生成的结果
    
    - train
    
      训练所使用数据
    
  - model
  
    - gloveModel
  
      词向量模型
  
    - RNN.pkl
  
      使用curTrain训练得到的模型
  
  - datahandle
  
    数据处理
  
  - Initialize
  
    使用train随机生成curTest和curTrain
  
  - model
  
    模型训练
  
  - test
  
    模型预测

## Parameter

- N 

  将数据分成N份进行训练

- RNNClassificationModel

  - lr

    learning rate 学习速率

  - epoches

    训练周期

  - x

    每个训练周期用的数据量

    ```py
    for epoch in range(self.epoches):
        for i in range(x):
    ```

- SeqRNN

  - dropout

    将神经元丢弃的概率，即不进行训练，避免过拟合

- test

  ```python
  for i in range(x):
      onetest = testData[math.floor(i / x * lenOftest):math.floor((i + 1) / x * lenOftest)]
  ```





1. 初始化数据
2. 训练1次，得到测试结果
3. 再初始化数据，重复4-5次，计算出平均值
4. 一次参数调整完成