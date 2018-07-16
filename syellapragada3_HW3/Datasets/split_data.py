import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split

allData = pd.read_csv('spambase.csv', header=None, delimiter=",")
# allData = allData.iloc[:, 1:]
trainData, testData = train_test_split(allData, test_size=0.3)
trainData.to_csv('spam-train.csv', index=False, header=False)
testData.to_csv('spam-test.csv', index=False, header=False)
# train = trainData.iloc[:, :-1]
# trainY = trainData.iloc[:, -1]
# train["Y"] = trainY
# test = testData.iloc[:, :-1]
# testY = testData.iloc[:, -1]
# test["Y"] = testY


# np.savetxt('spam-test.csv', test_instances, delimiter=',')
# np.savetxt('spam-train.csv', train_instances, delimiter=',')
