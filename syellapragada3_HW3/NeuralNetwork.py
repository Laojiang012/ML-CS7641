from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import pandas as pd
import time
import matplotlib.pyplot as plt

HIDDEN_LAYERS_SPAM = 15
HIDDEN_LAYERS_SAT = 25

def neuralNetCrossValidation(trainX, trainY, activation='relu', solver="adam", learning_rate="constant", data="Spam"):
    layers = (HIDDEN_LAYERS_SPAM) if data == "Spam" else (HIDDEN_LAYERS_SAT, HIDDEN_LAYERS_SAT, HIDDEN_LAYERS_SAT)
    mlp = MLPClassifier(hidden_layer_sizes=layers, learning_rate_init=0.01, activation=activation, solver=solver, learning_rate=learning_rate)
    scores = cross_val_score(mlp, trainX, trainY, cv=10)
    # score = accuracy_score(statDataY, prediction)
    # print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()

def neuralNetTest(trainX, trainY, testX, testY, activation='relu', solver="adam", learning_rate="constant", data="Spam"):
    start = time.clock()
    out = []
    layers = (HIDDEN_LAYERS_SPAM) if data == "Spam" else (HIDDEN_LAYERS_SAT, HIDDEN_LAYERS_SAT, HIDDEN_LAYERS_SAT)
    mlp = MLPClassifier(hidden_layer_sizes=layers, learning_rate_init=0.01, activation=activation, solver=solver,
                        learning_rate=learning_rate)
    mlp.fit(trainX, trainY)
    train_prediction = mlp.predict(trainX)
    train_score = accuracy_score(trainY, train_prediction)
    print ("Train Accuracy: %0.2f" % train_score)
    out += [train_score]
    time_elapsed = time.clock() - start
    out += [time_elapsed]
    start = time.clock()
    print ("Training Time: %0.2f" % time_elapsed)

    prediction = mlp.predict(testX)
    score = accuracy_score(testY, prediction)
    print ("Test Accuracy: %0.2f" % score)
    print ("Train Accuracy: %0.2f" % train_score)
    out += [train_score]
    time_elapsed = time.clock() - start
    print ("Testing Time: %0.2f" % time_elapsed)
    out += [time_elapsed]
    return out

def runTests(allData=None, data="Spam"):
    train_accuracies = []
    test_accuracies = []
    train_times = []
    test_times = []
    if not allData:
        allData = pd.read_csv("Datasets/" + ("spambase.csv" if data == "Spam" else "sat-data.csv"),header=None, delimiter=",")
    # allData = allData.iloc[:, 1:]
    out = neuralNet(allData, data=data)
    print "___________________"
    train_accuracies += [out[0]]
    train_times += [out[1]]
    test_accuracies += [out[2]]
    test_times += [out[3]]
    types = ["PCA", "ICA", "RP", "RF"]
    for type in types:
        print data + " " + type
        dataset = pd.read_csv("Datasets/" + data + "_DR_" + type + ".csv", delimiter=",")
        out = neuralNet(dataset, data=data)
        train_accuracies += [out[0]]
        train_times += [out[1]]
        test_accuracies += [out[2]]
        test_times += [out[3]]
        print "___________________"
    plotGraphs([train_accuracies, test_accuracies, train_times, test_times], data=data)

def neuralNet(allData, data = "Spam"):
    trainData, testData = train_test_split(allData, test_size=0.3)
    trainX, trainY = trainData.iloc[:, :-1], trainData.iloc[:, -1]
    testX, testY = testData.iloc[:, :-1], testData.iloc[:, -1]
    return neuralNetTest(trainX, trainY, testX, testY, data=data)

def plotGraphs(plotVariables, data="Spam"):
    order = ["Original", "PCA", "ICA", "RP", "RF"]
    ylabels = ["Train Accuracy", "Test Accuracy", "Train Time", "Test Time"]

    for i, variables in enumerate(plotVariables):
        plt.plot(range(5), variables)
        plt.xticks(range(5), order)
        plt.xlabel("Datasets")
        plt.ylabel(ylabels[i])
        plt.savefig(data + "/"+data+ylabels[i])
        plt.close()

if __name__ == "__main__":
    print "Spam"
    runTests(data="Spam")
    print "------------------"

    print "Sat"
    runTests(data="Sat")



