import numpy as np
import pandas as pd
from sklearn.svm import SVC
import sklearn.tree as tree
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

def decisionTree(statDataX, statDataY, criterion= "gini", min_samples_split = 2, max_depth = None):
    clf = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split = min_samples_split, max_depth=max_depth)
    scores = cross_val_score(clf, statDataX, statDataY, cv=10)
    # score = accuracy_score(statDataY, prediction)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()


def decisionTreeTest(statDataX, statDataY, testX, testY, criterion= "gini", min_samples_split = 2, max_depth = None):
    clf = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split, max_depth=max_depth)
    clf.fit(statDataX, statDataY)
    prediction = clf.predict(testX)
    score = accuracy_score(testY, prediction)
    print ("Test Accuracy: %0.2f" % score)
    return score


def knn(statDataX, statDataY, weights="uniform", n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    scores = cross_val_score(knn, statDataX, statDataY, cv=10)
    # score = accuracy_score(statDataY, prediction)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return scores.mean()


def knnTest(statDataX, statDataY, testX, testY, weights="uniform", n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=5, weights=weights)
    knn.fit(statDataX, statDataY)
    prediction = knn.predict(testX)
    score = accuracy_score(testY, prediction)
    print ("Test Accuracy: %0.2f" % score)
    return score

def neuralNet(statDataX, statDataY, activation='relu', solver="adam", learning_rate="constant"):
    mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), learning_rate_init=0.01, activation=activation, solver=solver, learning_rate=learning_rate)
    scores = cross_val_score(mlp, statDataX, statDataY, cv=10)
    # score = accuracy_score(statDataY, prediction)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()


def neuralNetTest(statDataX, statDataY, testX, testY, activation='relu', solver="adam", learning_rate="constant"):
    mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), learning_rate_init=0.01, activation=activation, solver=solver,
                        learning_rate=learning_rate)
    mlp.fit(statDataX, statDataY)
    prediction = mlp.predict(testX)
    score = accuracy_score(testY, prediction)
    print ("Test Accuracy: %0.2f" % score)
    return score

def svm(statDataX, statDataY, kernel="rbf", max_iter=-1, c=1):
    clf = SVC(kernel=kernel, C=c, max_iter=max_iter)
    # clf.fit(X, y)
    scores = cross_val_score(clf, statDataX, statDataY, cv=10)
    # score = accuracy_score(statDataY, prediction)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()


def svmTest(statDataX, statDataY, testX, testY, kernel="rbf", max_iter=-1, c=1):
    clf = SVC(kernel=kernel, C=c, max_iter=max_iter)
    clf.fit(statDataX, statDataY)
    prediction = clf.predict(testX)
    score = accuracy_score(testY, prediction)
    print ("Test Accuracy: %0.2f" % score)
    return score

def boosting(statDataX, statDataY, n_estimators=100, learning_rate=1, algorithm="SAMME.R", criterion="entropy", min_samples_split=20, max_depth=20):
    dt = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split = min_samples_split, max_depth=max_depth)
    clf = AdaBoostClassifier(n_estimators=n_estimators, base_estimator=dt, learning_rate=learning_rate, algorithm=algorithm)
    scores = cross_val_score(clf, statDataX, statDataY, cv=10)
    # score = accuracy_score(statDataY, prediction)
    print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()


def boostingTest(statDataX, statDataY, testX, testY, n_estimators=100, learning_rate=1, algorithm="SAMME", criterion="entropy", min_samples_split=20, max_depth=20):
    dt = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split, max_depth=max_depth)
    clf = AdaBoostClassifier(n_estimators=n_estimators, base_estimator=dt, learning_rate=learning_rate,
                             algorithm=algorithm)
    clf.fit(statDataX, statDataY)
    prediction = clf.predict(testX)
    score = accuracy_score(testY, prediction)
    print ("Test Accuracy: %0.2f" % score)
    return score

def plotGraph(title, trainScores, testScores, trainSizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_scores_mean = np.mean(trainScores, axis=1)
    train_scores_std = np.std(trainScores, axis=1)
    test_scores_mean = np.mean(testScores, axis=1)
    test_scores_std = np.std(testScores, axis=1)
    plt.grid()

    plt.fill_between(trainSizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(trainSizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(trainSizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(trainSizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# def crossValidation(X, Y, algoType=1, criterion= "gini", min_samples_split = 2, max_depth = None, knnWeights="uniform"):
#     avgAcc = 0
#     avgAcc1 = 0
#     for train_index, test_index in kf.split(X, y=Y):
#         # print("TRAIN:", train_index, "TEST:", test_index)
#         trainX = X.iloc[train_index.tolist(), :]
#         trainY = Y.iloc[train_index.tolist()]
#         testX = X.iloc[test_index.tolist(), :]
#         testY = Y.iloc[test_index.tolist()]
#         if algoType == 1:
#             acc = decisionTree(trainX, trainY, testX, testY, criterion=criterion, min_samples_split = min_samples_split, max_depth = max_depth)
#             avgAcc += acc
#         elif algoType == 2:
#             acc1 = knn(trainX, trainY, testX, testY, weights=knnWeights)
#             avgAcc1 += acc1
#
#     avgAcc = avgAcc / 10
#     avgAcc1 = avgAcc1/10
#     print "DT accuracy: {.3f}".format(avgAcc)
#     print "KNN accuracy: {.3f}".format(avgAcc1)

def pruningPlots(name="Decision Tree", xAxis="", yAxis = "Accuracy", xValues=[], yValues=[], xTicks = None, color="b", label="Cross Validation Score"):
    plt.figure()
    plt.title(name)
    plt.xlabel(xAxis)
    plt.ylabel(yAxis)
    plt.grid()
    if xTicks:
        plt.xticks(xValues, xTicks)
    plt.plot(xValues, yValues, label=label, color = color)
    plt.legend(loc="best")
    # plt.show()
    plt.savefig(name+ "-"+xAxis)

if __name__ == "__main__":
    # Loading data set
    print 'reading data'

    # allData = pd.read_csv('spambase.data', header=None, delimiter=",")
    # allData = allData.iloc[:, 1:]
    # trainData, testData = train_test_split(allData, test_size=0.25, shuffle=True)
    # X = trainData.iloc[:, :-1]
    # Y = trainData.iloc[:, -1]
    # testX = testData.iloc[:, :-1]
    # testY = testData.iloc[:, -1]

    statData = pd.read_csv('sat-trn.csv', header=None, delimiter=" ")
    testData = pd.read_csv('sat-tst.csv', header=None, delimiter=" ")

    headers = []
    for i in range(36):
        headers += ["S"+str(i/9 +1)+"P"+str(i%9 + 1)]

    headers += ["Soil Type"]
    statData.columns = headers

    X = statData.iloc[:, :-1]
    Y = statData.iloc[:, -1]
    testX = testData.iloc[:, :-1]
    testY = testData.iloc[:, -1]

    testAcc = []
    print "DT\n"
    acc = []
    xValues = []
    for i in range(2, 110, 10):
        xValues += [i]
        acc += [decisionTree(X, Y, criterion="entropy", min_samples_split=i, max_depth=20)]

    pruningPlots("Decision Tree Pruning", xAxis="Minimum samples split", xValues=xValues, yValues=acc)

    acc = []
    xValues = []
    for i in range(20, 110, 10):
        xValues += [i]
        acc += [decisionTree(X, Y, criterion="entropy", min_samples_split=80, max_depth=i)]

    pruningPlots("Decision Tree Pruning", xAxis="Maximum depth", xValues=xValues, yValues=acc)

    testAcc += [decisionTreeTest(X, Y, testX, testY,criterion="entropy", min_samples_split=55, max_depth=20)]

    print "\nKNN\n"

    acc = []
    xValues = []
    for i in range(3, 20):
        xValues += [i]
        acc += [knn(X, Y, weights="distance", n_neighbors=i)]

    pruningPlots("K- Nearest Neighbours Pruning", xAxis="Neighbours Count", xValues=xValues, yValues=acc, color="g")
    knn(X, Y)
    knn(X, Y, weights="distance", n_neighbors=3)
    knn(X, Y, weights="distance", n_neighbors=5)
    knn(X, Y, weights="distance", n_neighbors=10)

    testAcc += [knnTest(X, Y, testX, testY, weights="distance", n_neighbors=6)]

    print "\nNN\n"

    act = ["relu", "logistic"]
    solver = ["sgd", "lbfgs", "adam"]
    learningRate = ["constant", "adaptive", "invscaling"]
    acc = []
    for i in range(3):
        acc += [neuralNet(X, Y, activation='relu', solver=solver[i], learning_rate="constant")]

    pruningPlots("Artificial Neural Networks (activation = relu, learning rate = constant)", xAxis="Solver", xValues=range(3), yValues=acc, xTicks=solver, color="r")
    acc = []
    for i in range(3):
        acc += [neuralNet(X, Y, activation='relu', solver="sgd", learning_rate=learningRate[i])]

    pruningPlots("Artificial Neural Networks (activation =relu, solver = adam)", xAxis="Learning Rate", xValues=range(3), yValues=acc, xTicks=learningRate, color="r")

    testAcc += [neuralNetTest(X, Y, testX, testY, activation='relu', solver="adam", learning_rate="constant")]

    print "\nSVM\n"
    kernels = ["rbf", "sigmoid", "linear"]
    acc = []
    for i in range(3):
        acc += [svm(X, Y, kernel=kernels[i], max_iter=20000)]
    pruningPlots("Support Vector Machines (Penalty parameter(C) = 1)", xAxis="Kernels", xValues=range(3), yValues=acc, xTicks=kernels, color="c")

    acc=[]
    for i in range(2, 10, 2):
        acc += [svm(X, Y, kernel="rbf", c=i, max_iter=20000)]
    pruningPlots("Support Vector Machines (Kernel = rbf)", xAxis="Penalty Parameter(C)", xValues=range(2, 10, 2), yValues=acc, color="c")

    testAcc += [svmTest(X, Y, testX, testY, kernel="rbf", c=6, max_iter=20000)]

    print "\nBoosting\n"
    acc =[]
    for i in range(50, 110, 10):
        acc += [boosting(X, Y, n_estimators=i, learning_rate=1.0)]

    pruningPlots("Boosting (learning rate = 1)", xAxis="nestimators", xValues=range(50, 100, 10),
                 yValues=acc, color="y")

    acc =[]
    xValues = []
    for i in range(1 * 2, 5 * 2):
        xValues += [0.5*i]
        acc += [boosting(X, Y, n_estimators=100, learning_rate=0.5*i)]

    pruningPlots("Boosting (n estimators = 100)", xAxis="Learning Rate", xValues=xValues,
                     yValues=acc, color="y")

    testAcc += [boostingTest(X, Y, testX, testY, n_estimators=100, learning_rate=1)]

    pruningPlots("Comparing different algorithms", xAxis="Algorithms", xValues=range(5), yValues=testAcc, xTicks=["DT", "KNN", "NN", "SVM", "Boosting"], label="Test Accuracy")