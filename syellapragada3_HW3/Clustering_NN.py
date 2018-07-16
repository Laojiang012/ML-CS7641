from sklearn.cluster import KMeans as kmeans
import pandas as pd
import NeuralNetwork as nn
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cross_validation import train_test_split


def NN_test(dataset):
    trainData, testData = train_test_split(dataset, test_size=0.3)
    return nn.neuralNetTest(trainData[:, :-1], trainData[:, -1], testData[:, :-1], testData[:, -1])

def crossValidation(dataset, data = "Spam"):
    trainData, testData = train_test_split(dataset, test_size=0.3)
    val = nn.neuralNetCrossValidation(trainData.iloc[:, :-1], trainData.iloc[:, -1], data=data)
    return val

def clustering(dataset, data = "Spam"):
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
    cross_val_acc_km = []
    cross_val_acc_gmm = []
    km = kmeans(random_state=5)
    gmm = GMM(random_state=5)
    for size in clusters:
        km.set_params(n_clusters=size)
        transformed_X = pd.DataFrame(km.fit_transform(X))

        cluster_indices = pd.DataFrame(km.predict(X))

        transformed_X["Cluster Index"] = cluster_indices
        # transformed_X.to_csv("Datasets/"+data+"_Clustering_KMeans.csv", index=False)
        transformed_X["Y"] = Y
        print size
        cross_val_acc_km += [crossValidation(transformed_X, data = data)]

        gmm.set_params(n_components=35)

        transformed_X = pd.DataFrame(gmm.fit(X).predict_proba(X))

        cluster_indices = pd.DataFrame(gmm.predict(X))

        transformed_X["Cluster Index"] = cluster_indices
        # transformed_X.to_csv("Datasets/" + data + "_Clustering_EM.csv", index=False)
        cross_val_acc_gmm += [crossValidation(transformed_X, data = data)]
    return cross_val_acc_km, cross_val_acc_gmm

def plotGraph(cross_val_acc_km, cross_val_acc_gmm, data="Spam"):
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    x = "Number of clusters"
    y = "Cross Validation Accuracy"
    clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
    cross_val_acc_km.plot(color=colors, label='rbf', marker='o', linestyle=':')
    plt.ylabel(y)
    plt.xlabel(x)
    plt.xticks(cross_val_acc_km.index.values, clusters)
    plt.savefig(data + "/" + data + "_km_nn_CrossValidation", bbox_inches='tight')
    plt.close()

    cross_val_acc_gmm.plot(color=colors, label='rbf', marker='o', linestyle=':')
    plt.ylabel(y)
    plt.xlabel(x)
    plt.xticks(cross_val_acc_gmm.index.values, clusters)
    plt.savefig(data + "/" + data + "_em_nn_CrossValidation", bbox_inches='tight')
    plt.close()

def analysis(data="Spam"):
    km = pd.DataFrame(columns=["Original", "PCA", "ICA", "RP", "RF"])
    gmm = pd.DataFrame(columns=["Original", "PCA", "ICA", "RP", "RF"])
    dataset = pd.read_csv("Datasets/" + ("spambase.csv" if data == "Spam" else "sat-data.csv"), header=None,
                          delimiter=",")
    cross_val_acc_km, cross_val_acc_gmm = clustering(dataset, data=data)
    km["Original"] = cross_val_acc_km
    gmm["Original"] = cross_val_acc_gmm
    types = ["PCA", "ICA", "RP", "RF"]
    for type in types:
        print data + " " + type
        dataset = pd.read_csv("Datasets/" + data + "_DR_" + type + ".csv", delimiter=",")
        cross_val_acc_km, cross_val_acc_gmm = clustering(dataset, data=data)
        km[type] = cross_val_acc_km
        gmm[type] = cross_val_acc_gmm

    plotGraph(km, gmm, data=data)

datas = ["Sat", "Spam"]
for data in datas:
    analysis(data=data)