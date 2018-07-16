from sklearn.cluster import KMeans as kmeans
import pandas as pd
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
import plotly.plotly as py

import helpers
from sklearn.metrics import homogeneity_score as hs, silhouette_score as ss, completeness_score as cs

def clustering(dataset, data = "Spam", type=None):
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]
    clusters = [2,5,10,15,20,25,30,35,40]
    kmDF = pd.DataFrame(columns=["Cluster size", "Silhouette Score", "Completeness Score", "Homogeneity Score"])
    gmmDF = pd.DataFrame(columns=["Cluster size", "Silhouette Score", "Completeness Score", "Homogeneity Score"])

    km = kmeans(random_state=5)
    gmm = GMM(random_state=5)
    for size in clusters:
        km.set_params(n_clusters = size)
        km.fit(X)

        gmm.set_params(n_components=size)
        gmm.fit(X)

        kmValues = [size, ss(X, km.predict(X)), cs(Y, km.predict(X)), hs(Y, km.predict(X))]
        gmValues = [size, ss(X, gmm.predict(X)), cs(Y, gmm.predict(X)), hs(Y, gmm.predict(X))]

        temp = pd.DataFrame([kmValues], columns=["Cluster size", "Silhouette Score", "Completeness Score", "Homogeneity Score"])

        kmDF = pd.concat([kmDF, temp])

        temp = pd.DataFrame([gmValues], columns=["Cluster size", "Silhouette Score", "Completeness Score", "Homogeneity Score"])
        gmmDF = pd.concat([gmmDF, temp])

    kmDF.to_csv(data + "/kmeans/Clustering"+ ("_"+type if type is not None else "")+".csv", index=False)
    gmmDF.to_csv(data + "/EM/Clustering"+ ("_"+type if type is not None else "")+".csv", index=False)

def plotGraphs(data="Spam", clustering="kmeans", type=None, vline=True):
    file = pd.read_csv(data+"/"+ clustering+"/Clustering"+(("_"+type) if type else "")+ ".csv")
    #
    # file.plot(x=file.index, y='Silhouette Score', style='r:',  marker='o')
    # plt.xticks(file.index, file['Cluster size'])
    # plt.xlabel('Cluster size')
    # plt.savefig(data + "/kmeans/" +(type if type is not None else "")+"_Silhouette Score")
    # plt.close()
    #
    # file.plot(x=file.index, y='Completeness Score', style=':',  marker='o')
    # plt.xticks(file.index, file['Cluster size'])
    # plt.xlabel('Cluster size')
    # plt.savefig(data + "/"+ clustering+"/" +(type if type is not None else "")+"_Completeness Score")
    # plt.close()
    #
    # file.plot(x=file.index, y='Homogeneity Score', style='g:',  marker='o')
    # plt.xticks(file.index, file['Cluster size'])
    # plt.xlabel('Cluster size')
    # plt.savefig(data + "/"+ clustering+"/" +(type if type is not None else "")+"_Homogeneity Score")
    # plt.close()

    file.plot(x=file.index, y=['Silhouette Score','Completeness Score','Homogeneity Score'], style=['r:', ':', 'g:'], marker='o')
    plt.xticks(file.index, file['Cluster size'])
    plt.xlabel('Number of clusters')
    if vline:
        plt.axvline(x=(0 if data == "Spam" else 1.5), color='y', linestyle='dashed', linewidth=5)
    plt.savefig(data + "/"+ clustering+"/" +data + "_" +("EM_" if clustering is not "kmeans" else "")+(type if type is not None else "")+"_Combined")
    plt.close()

def clusterGraph(dataset, algo_type="kmeans", data="Spam", cluster_size=2, type=None):
    algo = kmeans(random_state=5, n_clusters=cluster_size) if algo_type == "kmeans" else GMM(random_state=5, n_components=cluster_size)
    X = dataset.iloc[:, :-1]
    algo.fit(X)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=pd.Series(algo.predict(X)), s=50, cmap='viridis')

    plt.xlabel("First Attribute")
    plt.ylabel("Second Attribute")
    if algo_type == "kmeans" :
        centers = algo.cluster_centers_ 
        plt.scatter(centers[:, 1], centers[:, 1], c='black', s=200, alpha=1)
    plt.savefig(data + "/"+ algo_type+"/" +data + "_"+(type if type is not None else "")+"_Clustering")
    plt.close()
    # plt.show()

datas = ["Sat", "Spam"]
for data in datas:
    dataset = pd.read_csv("Datasets/" + ("spambase.csv" if data == "Spam" else "sat-data.csv"),header=None, delimiter=",")
    clustering(dataset, data=data)
    plotGraphs(data=data, clustering="kmeans")
    plotGraphs(data=data, clustering="EM")
    clusterGraph(dataset, data=data, algo_type="kmeans", cluster_size=(2 if data == "Spam" else 7))
    clusterGraph(dataset, data=data, algo_type="EM", cluster_size=(2 if data == "Spam" else 7))

    types = ["PCA", "ICA", "RP", "RF"]

    for type in types:
        dataset = pd.read_csv("Datasets/"+data+"_DR_"+type+".csv",delimiter=",")
        clustering(dataset, data=data, type= type)
        plotGraphs(data=data, type=type, clustering="kmeans", vline=False)
        plotGraphs(data=data, type=type, clustering="EM", vline=False)
