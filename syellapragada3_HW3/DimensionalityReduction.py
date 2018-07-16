import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from collections import defaultdict
from itertools import product
from helpers import pairwiseDistCorr, reconstructionError
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier
import DR_Clustering as dr

def computePCA(X, Y, data = "Spam"):
    pca = PCA(random_state=5)
    pca.fit(X)
    tmp = pd.Series(data=pca.explained_variance_, index=range(1, X.shape[1] + 1))
    tmp.to_csv(data + "/DimRed/PCA.csv")
    m1 = max(tmp) * 60 / 100
    # m2 = sum(tmp) * 80 / 100
    dr.PCA_Classification(X, Y, data=data, dim=len(tmp[tmp>m1].index.values))
    plotGraph(tmp, x="Features", y="Variance", path=data + "/DimRed/"+data+"_PCA_plot", data=data, vlines=tmp[tmp>m1].index.values)
    # return

def computeICA(X, Y, data= "Spam"):
    dims = range(2, X.shape[1])
    ica = FastICA(random_state=5)
    kurt = {}
    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt[dim] = tmp.abs().mean()

    kurt = pd.Series(kurt)

    kurt.to_csv(data + "/DimRed/ICA.csv")
    m1 = max(kurt) * 60 / 100
    dr.ICA_Classification(X, Y, data=data, dim=len(kurt[kurt > m1].index.values))
    plotGraph(kurt, x="Features", y="Kurtosis", path=data + "/DimRed/"+data+"_ICA_plot", data=data, vlines=kurt[kurt>m1].index.values)
    # return

def randomizedProjections(X, Y, data= "Spam"):
    dims = range(2, X.shape[1])
    tmp = defaultdict(dict)
    for i, dim in product(range(3), dims):
        rp = SparseRandomProjection(random_state=i, n_components=dim)
        rp.fit(X)
        tmp[dim][i] = reconstructionError(rp, X)
    tmp = pd.DataFrame(tmp).T
    tmp.to_csv(data + '/DimRed/RP.csv')

    vlines = plotGraph(tmp, x="Features", y="Reconstruction Error", path=data + "/DimRed/"+data+"_RP_plot", data=data, colors= ['C0', 'C1', 'C2'])
    dr.RP_Classification(X, Y, data=data, dim=len(list(set(vlines))))
    # return

def randomFeature(X, Y, data= "Spam"):
    tree = DecisionTreeClassifier(criterion="entropy")
    fs = tree.fit(X, Y).feature_importances_ #measure of information content or gain for each feature

    tmp = pd.Series(fs[::-1])
    tmp.to_csv(data + '/DimRed/RF.csv')
    m1 = max(tmp) * 60 / 100
    plotGraph(tmp, x="Features", y="Information gain", path=data + "/DimRed/"+data+"_RF_plot", data=data, vlines=tmp[tmp>m1].index.values)
    dr.RF_Classification(X, Y, data=data, features=tmp[tmp>m1].index.values)
    # return

def plotGraph(tmp, x, y, path, data="Spam", vlines = [], colors = []):
    if colors == []:
        colors = 'g' if data == "Spam" else 'r'
        plt.plot(tmp, label='rbf', marker='o', linestyle=':' , c=colors)
        for xc in vlines:
            plt.axvline(x=xc, color='grey', linestyle='--')
    else:
        tmp.plot(color=colors, label='rbf', marker='o', linestyle=':')
        i=0
        for column in tmp:
            tmp1 = tmp[column]
            m1 = max(tmp1) * 60 / 100
            v = tmp1[tmp1>m1].index.values
            # v = tmp1.nlargest(5).index.values
            vlines += list(v)
            for xc in v:
                plt.axvline(x=xc, color=colors[i], linestyle='--')
            i+=1
    plt.ylabel(y)
    plt.xlabel(x)
    plt.xticks(list(plt.xticks()[0]) + list(vlines), rotation = 45)

    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return vlines


datas = ["Sat", "Spam"]
for data in datas:
    dataset = pd.read_csv("Datasets/" + ("spambase.csv" if data == "Spam" else "sat-data.csv"),header=None, delimiter=",")
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]
    computePCA(X, Y,  data=data)
    computeICA(X, Y, data=data)
    randomizedProjections(X, Y, data=data)
    randomFeature(X, Y, data=data)