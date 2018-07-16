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


def PCA_Classification(X, Y, data="Spam", dim=5):

    pca = PCA(n_components=dim,random_state=10)
    new_DF = pd.DataFrame(pca.fit_transform(X))
    new_DF["Class"] = Y
    new_DF.to_csv("Datasets/"+data+ "_DR_PCA.csv", index=False)

def ICA_Classification(X, Y, data="Spam", dim = 5):

    ica = FastICA(random_state=5, n_components=dim)
    new_DF = pd.DataFrame(ica.fit_transform(X))
    new_DF["Class"] = Y
    new_DF.to_csv("Datasets/"+data+ "_DR_ICA.csv", index=False)


def RP_Classification(X, Y, data= "Spam", dim = 5):
    rp = SparseRandomProjection(random_state=5, n_components=dim)
    new_DF = pd.DataFrame(rp.fit_transform(X))
    new_DF["Class"] = Y
    new_DF.to_csv("Datasets/" + data + "_DR_RP.csv", index=False)
    # return

def RF_Classification(X, Y,  features, data= "Spam"):
    new_DF = pd.DataFrame(X.iloc[:, features])
    new_DF["Class"] = Y
    new_DF.to_csv("Datasets/" + data + "_DR_RF.csv", index=False)

datas = ["Sat", "Spam"]
for data in datas:
    dataset = pd.read_csv("Datasets/" + ("spambase.csv" if data == "Spam" else "sat-data.csv"),header=None, delimiter=",")
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]
    PCA_Classification(X, Y, data=data)
    ICA_Classification(X, Y, data=data)
    RP_Classification(X, Y, data=data)
    RF_Classification(X, Y, [1, 4, 5, 32, 50] if data == "Spam" else [16, 18, 19, 20, 22], data=data)