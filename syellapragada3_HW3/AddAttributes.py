import pandas as pd

# spambase = pd.read_csv("spambase.csv",header=None, delimiter=",")
#
# headers = []
# for i in range(len(spambase.columns)):
#     headers += ["col" + str(i)]
#
# spambase.columns = headers
# spambase.to_csv('spam-withHeaders.csv', index=False)
# print

sat_trn = pd.read_csv("Datasets/sat-trn.csv",header=None, delimiter=" ")
sat_tst = pd.read_csv("Datasets/sat-tst.csv",header=None, delimiter=" ")

sat_combined = pd.concat([sat_trn, sat_tst])
sat_combined.to_csv('Datasets/sat-data.csv', index=False)