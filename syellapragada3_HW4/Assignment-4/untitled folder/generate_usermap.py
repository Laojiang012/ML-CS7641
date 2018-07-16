import pickle
import numpy
userMap = []

for i in range(0,40):
    
    userMap.append(numpy.random.choice(numpy.arange(-2, 3), 40, p=[0.05, 0.05, 0.7, 0.2]).tolist())

print type(userMap)
print type(userMap[0][0])

pickle.dump( userMap, open( "userMap.p", "wb" ) )
