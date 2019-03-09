from starter import *

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

print(np.shape(trainData))
print(np.shape(trainTarget))

print(np.shape(validData))
print(np.shape(validTarget))

print(np.shape(testData))
print(np.shape(testTarget))