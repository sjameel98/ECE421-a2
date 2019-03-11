from starter import *

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)


### Now need a NN implementation: 3 layers [784, 1000, 10]

hidden_units = 1000
epochs = 100
alpha = 0.001
#W0array = np.zeros((epochs, 784, hidden_units))
#W1array = np.zeros((epochs, hidden_units, 10))

#Making weight matrices
W0 = np.random.normal(0, np.sqrt(2/(784 + hidden_units)), (784, hidden_units))
W1 = np.random.normal(0, np.sqrt(2/(hidden_units + 10)), (hidden_units, 10))

#Inititalizing biases
b0 = np.random.normal(0, np.sqrt(2/(784 + hidden_units)), (1, hidden_units))
b1 = np.random.normal(0, np.sqrt(2/(hidden_units + 10)), (1, 10))

#Intitializing mu matrices
mu0_w = np.full((784, hidden_units), 1e-5)
mu0_b = np.full((1, hidden_units), 1e-5)

mu1_w = np.full((hidden_units, 10), 1e-5)
mu1_b = np.full((1, 10), 1e-5)

#Intializing gamma
gamma0 = 0.9
gamma1 = 0.9

N = trainData.shape[0]
trainData = trainData.reshape(-1, 784)
validData = validData.reshape(-1, 784)
testData = testData.reshape(-1, 784)

#Plotting stuff
trainloss = list()
validloss = list()
testloss = list()

trainacc = list()
validacc = list()
testacc = list()

relevantepoch = list()


for epoch in range(epochs):
    print(epoch)
    x_hidden = relu(computeLayer(trainData, W0, b0))
    x_outer = softmax(computeLayer(x_hidden, W1, b1))

    x_valid = softmax(computeLayer(relu(computeLayer(validData, W0, b0)), W1, b1))
    x_test = softmax(computeLayer(relu(computeLayer(testData, W0, b0)), W1, b1))

    #Loss
    trainloss.append(CE(newtrain, x_outer))
    validloss.append(CE(newvalid, x_valid))
    testloss.append(CE(newtest, x_test))

    #Accuracy
    trainacc.append((np.sum(np.argmax(x_outer, axis=1) == np.argmax(newtrain, axis=1))/N))
    validacc.append((np.sum(np.argmax(x_valid, axis=1) == np.argmax(newvalid, axis=1)) / N))
    testacc.append((np.sum(np.argmax(x_test, axis=1) == np.argmax(newtest, axis=1)) / N))

    relevantepoch.append(epoch)


    #Need backprop for optimization now
    #delta_1 = -1/N * (np.array([1]) - x_outer) * newtrain
    delta_1 = 1/N * (x_outer - newtrain)
    dldw_outer = x_hidden.T @ delta_1
    dldb_outer = np.sum(delta_1, axis=0).reshape(1, 10)


    delta_0 = (x_hidden > 0) * (delta_1@W1.T) #Revise this 1/N ?
    dldw_inner = trainData.T @ delta_0
    dldb_inner = np.sum(delta_0, axis=0).reshape(1, hidden_units) #This is basically [1 1 .... 1] * delta

    #Update weights
    mu0_w = gamma0 * mu0_w + alpha*dldw_inner
    W0 = W0 - mu0_w

    mu0_b = gamma0 * mu0_b + alpha*dldb_inner
    b0 = b0 - mu0_b

    mu1_w = gamma0 * mu1_w + alpha * dldw_outer
    W1 = W1 - mu1_w

    mu1_b = gamma0 * mu1_b + alpha * dldb_outer
    b1 = b1 - mu1_b


#Training done, now plot accuracies and losses
fig = plt.figure()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(np.array(relevantepoch), np.array(trainloss), label='Training Loss')
plt.plot(np.array(relevantepoch), np.array(testloss), label = 'Test Loss')
plt.plot(np.array(relevantepoch), np.array(validloss), label = 'Validation Loss')
plt.plot()
mytitle = "CELoss|{}Hidden_Units".format(hidden_units)
plt.title(mytitle, fontsize = 12)
plt.legend()
title = "{} Hidden_Units_CELossa".format(hidden_units)
plt.savefig(title)


fig = plt.figure()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(np.array(relevantepoch), np.array(trainacc), label='Training Accuracy')
plt.plot(np.array(relevantepoch), np.array(testacc), label = 'Test Accuracy')
plt.plot(np.array(relevantepoch), np.array(validacc), label = 'Validation Accuracy')
plt.plot()
mytitle = "CE Accuracy|{}Hidden_Units.png".format(hidden_units)
plt.title(mytitle, fontsize = 12)
plt.legend()
title = "{} Hidden_Units_Accuracya".format(hidden_units)
plt.savefig(title)

print(trainloss)
print(trainacc)








    #Saving Weights
    #W0array[epoch] = W0
    #W1array[epoch] = W1




    #print(np.shape(x_outer))
    #print(W0.shape)
    #print(W1.shape)










