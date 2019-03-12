from starter import *
import tensorflow as tf
#import scipy.signal

def CNN(x, conv_layer_kernel, conv_layer_bias, linear_layer1_w, linear_layer1_bias, linear_layer2_w, linear_layer2_bias, stride=1):

    # Defining model behaviour
    output = tf.nn.conv2d(x, filter = conv_layer_kernel, strides=[1, stride, stride, 1], padding ='SAME')
    output = tf.nn.bias_add(output, conv_layer_bias)
    output = tf.nn.relu(output)

    mean, var = tf.nn.moments(output, axes=[0])
    output = tf.nn.batch_normalization(output, mean=mean, variance=var, offset=None, scale=None, variance_epsilon=1e-8)
    #output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    output = tf.layers.max_pooling2d(output, 2, 2)
    output = tf.contrib.layers.flatten(output)

    output = tf.add(tf.matmul(output, linear_layer1_w), linear_layer1_bias)

    output = tf.nn.dropout(output, rate=0.9)
    output = tf.nn.relu(output)
    output = tf.add(tf.matmul(output, linear_layer2_w), linear_layer2_bias)

    return tf.nn.softmax(output)


if __name__ == "__main__":
    np.random.seed(0)
    tf.random.set_random_seed(0)

    #Getting Data
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)

    #Setting the dimensions right for CNN
    trainData = np.expand_dims(trainData, 3)
    validData = np.expand_dims(validData, 3)
    testData = np.expand_dims(testData, 3)

    #Inititalizing filters and biases tensors
    conv_layer_kernel = tf.get_variable("W0", (3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer())
    conv_layer_bias = tf.get_variable("b0", (32), initializer=tf.contrib.layers.xavier_initializer())

    linear_layer1_w = tf.get_variable("W1", (32 * 14 * 14, 784), initializer=tf.contrib.layers.xavier_initializer())
    linear_layer1_bias = tf.get_variable("b1", (784), initializer=tf.contrib.layers.xavier_initializer())

    linear_layer2_w = tf.get_variable("W2", (784, 10), initializer=tf.contrib.layers.xavier_initializer())
    linear_layer2_bias = tf.get_variable("b2", (10), initializer=tf.contrib.layers.xavier_initializer())

    input = tf.placeholder("float", shape = [None, 28, 28, 1])
    labels = tf.placeholder("float", shape = [None, 10])


    #Hyperparameters for training
    reg = 0
    alpha = 1e-4
    batch_size = 32
    epochs = 50

    # Now lets use the model to make predictions and define regularized loss
    predictions = CNN(input, conv_layer_kernel, conv_layer_bias, linear_layer1_w, linear_layer1_bias, linear_layer2_w, linear_layer2_bias)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=predictions)) + \
            tf.multiply(tf.reduce_mean(tf.nn.l2_loss(conv_layer_kernel) + tf.nn.l2_loss(linear_layer1_w) \
            + tf.nn.l2_loss(linear_layer2_w)), reg)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels,1), tf.argmax(predictions, 1)), dtype=tf.float64))

    #Define the optimizer and initialize it
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
    optimizer = optimizer.minimize(loss)


    #Arrays to take care of plotting
    trainloss_array = np.zeros(epochs+1)
    validloss_array = np.zeros(epochs+1)
    testloss_array = np.zeros(epochs+1)

    trainacc_array = np.zeros(epochs+1)
    validacc_array = np.zeros(epochs+1)
    testacc_array = np.zeros(epochs+1)

    relevantepoch = np.zeros(epochs+1)


    #The computational graph has now been set up. We can start an interactive session to feed in data for training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        trainloss, trainacc = sess.run([loss, accuracy], feed_dict={input: trainData[:1000], labels: newtrain[:1000]})
        validloss, validacc = sess.run([loss, accuracy], feed_dict={input: validData[:1000], labels: newvalid[:1000]})
        testloss, testacc = sess.run([loss, accuracy], feed_dict={input: testData[:1000], labels: newtest[:1000]})

        trainloss_array[0] = trainloss
        validloss_array[0] = validloss
        testloss_array[0] = testloss
        trainacc_array[0] = trainacc
        validacc_array[0] = validacc
        testacc_array[0] = testacc
        relevantepoch[0] = 0

        for epoch in range(epochs):

            shuffle(trainData, newtrain)

            for batch in range(int(trainData.shape[0] / batch_size)):
                data = trainData[batch * batch_size: min((batch + 1) * batch_size, len(trainData))]

                lab = newtrain[batch * batch_size: min((batch + 1) * batch_size, len(trainData))]

                sess.run(optimizer, feed_dict={input: data, labels: lab})

                print("Epoch: {}/{}, Batch: {}/{}".format(epoch+1, epochs, batch, int(trainData.shape[0] / batch_size)))


            trainloss, trainacc = sess.run([loss, accuracy], feed_dict={input: trainData[:1000], labels: newtrain[:1000]})
            validloss, validacc = sess.run([loss, accuracy], feed_dict={input: validData[:1000], labels: newvalid[:1000]})
            testloss, testacc = sess.run([loss, accuracy], feed_dict={input: testData[:1000], labels: newtest[:1000]})

            trainloss_array[epoch+1] = trainloss
            validloss_array[epoch+1] = validloss
            testloss_array[epoch+1] = testloss
            trainacc_array[epoch+1] = trainacc
            validacc_array[epoch+1] = validacc
            testacc_array[epoch+1] = testacc
            relevantepoch[epoch+1] = epoch+1


    # Training done, now plot accuracies and losses
    fig = plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(np.array(relevantepoch), np.array(trainloss_array), label='Training Loss')
    plt.plot(np.array(relevantepoch), np.array(testloss_array), label='Test Loss')
    plt.plot(np.array(relevantepoch), np.array(validloss_array), label='Validation Loss')
    plt.plot()
    mytitle = "CELoss|{} Regularization Constant".format(reg)
    plt.title(mytitle, fontsize=12)
    plt.legend()
    title = "Dropout{}Loss".format(1) #Change this 1 = 0.01, 2 = 0.1 etc
    plt.savefig(title)

    fig = plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(np.array(relevantepoch), np.array(trainacc_array), label='Training Accuracy')
    plt.plot(np.array(relevantepoch), np.array(testacc_array) , label='Test Accuracy')
    plt.plot(np.array(relevantepoch), np.array(validacc_array), label='Validation Accuracy')
    plt.plot()
    mytitle = "CE Accuracy|{} Regularization Constant".format(reg)
    plt.title(mytitle, fontsize=12)
    plt.legend()
    title = "Dropout{}Accuracy".format(1)#Change this 1 = 0.01, 2 = 0.1 etc or 1==>p=0.9, 2=>p=0.75, 3=>p=0.5
    plt.savefig(title)

    print("final test accuracy: ", testacc_array[-1])
    print("final validation accuracy: ", validacc_array[-1])
    print("final train accuracy", trainacc_array[-1])


