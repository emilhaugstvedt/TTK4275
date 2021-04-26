from sklearn import datasets
import numpy as np

## Defining constants ##
traingStart = 0
traingStop = 30
testingStart = 30
testingStop = 50 

## Parameters for linear classifier ##
alpha = 0.01
N_iter = 5000 # Number of iterations


def extractData(irisData):
    class1 = irisData[0:50]
    class2 = irisData[50:100]
    class3 = irisData[100:150]
    return class1, class2, class3

def trainingData(irisClass, traingStart, traingStop):
    return irisClass[traingStart:traingStop]

def testingData(irisClass, testingStart, testingStop):
    return irisClass[testingStart:testingStop]

def linearClassifier(classes, N_training, N_testing):
    features = [0,1,2,3]

    C = len(classes) # Number of classes
    F = len(features) # Number of features

    W = np.zeros((C, F))
    w_0 = np.zeros((C, 1))

    W = np.concatenate((W, w_0), axis=1)

    # Extracting training data #
    training1 = [trainingData(classes[0], 0, N_training), features]
    training2 = [trainingData(classes[1], 0, N_training), features]
    training3 = [trainingData(classes[2], 0, N_training), features]
    training = [training1, training2, training3]

    print(training)


    # Extracting testing data #
    testing1 = testingData(classes[0], N_training, N_training + N_testing)
    testing2 = testingData(classes[1], N_training, N_training + N_testing)
    testing3 = testingData(classes[2], N_training, N_training + N_testing)

    # Creating targets #
    target_k1 = np.array([[1],[0],[0]]) #class1
    target_k2 = np.array([[0],[1],[0]]) #class2
    target_k3 = np.array([[0],[0],[1]]) #class3
    target_k  = [target_k1, target_k2, target_k3]

    for _ in range(N_iter):
        W_prev = W
        grad_MSE = np.zeros((C,F+1))
        for k in range(N_training):
            for (xk, tk) in zip(training, target_k):
                xk = np.append(xk[k], 1) 
                print(xk)
                xk = xk.reshape(F+1, 1) 
                zk = W@xk
                gk = sigmoid(zk)
                temp = np.multiply(gk-tk, gk)
                temp = np.multiply(temp, np.ones((C,1))-gk)
                grad_MSE += temp@xk.T
        W = W_prev - alpha*grad_MSE
    return W

def sigmoid(x):
    return 1/(1+np.exp(-x))

if __name__ == "__main__":
    irisData = datasets.load_iris()['data']
    class1, class2, class3 = extractData(irisData)
    classes = [class1, class2, class3]
    W = linearClassifier(classes, 30, 20)
    print(W)