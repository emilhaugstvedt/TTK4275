from sklearn import datasets




def extractData(irisData):
    class1 = irisData[0:50]
    class2 = irisData[50:100]
    class3 = irisData[100:150]
    return class1, class2, class3

def testingData(irisClass):
    return irisClass[30:50]

def traingData(irisClass):
    return irisClass[0:30]



if __name__ == "__main__":
    irisData = datasets.load_iris()['data']
    class1, class2, class3 = extractData(irisData)
    print(class1, class2)
