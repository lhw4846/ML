from numpy import *
import svmutil as svm

def textParse(bigString):    #input is big string, #output is word list
    listOfTokens = bigString.strip().split(' ')
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def loadDataSet():
    # positive
    fr=open('data/rt-polarity.pos')
    posPosting = [textParse(inst) for inst in fr.readlines()]
    posLen = len(posPosting)
    posLabels = ones(posLen).tolist()

    # negative
    fr=open('data/rt-polarity.neg')
    negPosting = [textParse(inst) for inst in fr.readlines()]
    negLen = len(negPosting)
    negLabels = zeros(negLen).tolist()

    # merge
    posPosting.extend(negPosting)
    posLabels.extend(negLabels)

    return posPosting, posLabels

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        #else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones()
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    # training
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    # test
    errorCount = 0
    for testIndex in range(0, len(listOPosts)):
        thisDoc = array(setOfWords2Vec(myVocabList, listOPosts[testIndex]))
        if classifyNB(thisDoc,p0V,p1V,pAb) != listClasses[testIndex]:
            errorCount += 1
    print 'the error rate is:', float(errorCount)/len(listOPosts)*100


def trainSVM(trainMatrix, trainCategory):
    svm.svm_model.predict = lambda self, x: svm.svm_predict([0], [x], self)[0][0]

    prob = svm.svm_problem(trainCategory, trainMatrix)
    param = svm.svm_parameter()
    param.kernel_type = svm.LINEAR
    param.C = 10

    model = svm.svm_train(prob, param)
    return model

def testingSVM(trainMat, listClasses):
    listClasses = map(int, listClasses)
    model = trainSVM(trainMat, listClasses) # training svm
    svm.svm_predict(listClasses, trainMat, model)