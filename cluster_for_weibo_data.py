#! /usr/bin/python

# -*- coding: utf-8 -*-


from numpy import *
import math
import jieba

def loadWeiboData(fileName):
    weiboData = []
    i = 0
    with open(fileName) as f:
        for line in f:
            #print line
            i  += 1
            lineSplit = line.strip().split(',')
            #print lineSplit
      
            if len(lineSplit) == 15:
            #if len(lineSplit) == 16:
                data = []
                data.append(i)
                data.append(lineSplit[7].strip().decode('utf-8'))
                #data.append(lineSplit[5].strip().decode('utf-8'))
                weiboData.append(data)
                
            if len(lineSplit) == 22:
            #if len(lineSplit) == 16:
                data = []
                data.append(i)
                data.append(lineSplit[10].strip().decode('utf-8'))
                #data.append(lineSplit[5].strip().decode('utf-8'))
                weiboData.append(data)

            #if len(lineSplit) == 15:
            if len(lineSplit) == 16:
                data = []
                data.append(i)
                #data.append(lineSplit[7].strip().decode('utf-8'))
                data.append(lineSplit[5].strip().decode('utf-8'))
                weiboData.append(data)

    return array(weiboData)


def getStopWords():
    stopwords = []
    for word in open("stopwords.txt", "r"):
        stopwords.append(word.decode('utf-8').strip())
    return stopwords


def cutContent(content, stopwords):
    #print stopwords
    cutWords = []
    words = jieba.cut(content)
    #print words
    for word in words:
        if word == u' ':
            continue
        if word not in stopwords:
            cutWords.append(word)
            #print unicode(word)
    return cutWords

def getTfid(word, recordContent):
    i = 0
    for wordData in recordContent:
        if wordData == word:
            i = i+1
    return i

dictData = {}

def getNi(word, documents):
    #print 'getNi'
    global dictData
    
    if word in dictData.keys():
        return dictData[word]
    
    j = 0
    n = documents.shape[0]
    
    for i in range(n):
        if word in documents[i][1]:
            j = j+1

    dictData[word] = j
    return j

'''
ｗ ｉ＝ｔ ｆ ｉ（ ｄ） ＊ｌｏ ｇ（ Ｎ／ ｎ ｉ） （ １）
其中ｔ ｆ ｉ（ ｄ） 为特征项ｔ ｉ在文档ｄ 中出现的频率， Ｎ
为所有文档数目， ｎ ｉ为含有项ｔ ｉ的文档数目。

'''
def VSMdocument(i, documents):
    N = documents.shape[0]
    recordContent = documents[i][1] #分词列表
    #compute the term's weights
    
    VSM = []
    for word in set(recordContent):
        termWeight = []
        wi = 0
        tfid = getTfid(word, recordContent)
        n = getNi(word, documents)
        wi = tfid*log(float(N)/n)
        #print wi
        termWeight.append(word)
        termWeight.append(wi)
        VSM.append(termWeight)
    return array(VSM, dtype = object)

def simcos(vecA, vecB):
    k = min(vecA.shape[0], vecB.shape[0])

    numerator = 0
    for i in range(k):
        numerator += vecA[i][1]*vecB[i][1]

    denoinatorA = 0
    denoinatorB = 0

    for i in range(k):
        denoinatorA += math.pow(vecA[i][1], 2)

    for i in range(k):
        denoinatorB += math.pow(vecB[i][1], 2)
    
    denoinator = sqrt(denoinatorA*denoinatorB)

    return (numerator)/denoinator

def jaccardCoeff(vecA, vecB):
    #print vecA
    if len(vecA) == 0:
        return 0.00000001
    if len(vecB) == 0:
        return 0.00000001
    #print vecA[:,0]
    setA = set(vecA[:,0])
    setB = set(vecB[:,0])
    #print set(vecA[:,0])
    #print vecA[:,0]
    #print vecA
    #setA = set(vecA[:])
    #setB = set(vecB[:])

    unionset = setA | setB

    interset = setA & setB

    answer = (float)(len(interset))/len(unionset)
    if answer < 0.0000000001:
        return 0.00000001
    
    return answer


def getMaxSimilarity(Vec, v):
    similarity = []
    for index,item in enumerate(v):
        similarity.append(jaccardCoeff(Vec, item))
    #print max(similarity)
    return max(similarity)

dictTopic = {}
numTopic = 0 
def single_pass(Vec, TC):
    #find the old topic
    if len(Vec) == 0: return
    global dictTopic
    global numTopic
    allSimilarity = []
    #实现只和话题中的第一个进行比较
    #oneSimilarity = []
    
    if numTopic == 0:
        dictTopic[numTopic] = []
        dictTopic[numTopic].append(Vec)
        numTopic += 1
    
    else:
        maxValue = 0
        maxIndex = -1
        for k,v in dictTopic.iteritems():
            oneSimilarity = getMaxSimilarity(Vec, v)#jaccardCoeff(Vec, v[0])
            if oneSimilarity > maxValue:
                maxValue = oneSimilarity
                maxIndex = k
            
            #allSimilarity.append(oneSimilarity)
        #if the similarity is bigger than TC
        #join the most similar topic
        if maxValue > TC:
            dictTopic[maxIndex].append(Vec)
        #else create the new topic
        else:
            dictTopic[numTopic] = []
            dictTopic[numTopic].append(Vec)
            numTopic += 1



def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    #print dataMat
    return array(dataMat)

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

def randWeiboCent2(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

def randWeiboCent(dataSet, k):
    n = shape(dataSet)[0]
    #print n
    kset = []
    while(1):
        if len(kset) >= k:
            break
        j = random.randint(0,n-1)
        if j not in kset:
            kset.append(j)

    return kset

def takemin(ptsInClust, dataSet, distMeas=simcos):
    setCent = list(ptsInClust)
    n = len(setCent)
    #print setCent
    minIndex = -1
    #print n
    #if n == 0:
    #    return -1
    for i in range(n):
        distsum = 0
        mindist = inf
        for j in range(n):
            distsum += (float)(1)/distMeas(dataSet[ptsInClust[i]], dataSet[ptsInClust[j]])
        if distsum < mindist:
            mindist = distsum
            minIndex = i
    #print minIndex
    return setCent[minIndex]
    
    


def weibokMeans(dataSet, k, distMeas=simcos, createCent=randWeiboCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)  #list contains record ids 
    clusterChanged = True
    sumcount = 0
    while clusterChanged:
        #sumcount += 1
        #if sumcount > 5 :
        #    break
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = 1/distMeas(dataSet[centroids[j]], dataSet[i])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = centroids[j]
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist
        print centroids
        print '--------------------'
        #delIndex = -1
        for cent in range(k):
            #ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            ptsInClust = nonzero(clusterAssment[:,0].A == centroids[cent])
            #print type(ptsInClust)
            #print ptsInClust[0]
      
            lenpts = len(list(ptsInClust[0]))
            #print lenpts
            if lenpts == 0:
                clusterAssment[centroids[cent],:] = centroids[cent], 1
                #print 'is 0'
                continue
            
            centroids[cent] = takemin(ptsInClust[0],dataSet)
            '''
            #if centroids[cent] == -1:
                #delIndex = cent
            if delIndex != -1:
            k = k - 1
            del centroids[delIndex]
            '''
        
    return centroids, clusterAssment


def weibokMeans2(dataSet, k, distMeas=simcos, createCent=randWeiboCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        #print centroids
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment



if __name__ == '__main__':
    #datMat = loadWeiboData('yibin2.csv')
    #datMat = loadWeiboData('yibin.csv') #1
    #datMat = loadWeiboData('yulebao.csv') #2
    datMat = loadWeiboData('yibin_yulebao.csv') #4
    #datMat = loadWeiboData('suiji.csv') #3
    #datMat = loadWeiboData('suiji.csv')
    #print type(datMat)
    stopWords = getStopWords()

    n = datMat.shape[0]
    print 'total records:', n

    cutWeiboData = []
    for i in range(n):
        #print datMat[i][1]
        data = []
        data.append(datMat[i][0])
        data.append(cutContent(datMat[i][1], stopWords))
        cutWeiboData.append(data)
    cutWeiboData = array(cutWeiboData, dtype=object)
    print 'cutWeiboData is done'
    #print cutWeiboData[0]
    #get VSM
    recordVSMs = []
    for i in range(n):
        recordVSM = []
        recordVSM = VSMdocument(i, cutWeiboData)
        recordVSMs.append(recordVSM)
    #print recordVSMs[0]


    recordVSMs = array(recordVSMs)
    print 'VSM is done'
    print 'kMeans is starting..'
    #print simcos(recordVSMs[0], recordVSMs[1])
    #print jaccardCoeff(recordVSMs[0], recordVSMs[1])
    
    #randK = randWeiboCent(recordVSMs, 3)
    '''
    centroids, clusterAssment = weibokMeans(recordVSMs, 3, jaccardCoeff)
    #centroids, clusterAssment = weibokMeans(recordVSMs, 3)
   
    #print centroids
    #print '---------------'
    #print clusterAssment

    for i in centroids:
        print datMat[i][1]
    '''

    for vec in recordVSMs:
        single_pass(vec, 0.03)

    print numTopic
    #print dictTopic

    for i in range(5):
        for key in  dictTopic[i][0]:
            print key[0]
        print '---------------------'
    #print dictTopic
    
