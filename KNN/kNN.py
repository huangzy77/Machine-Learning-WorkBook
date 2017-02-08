# -*- coding:utf-8 -*- 
from numpy import *
import operator
import matplotlib.pyplot as plt

#page17
def creatDataSet():
	group=array([[1,1.1],[1,1],[0,0],[0,1]])
	labels=['A','A','B','B']
	return group,labels

#page19
def classify0(inX,dataSet,labels,k):  
	dataSetSize=dataSet.shape[0]
	diffMat=tile(inX,(dataSetSize,1))-dataSet
	sqDiffMat=diffMat**2
	sqDistances=sqDiffMat.sum(axis=1)
	distances=sqDistances**0.5
	sortedDistIndicies=distances.argsort()
	classCount={}
	for i in range(k):
		voteIlabel=labels[sortedDistIndicies[i]]
		classCount[voteIlabel]=classCount.get(voteIlabel,0)+1#这个叫迭代吗？
	sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

#page21
def file2matrix(filename):
	fr=open(filename)
	arrayOlines=fr.readlines()
	numberOflines=len(arrayOlines)
	returnMat=zeros((numberOflines,3))
	classLabelVector=[]
	index=0
	for line in arrayOlines:
		line=line.strip()#Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）
		listFromLine=line.split('\t')#Python split()通过指定分隔符对字符串进行切片
		returnMat[index,:]=listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat,classLabelVector

#page25 数据归一化
def autoNorm(dataSet):
	minVals=dataSet.min(0)
	maxVals=dataSet.max(0)
	ranges=maxVals-minVals
	#normDataSet=zeros(shape(dataSet))
	m=dataSet.shape[0]
	normDataSet=dataSet-tile(minVals,(m,1))
	normDataSet=normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minVals
#page27 约会网站测试代码
def datingClassTest():
	hoRatio=0.1#提出１０％的数据用于测试
	datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
	normMat,ranges,minVals=autoNorm(datingDataMat)
	m=normMat.shape[0]
	numTestVecs=int(m*hoRatio)
	errorCount=0
	for i in range(numTestVecs):
		classifierResult=classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
		print("the classifier came back with: %d, the real answer is: %d" %(classifierResult,datingDataMat[i]))
		if(classifierResult != datingLabels[i]):errorCount+=1
	print("the total error rate is: %f"%(errorCount/float(numTestVecs)))
	
#group,labels=creatDataSet()
#print(classify0([0,0],group,labels,3))

datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
#print(datingDataMat)

fig=plt.figure()
ax=fig.add_subplot(111)
#page24 图２－４
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2],s=15*array(datingLabels),c=15*array(datingLabels))
#page24 图２－５
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],s=15*array(datingLabels),c=15*array(datingLabels))
plt.show()


