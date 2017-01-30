from numpy import *
import operator
import matplotlib
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
		classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
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
		line=line.strip()
		listFromLine=line.split('\t')
		returnMat[index,:]=listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat,classLabelVector


group,labels=creatDataSet()
#print(classify0([0,0],group,labels,3))

datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
print(datingDataMat)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15*array(datingLabels),15*array(datingLabels))
plt.show()