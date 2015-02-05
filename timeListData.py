#! /bin/usr/python
import random
import numpy as np
from singleThread_PPCCAEMalgorithmn import PCCAEMalgorithmn
import itertools
import pickle
from plotTimeList import readData

class timeListData(object):
	
	###This Classgenerate timeListData X and Y
	###the relation of X and Y is:
	### x[t]=a[0]*x[t-1]+a[1]*x[t-2]+a[2]*x[t-3]...a[n-1]*x[t-n]+Err(mean_x,var_x)
	### y[t]=b[0]*y[t-1]+b[1]*y[t-2]+b[2]*y[t-3]...b[m-1]*y[t-m]+c[0]*x[t-1]+c[1]*x[t-2]+c[2]*x[t-3]...c[s-1]*x[t-s]+Err(mean_y,var_y)
	### X=[x[t],x[t-1]...]
	### Y=[y[t],y[t-1]...]

	def __init__(self,n,m,s,rate,lengthOfData,lag_x,lag_y_x,lag_y_y,randomDataSwitch=True):
		#the init parameter of the Class is:
		#####list of float:rate--> the rate of random parameter, item in rate at the range of [0,1] eg. [0.3,0.5,0.1], rest are the rate of Noise.
		#####int:lengthOfData--> length of the Data you want to generate
		#####list of list :a -->a[0]=[a[0],a[1]...a[n]]
		#####list of list:b -->b[0]=[b[0],b[1]...b[m]]
		#####list of list:c -->c[0]=[c[0],c[1]...c[s]]
		#####list:init_x-->init_x=[x[0],x[1]...x[n-1]] 
		#####list:init_y-->init_y=[y[0],y[1]...y[o-1]] o=max(s,m) 
		#####float:mean_x,mean_y,var_x,var_y
		#####int:lag_x,lag_y_x,lag_y_y-->time lag 
		self.randomDataSwitch=randomDataSwitch
		self.n=n
		self.m=m
		self.s=s
		self.rate=rate
		self.lengthOfData=lengthOfData
		self.lag_x=lag_x
		self.lag_y_x=lag_y_x
		self.lag_y_y=lag_y_y

		self.upLimit=1

	def generateBinaryWeight(self,length):
		#return a list of binary weight with length
		binaryWeight=[]
		numOfOne=0
		for i in range(length):
			randNum=random.randint(0,1)
			binaryWeight.append(randNum)
			if randNum==1:
				numOfOne+=1

		if numOfOne<float(length/4):
			return self.generateBinaryWeight(length)
		elif numOfOne>float(length*3/4):
			return self.generateBinaryWeight(length)
		else:
			return binaryWeight
	
	def generateFloatWeight(self,length):
		#return a list of float weight with length
		floatWeight=[]
		for i in range(length):
			randNum=random.uniform(0,1)
			floatWeight.append(randNum)
		sumOfWeight=sum(floatWeight)
		floatWeight=[j/sumOfWeight for j in floatWeight]
		
		return floatWeight


	def generateInit(self,length):
		#return a list of initial value with length
		initVal=[]
		for i in range(length):
			randNum=random.uniform(0,self.upLimit)
			initVal.append(randNum)

		return initVal

	def judgePattern(self,randNum,rateList):
		temp=[0.0]
		temp.extend(rateList)
		rateList=temp
		for i in range(1,len(rateList)):
			if randNum>sum(rateList[:i])*10 and randNum<sum(rateList[:i+1])*10:
				return i-1


	def generateTimeListData_x(self,list_x,judgeList_x,a,mean_x=0,var_x=0):
		randNum=random.uniform(0,10)
		len_a=len(a[0])
		index=self.judgePattern(randNum,self.rate)
		len_x=len(list_x)

		if index<len(a):
			array_x=np.array(list_x[len_x-self.lag_x-len_a+1:len_x-self.lag_x+1])
			array_a=np.array(a[index])
			error=random.normalvariate(mean_x,var_x)
			newNum=np.dot(array_x,array_a.T)+error
			list_x.append(newNum)
			judgeList_x.append(index+1)

		if index>=len(a):
			if list_x[-1]<0:
				newNum=random.uniform(self.upLimit,1)
			elif list_x[-1]>2*self.upLimit:
				newNum=random.uniform(0,self.upLimit/2)
			else:
				newNum=random.uniform(0,self.upLimit)
			list_x.append(newNum)
			judgeList_x.append(0)
		
		return list_x,judgeList_x

	def generateTimeListData_y(self,list_x,list_y,judgeList_y,b,c,mean_y=0,var_y=0):
		len_b=len(b[0])
		len_c=len(c[0])
		len_y=len(list_y)
		randNum=random.uniform(0,10)
		index=self.judgePattern(randNum,self.rate)
		
		if index<len(b):
			array_y=np.array(list_y[len_y-self.lag_y_y+1-len_b:len_y-self.lag_y_y+1])
			array_x=np.array(list_x[len_y-len_c-self.lag_y_x+1:len_y-self.lag_y_x+1])
			array_b=np.array(b[index])
			array_c=np.array(c[index])
			error=random.normalvariate(mean_y,var_y)
			newNum=np.dot(array_y,array_b.T)+np.dot(array_x,array_c.T)+error
			list_y.append(newNum)
			judgeList_y.append(index+1)

		if index>=len(b):
			if list_y[-1]<0:
				newNum=random.uniform(self.upLimit,1)
			elif list_y[-1]>2*self.upLimit:
				newNum=random.uniform(0,self.upLimit/2)
			else:
				newNum=random.uniform(0,self.upLimit)
			list_y.append(newNum)
			judgeList_y.append(0)
		
		return list_y,judgeList_y


	def generateTimeListData(self,mean_x=0,mean_y=0,var_x=0,var_y=0):
		#self.a=self.generateBinaryWeight(self.n)
		#self.b=self.generateBinaryWeight(self.m)
		#self.c=self.generateBinaryWeight(self.s)
		self.a=[self.generateFloatWeight(self.n) for l in range(len(self.rate)-1)]
		self.b=[]
		self.c=[]
		for m in range(len(self.rate)-1):
			weight_y=self.generateFloatWeight(self.m+self.s)
			self.b.append(weight_y[:self.m])
			self.c.append(weight_y[self.m:])

		initLen_x=self.lag_x+self.n-1
		initLen_y=max(self.m+self.lag_y_y,self.s+self.lag_y_x)-1
			
		self.init_x=self.generateInit(initLen_x)
		self.init_y=self.generateInit(initLen_y)
		judgeList_x=[0]*initLen_x
		judgeList_y=[0]*initLen_y
		list_x=self.init_x
		list_y=self.init_y
		

		while len(list_x)<self.lengthOfData:
			list_x,judgeList_x=self.generateTimeListData_x(list_x,judgeList_x,self.a)

		while len(list_y)<self.lengthOfData:
			list_y,judgeList_y=self.generateTimeListData_y(list_x,list_y,judgeList_y,self.b,self.c)
	
		return list_x,list_y,judgeList_x,judgeList_y

	def makeEmbededVec(self,list_x,list_y,lenEmbeded_x,lenEmbeded_y,lagX=1,lagY=1):
		initPoint=max(lenEmbeded_x+lagX,lenEmbeded_y+lagY)-1
		
		
		#data=[[np.matrix(list_y[i]).T,np.matrix(list_x[i-lagX+1-lenEmbeded_x:i-lagX+1]).T,np.matrix(list_y[i-lagY+1-lenEmbeded_y:i-lagY+1]).T] for i in range(initPoint,len(list_x))]	
		data=[[np.matrix(list_y[i-lagY+1-lenEmbeded_y:i-lagY+1]).T,np.matrix(list_x[i-lagX+1-lenEmbeded_x:i-lagX+1]).T,np.matrix(list_y[i]).T] for i in range(initPoint,len(list_x))]
		
		#data=[[np.matrix(list_y[i]).T,np.matrix(list_y[i-lagY+1-lenEmbeded_y:i-lagY+1]).T,np.matrix(list_x[i-lagX+1-lenEmbeded_x:i-lagX+1]).T] for i in range(initPoint,len(list_x))]

		#EM = PCCAEMalgorithmn(data , numCluster )
		#serial_results = EM.calcUntilNoChangeClustering()
		return data

	def calculateData(self,data,numCluster):
		try:
			EM = PCCAEMalgorithmn(data , numCluster )
			serial_results = EM.calcUntilNoChangeClustering()
			return serial_results
		except:
			return self.calculateData(data,numCluster)

		else:
			return serial_results

	def calculateAccuracy(self,truth,result,truth_unique,result_unique):
		ruleList=[[truth_unique[i],result_unique[i]] for i in range(len(truth_unique))]
		iterNum=min(len(truth),len(result))
		zeroInTruth=truth[-iterNum:].count(0)
		accuracy=0
		for i in range(1,iterNum+1):
			if [truth[-i],result[-i]] in ruleList:
				accuracy+=1
		accuracyRate=float(accuracy)/(iterNum-zeroInTruth)
		return accuracyRate
		
	
	def accuracyRate(self,truth,result):

		uniqueItem_truth=list(np.unique(truth))
		uniqueItem_truth.remove(0)
		uniqueItem_result=np.unique(result)
		len_truth=len(uniqueItem_truth)
		len_result=uniqueItem_result.shape[0]
		if len_truth>=len_result:
			list_truth=list(uniqueItem_truth)
			fullPermutations_truth=list(itertools.permutations(list_truth,len_result))
			maxAccuracy=0
			for i in fullPermutations_truth:
				accuracy=self.calculateAccuracy(truth,result,i,list(uniqueItem_result))
				if accuracy>maxAccuracy:
					maxAccuracy=accuracy

		if len_truth<len_result:
			list_truth=list(uniqueItem_truth)
			fullPermutations_result=list(itertools.permutations(list(uniqueItem_result),len_truth))
			maxAccuracy=0
			for j in fullPermutations_result:
				accuracy=self.calculateAccuracy(truth,result,list_truth,j)
				if accuracy>maxAccuracy:
					maxAccuracy=accuracy

		return maxAccuracy

def main():
	a=timeListData(4,3,3,[0.3,0.3,0.3,0.1],1000,2,2,2)
	maxAccuracy=0
	for ii in range(10):
		x_list,y_list,x_truth,y_truth=a.generateTimeListData()
	#dict for saving
		dictionary={"x_list":x_list,
			"y_list":y_list,
			"x_truth":x_truth,
			"y_truth":y_truth,
			"a":a.a,
			"b":a.b,
			"c":a.c}
		#output=open("data.pkl","wb")
		#pickle.dump(dictionary,output)
		accuracyRateList=[]
		for i in range(2):
			data=a.makeEmbededVec(x_list,y_list,4,4)
			result=a.calculateData(data,3)
			resultList=result[-1]
			accuracy=a.accuracyRate(y_truth,resultList)
			accuracyRateList.append(accuracy)
		print "acc",np.mean(accuracyRateList)
		if len(result)>5:
			accuracyRateList.append(np.mean(accuracyRateList))
			filename="newData"+str(ii)+".pkl"
			output=open(filename,"wb")
			pickle.dump(dictionary,output)

	return accuracyRateList
	#return maxAccuracy

def main_(sampleStartingNum=0):
	allList=[]
	for j in range (4,5):
		a=timeListData(4,3,3,[0.6,0.2,0.1,0.1],1000,2,2,2)
		accuracyRateList=[]
		for ii in range(10):
			x_list,y_list,x_truth,y_truth=readData("data52.pkl")
			#output=open("data.pkl","wb")
			#pickle.dump(dictionary,output)
			data=a.makeEmbededVec(x_list,y_list,j,j)
			data=data[sampleStartingNum:]
			result=a.calculateData(data,3)
			resultList=result[-1]
			accuracy=a.accuracyRate(y_truth,resultList)
			accuracyRateList.append(accuracy)
			print "acc",accuracy
		allList.append(accuracyRateList)

	return allList

def main1():
	AllList=[]
	for i in range(0,10):
		r=main_(i*100)
		AllList.append(r)
	return AllList
