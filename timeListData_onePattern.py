#! /bin/usr/python
import random
import numpy as np

class timeListData(object):
	
	###This Classgenerate timeListData X and Y
	###the relation of X and Y is:
	### x[t]=a[0]*x[t-1]+a[1]*x[t-2]+a[2]*x[t-3]...a[n-1]*x[t-n]+Err(mean_x,var_x)
	### y[t]=b[0]*y[t-1]+b[1]*y[t-2]+b[2]*y[t-3]...b[m-1]*y[t-m]+c[0]*x[t-1]+c[1]*x[t-2]+c[2]*x[t-3]...c[s-1]*x[t-s]+Err(mean_y,var_y)
	### X=[x[t],x[t-1]...]
	### Y=[y[t],y[t-1]...]

	def __init__(self,n,m,s,rate,lengthOfData,randomDataSwitch=True):
		#the init parameter of the Class is:
		#####list of float:rate--> the rate of random parameter, item in rate at the range of [0,1] eg. [0.3,0.5,0.1], rest are the rate of Noise.
		#####int:lengthOfData--> length of the Data you want to generate
		#####list:a -->a=[a[0],a[1]...a[n]]
		#####list:b -->b=[b[0],b[1]...b[m]]
		#####list:c -->c=[c[0],c[1]...c[s]]
		#####list:init_x-->init_x=[x[0],x[1]...x[n-1]] 
		#####list:init_y-->init_y=[y[0],y[1]...y[o-1]] o=max(s,m) 
		#####float:mean_x,mean_y,var_x,var_y
		self.randomDataSwitch=randomDataSwitch
		self.n=n
		self.m=m
		self.s=s
		self.rate=rate
		self.lengthOfData=lengthOfData

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

	def generateTimeListData_x(self,list_x,judgeList_x,a,mean_x=0,var_x=0.1):
		randNum=random.uniform(0,9)
		len_a=len(a)
		if randNum<=self.rate*10:
			array_x=np.array(list_x[-len_a:])
			array_a=np.array(a)
			error=random.normalvariate(mean_x,var_x)
			newNum=np.dot(array_x,array_a.T)+error
			list_x.append(newNum)
			judgeList_x.append(1)

		if randNum>self.rate*10:
			if list_x[-1]<0:
				newNum=random.uniform(self.upLimit,1)
			elif list_x[-1]>2*self.upLimit:
				newNum=random.uniform(0,self.upLimit/2)
			else:
				newNum=random.uniform(0,self.upLimit)
			list_x.append(newNum)
			judgeList_x.append(0)
		
		return list_x,judgeList_x

	def generateTimeListData_y(self,list_x,list_y,judgeList_y,b,c,mean_y=0,var_y=0.1):
		len_b=len(b)
		len_c=len(c)
		len_y=len(list_y)
		randNum=random.uniform(0,9)
		if randNum<=self.rate*10:
			array_y=np.array(list_y[-len_b:])
			array_x=np.array(list_x[len_y-len_c:len_y])
			array_b=np.array(b)
			array_c=np.array(c)
			error=random.normalvariate(mean_y,var_y)
			newNum=np.dot(array_y,array_b.T)+np.dot(array_x,array_c.T)+error
			list_y.append(newNum)
			judgeList_y.append(1)

		if randNum>self.rate*10:
			if list_y[-1]<0:
				newNum=random.uniform(self.upLimit,1)
			elif list_y[-1]>2*self.upLimit:
				newNum=random.uniform(0,self.upLimit/2)
			else:
				newNum=random.uniform(0,self.upLimit)
			list_y.append(newNum)
			judgeList_y.append(0)
		
		
		return list_y,judgeList_y


	def generateTimeListData(self,mean_x=0,mean_y=0,var_x=0.1,var_y=0.1):
		#self.a=self.generateBinaryWeight(self.n)
		#self.b=self.generateBinaryWeight(self.m)
		#self.c=self.generateBinaryWeight(self.s)
		self.a=self.generateFloatWeight(self.n)
		weight_y=self.generateFloatWeight(self.m+self.s)
		self.b=weight_y[:self.m]
		self.c=weight_y[self.m:]
		self.init_x=self.generateInit(self.n)
		self.init_y=self.generateInit(self.m)
		judgeList_x=[1]*self.n
		judgeList_y=[1]*self.m
		list_x=self.init_x
		list_y=self.init_y
		
		if self.s>self.n:
			for i in range(self.s-self.n):
				list_x,judgeList_x=self.generateTimeListData_x(list_x,judgeList_x,self.a)
		elif self.s<self.n:
			for j in range(self.n-self.s):
				list_y,judgeList_y=self.generateTimeListData_y(list_x,list_y,judgeList_y,self.b,self.c)

		while len(list_x)<self.lengthOfData and len(list_y)<self.lengthOfData:
			list_x,judgeList_x=self.generateTimeListData_x(list_x,judgeList_x,self.a)
			list_y,judgeList_y=self.generateTimeListData_y(list_x,list_y,judgeList_y,self.b,self.c)
	
		return list_x,list_y,judgeList_x,judgeList_y



	
