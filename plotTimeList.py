#! /usr/bin/python

import pickle
import matplotlib.pyplot as plt
import numpy as np

def readData(filename):
	dataFile=open(filename,"rb")
	dictionary=pickle.load(dataFile)

	x_list=dictionary["x_list"]
	y_list=dictionary["y_list"]
	x_truth=dictionary["x_truth"]
	y_truth=dictionary["y_truth"]
	a=dictionary["a"]
	b=dictionary["b"]
	c=dictionary["c"]



	return x_list,y_list,x_truth,y_truth

def plotData(filename,begin=0):
	
	x_list,y_list,x_truth,y_truth=readData(filename)
	length=min(len(x_list),len(y_list))
	x_listPlot=x_list[-length+begin:]
	y_listPlot=y_list[-length+begin:]
	x_truthPlot=x_truth[-length:]
	y_truthPlot=y_truth[-length:]

	x=np.arange(0,len(x_listPlot))
	plt.subplot(211)
	m=plt.plot(x,x_listPlot,color="r")
	plt.subplot(212)
	n=plt.plot(x,y_listPlot)
	plt.show()

#def plotPcaData(

def plotAccuracyRate():
	from scipy.interpolate import interp1d
	plt.subplot(1,2,1)
	sampleNum=np.arange(2,11)*100

	sampleAcc=[65.337078651681459,64.4776119403,62.808988764,61.70068027210885,58.582230623818523,58.971861471861472,58.504273504273504,59.07407407407407,58.2272727273]
	f2 = interp1d(sampleNum, sampleAcc)
	plt.ylabel('Accuracy Rate %')
	plt.xlabel('Number of Sample Points')
	plt.ylim(0,100)
	sampleNum_new=np.arange(200,1100,200)
	plt.plot(sampleNum,sampleAcc,"o",sampleNum,sampleAcc,"-")
	plt.title('Accuracy Rate')
	plt.subplot(1,2,2)

	
	embededAcc=[54.8127128263,52.1793416572,62.0620506999,58.2272727273,60.1251422071,53.7055685280,53.557582668187]
	plt.ylabel('Accuracy Rate %') 	
	plt.xlabel('Length of Embeded Vector')
	plt.ylim(0,100)
	embededNum=np.arange(2,9)
	plt.plot(embededNum,embededAcc,"o",embededNum,embededAcc,"-")
	embededVar=[]
	plt.show()
	

