import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#Histogram Encoding
def generatedata(d ) :
    N=5000 #user number
    data=pd.Series(np.random.randint(1,d+1,N,dtype="int32"))
    data.to_csv("user-data.csv")

def getFreq(data,d):
    F=np.zeros(d)
    for i in np.arange(int(data.shape[0])):
        F[data.iloc[i,1]-1]+=1
    return F

def Encode(data,d):
    N=int(data.shape[0])
    new_data=np.zeros((N,d),dtype="int32")
    for i in np.arange(N):
        new_data[i,data.iloc[i][1]-1]=1
    return new_data

def Perturb(data,d,epsilon):
    N=int(data.shape[0])
    new_data=np.zeros((N,d))
    for i in np.arange(0,N,dtype='int32'):
        for j in np.arange(d,dtype='int32'):
            new_data[i,j]=data[i,j]+np.random.laplace(0,2/epsilon,1)[0]
    return new_data
def Aggregate_SHE(data):
    return np.sum(data,axis=0)

def Aggregate_THE(data,d,epsilon,theta):
    n=data.shape[0]
    p=1-np.exp(epsilon*(theta-1)/2)/2
    q=np.exp(-1*theta*epsilon/2)/2
    f=(data>theta).sum(axis=0)
    return (f-n*q)/(p-q)


if __name__=='__main__':
    d=10   #domin size
    epsilon=4
    theta=1
    generatedata(d)
    data=pd.read_csv("user-data.csv")
    Freq_ori=getFreq(data,d)
    PE=Perturb(Encode(data,d),d,epsilon)
    Freq_SHE=Aggregate_SHE(PE)
    Freq_THE=Aggregate_THE(PE,d,epsilon,theta)
    x=np.arange(1,d+1,1)
    plt.figure(figsize=(20,8),dpi=80)
    bar_width = 0.15
    b1=plt.bar(x,Freq_SHE,width=bar_width,color='blue',label="SHE")
    b2=plt.bar(x+0.15,Freq_THE,width=bar_width,color='green',label="THE")
    b3=plt.bar(x+0.15*2,Freq_ori,width=bar_width,color='red',label="real")
    plt.legend((b1,b2,b3), ('SHE','THE','Real'))
    plt.title("Histogram Encoding")
    plt.show()
