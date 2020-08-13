import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# BASIC RAPPOR
def generatedata(d) :
    N=10#user number
    data=pd.Series(np.random.randint(1,d+1,N,dtype="int8"))
    data.to_csv("user-data.csv")

def getFreq(data,d):
    F=np.zeros(d)
    for i in np.arange(int(data.shape[0])):
        F[data.iloc[i,1]-1]+=1
    return F
def Encode(data,d):
    N=int(data.shape[0])
    new_data=np.zeros((N,d),dtype="int8")
    for i in np.arange(N):
        new_data[i,data.iloc[i][1]-1]=1
    return new_data

def Perturb(data,d):
    N=int(data.shape[0])
    f=0.5
    p=0.7
    q=0.3
    # Permanent randomized response
    new_data=np.zeros((N,d),dtype="int8")
    for i in np.arange(N):
        for j in np.arange(d):
            if bool(data[i,j]):
                new_data[i,j]=np.random.choice([0,1],1,p=[f/2,1-f/2])[0]
            else:
                new_data[i,j]=np.random.choice([0,1],1,p=[1-f/2,f/2])[0]
    #: Instantaneous randomized response
    for i in np.arange(N):
        for j in np.arange(d):
            if bool(data[i,j]):
                data[i,j]=np.random.choice(np.arange(2),size=1,replace=True,p=[p,q])[0]
            else:
                data[i,j]=np.random.choice(np.arange(2),size=1,replace=True,p=[q,p])[0]
    return new_data

def Aggregate(data):
    f= 0.5
    n=data.shape[0]
    new_data=(data.sum(axis=0)-f*n/2)/(1-f)
    return new_data

if __name__=='__main__':
    d=64   #domin size
    generatedata(d)
    data=pd.read_csv("user-data.csv")
    Freq_ori=getFreq(data,d)
    PE=Perturb(Encode(data,d),d)
    Freq=Aggregate(PE)
    x=np.arange(1,d+1,1)
    plt.figure(figsize=(20,8),dpi=80)
    bar_width = 0.25
    b1=plt.bar(x,Freq,width=bar_width,color='blue',label="Rappor")
    b2=plt.bar(x+0.25,Freq_ori,width=bar_width,color='red',label="real")
    plt.legend((b1, b2), ('Basic Rappor', 'Real'))
    plt.title("Basic Rappor")
    plt.show()
