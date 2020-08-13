import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#Unary Encoding
def generatedata(d) :
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

def Perturb(data,d,p,q):
    N=int(data.shape[0])
    new_data=np.zeros((N,d))
    for i in np.arange(N):
        for j in np.arange(d):
            if bool(data[i,j]):
                new_data[i,j]=np.random.choice([0,1],1,p=[1-p,p])[0]
            else:
                new_data[i,j]=np.random.choice([0,1],1,p=[1-q,q])[0]
    return new_data
def Aggregate(data,p,q):
    n=data.shape[0]
    return (np.sum(data==1,axis=0)-n*q)/(p-q)

if __name__=='__main__':
    d=15   #domin size
    epsilon=3
    p_sue=np.exp(epsilon/2)/(1+np.exp(epsilon/2))
    q_sue=1-p_sue
    p_oue=1/2
    q_oue=1/(1+np.exp(epsilon))
    generatedata(d)
    data=pd.read_csv("user-data.csv")
    Freq_ori=getFreq(data,d)
    PE_SUE=Perturb(Encode(data,d),d,p_sue,q_sue)
    PE_OUE=Perturb(Encode(data,d),d,p_oue,q_oue)
    Freq_SUE=Aggregate(PE_SUE,p_sue,q_sue)
    Freq_OUE=Aggregate(PE_OUE,p_oue,q_oue)
    x=np.arange(1,d+1,1)
    plt.figure(figsize=(20,8),dpi=80)
    bar_width = 0.15
    b1=plt.bar(x,Freq_OUE,width=bar_width,color='blue',label="OUE")
    b2=plt.bar(x+0.15,Freq_SUE,width=bar_width,color='green',label="SUE")
    b3=plt.bar(x+0.15*2,Freq_ori,width=bar_width,color='red',label="real")
    plt.legend((b1,b2,b3), ('OUE','SUE','Real'))
    plt.title("Unary Encoding")
    plt.show()
