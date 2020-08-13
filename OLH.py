import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xxhash

def generatedata(d) :
    N=10000#user number
    data=pd.Series(np.random.randint(1,d+1,N,dtype="int8"))
    data.to_csv("user-data.csv")

def getFreq(data,d):
    F=np.zeros(d)
    for i in np.arange(int(data.shape[0])):
        F[data.iloc[i,1]-1]+=1
    return F

def Encoding(data,d,g):
    n=int(data.shape[0])
    Hashtable=np.zeros((n,d),dtype='int32')
    newdata=np.zeros(n,dtype='int32')
    for i in np.arange(n,dtype='int32'):
        seed=i
        index =int(data[i]-1)
        newdata[seed] = (xxhash.xxh32(str(index), seed=seed).intdigest() % g)
    return newdata

def Perturb(data,d,epsilon,g):
    p=np.exp(epsilon)/(g-1+np.exp(epsilon))
    q=1/(g-1+np.exp(epsilon) )
    n=int(data.shape[0])
    for i in np.arange(n):
        p_sample = np.random.random_sample()
        if p_sample > p - q:
            y = np.random.randint(0, g)
            data[i]=y
    return data

def Aggregate(data,d,epsilon,g):
    p=0.5
    q=1/g
    n=int(data.shape[0])
    F=np.zeros(d)
    for ii in np.arange(n):
        for jj in np.arange(d,dtype='int32'):
            seed=ii
            if((xxhash.xxh32(str(jj), seed=seed).intdigest() % g)==data[ii]):
                F[jj]+=1
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    return a*F-b

if __name__=="__main__":
    d=15
    epsilon=2
    g=np.ceil(np.exp(epsilon)+1)
    generatedata(d)
    data=pd.read_csv("user-data.csv")
    Freq_ori=getFreq(data,d)     #get original frequency
    data=data.iloc[:,1]
    #generate hash functions
    E=Encoding(data,d,g)
    PE=Perturb(E,d,epsilon,g)
    Freq=Aggregate(PE,d,epsilon,g)
    print(Freq)
    x=np.arange(1,d+1,1)
    plt.figure(figsize=(20,8),dpi=80)
    bar_width = 0.25
    b1=plt.bar(x,Freq,width=bar_width,color='blue',label="OLH")
    b2=plt.bar(x+0.25,Freq_ori,width=bar_width,color='red',label="real")
    plt.legend((b1, b2), ('OLH', 'Real'))
    plt.title("Optimal local hashing")
    plt.show()


