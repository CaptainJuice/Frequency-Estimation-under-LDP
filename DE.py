import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# Direct Encoding
def generatedata(d ) :
    N=100 #user number
    data=pd.Series(np.random.randint(1,d+1,N,dtype="int32"))
    data.to_csv("user-data.csv")

def getFreq(data,d):
    F=np.zeros(d)
    for i in np.arange(int(data.shape[0])):
        F[data.iloc[i,1]-1]+=1
    return F

def Encode(data):
    return data

def Perturb(data,d,epsilon):
    N=int(data.shape[0])
    new_data=np.zeros(N)
    for i in np.arange(N):
        pro=np.full(d,1/(np.exp(epsilon)+d-1))
        pro[data[i]-1]=np.exp(epsilon)/(np.exp(epsilon)+d-1)
        new_data[i]=np.random.choice(np.linspace(1,d,d,dtype="int8"),1,p=pro)
    return new_data

def Aggregate(d,epsilon,data):
    p=np.exp(epsilon)/(np.exp(epsilon)+d-1)
    q=(1-p)/(d-1)
    n=data.shape[0]
    new_data=np.zeros(d,dtype='float32')
    for i in np.arange(d,dtype='int8'):
        new_data[i]=np.sum(data==(i+1))
    new_data=(new_data-n*q)/(p-q)
    return new_data

if __name__=='__main__':
    """
    #Frequency check :Succss
    d=15  #domin size
    epsilon=4
    generatedata(d)
    data=pd.read_csv("user-data.csv")
    Freq_ori=getFreq(data,d)
    data=data.iloc[:,1]
    PE=Perturb(Encode(data),d,epsilon)
    Freq=Aggregate(d,epsilon,PE)
    x=np.arange(1,d+1,1)
    plt.figure(figsize=(20,8),dpi=80)
    bar_width = 0.25
    b1=plt.bar(x,Freq,width=bar_width,color='blue',label="DE")
    b2=plt.bar(x+0.25,Freq_ori,width=bar_width,color='red',label="real")
    plt.legend((b1, b2), ('DE', 'Real'))
    plt.title("Direct Encoding")
    plt.show()
    """
    d_choice=np.array([2,4,16,128,2048])
    Var=np.zeros((5,10),dtype='float32')
    i=0
    for d in d_choice:
        epsilon_choice=np.linspace(0.5,5,10)
        j=0
        for epsilon in  epsilon_choice:
            Data_temp=np.zeros(100,dtype='float32')
            for count in np.arange(100,dtype='int8'):
                generatedata(d)
                data=pd.read_csv("user-data.csv")
                Freq_ori=getFreq(data,d)
                data=data.iloc[:,1]
                PE=Perturb(Encode(data),d,epsilon)
                Freq=Aggregate(d,epsilon,PE)
                Data_temp[count]=Freq[0]
            Var[i,j]=np.var(Data_temp)
            j+=1
        i+=1
    fig,axis=plt.subplots()
    Var=np.log10(Var)
    axis.plot(epsilon_choice,Var[0,:],'ro-',label='d=2')
    axis.plot(epsilon_choice,Var[1,:],'bo-',label='d=4')
    axis.plot(epsilon_choice,Var[2,:],'go-',label='d=16')
    axis.plot(epsilon_choice,Var[3,:],'yo-',label='d=128')
    axis.plot(epsilon_choice,Var[4,:],'ko-',label='d=2048')
    leg=axis.legend()
    plt.title('fixing d vary epsilon')
    plt.xlabel('epsilon')
    plt.ylabel('Variance（log10)')
    plt.show()







