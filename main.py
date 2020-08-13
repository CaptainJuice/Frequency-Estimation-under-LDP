import DE
import UE
import HE
import BLH
import OLH
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d=128
epsilon_choice=np.linspace(0.5,5,10)
Var=np.zeros((6,10),dtype='float32')
id=0
for epsilon in (epsilon_choice):
    Data=np.zeros((6,100),dtype='float32')
    for count in np.arange(100,dtype='int8'):
        print(id,count)
        DE.generatedata(d)
        data=pd.read_csv('user-data.csv')
        data_ori=data                     #stoer original data
        Freq_ori=DE.getFreq(data,d)       #store original frequency
        data=data.iloc[:,1]               #process data
        #Direct Encoding
        PE=DE.Perturb(DE.Encode(data),d,epsilon)
        Freq_DE=DE.Aggregate(d,epsilon,PE)
        Data[0,count]=Freq_DE[0]
        #Symmetry Histogram Encoding
        PE=HE.Perturb(HE.Encode(data_ori,d),d,epsilon)
        Freq_SHE=HE.Aggregate_SHE(PE)
        Data[1,count]=Freq_SHE[0]
        #Symmetry Unary Encoding
        p_sue=np.exp(epsilon/2)/(1+np.exp(epsilon/2))
        q_sue=1-p_sue
        PE_SUE=UE.Perturb(UE.Encode(data_ori,d),d,p_sue,q_sue)
        Freq_SUE=UE.Aggregate(PE_SUE,p_sue,q_sue)
        Data[2,count]=Freq_SUE[0]
        #Optimal Unary Encoding
        p_oue=1/2
        q_oue=1/(1+np.exp(epsilon))
        PE_OUE=UE.Perturb(UE.Encode(data_ori,d),d,p_oue,q_oue)
        Freq_OUE=UE.Aggregate(PE_OUE,p_oue,q_oue)
        Data[3,count]=Freq_OUE[0]
        #binary local hashing
        g=2
        E=BLH.Encoding(data,d,g)
        PE=BLH.Perturb(E,d,epsilon,g)
        Freq_BLH=BLH.Aggregate(PE,d,epsilon,g)
        Data[4,count]=Freq_BLH[0]
        #Optimal local hashing
        g=np.ceil(np.exp(epsilon)+1)
        E=OLH.Encoding(data,d,g)
        PE=OLH.Perturb(E,d,epsilon,g)
        Freq_OLH=OLH.Aggregate(PE,d,epsilon,g)
        Data[5,count]=Freq_OLH[0]
    Var[:,id]=np.var(Data,axis=1)
    id+=1
fig,axis=plt.subplots()
axis.plot(epsilon_choice,np.log10(Var[0,:]),'ro-',label='DE')
axis.plot(epsilon_choice,np.log10(Var[1,:]),'bo-',label='SHE')
axis.plot(epsilon_choice,np.log10(Var[2,:]),'go-',label='SUE')
axis.plot(epsilon_choice,np.log10(Var[3,:]),'yo-',label='OUE')
axis.plot(epsilon_choice,np.log10(Var[4,:]),'ko-',label='BLH')
axis.plot(epsilon_choice,np.log10(Var[5,:]),'co-',label='OLH')
leg=axis.legend()
plt.title('fixing d=128 vary epsilon')
plt.xlabel('epsilon')
plt.ylabel('Variance(log10)')
plt.show()




