import HE
import DE
import BLH
import OLH
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


epsilon=4
d_choice=2**np.linspace(2,14,7)
Var_emp=np.zeros((3,7),dtype='float32')
Var_ana=np.zeros((3,7),dtype='float32')
id=0
for d in d_choice:
    d=int(d)
    DE.generatedata(d)
    data=pd.read_csv('user-data.csv')
    data_ori=data                     #stoer original data
    Freq_ori=DE.getFreq(data,d)       #store original frequency
    data=data.iloc[:,1]               #process data
    #Symmetry Histogram Encoding
    PE=HE.Perturb(HE.Encode(data_ori,d),d,epsilon)
    Freq_SHE=HE.Aggregate_SHE(PE)
    Var_emp[0,id]=np.mean((Freq_SHE-Freq_ori)**2)
    Var_ana[0,id]=100*8/epsilon**2
    #Binary Local Hashing
    g=2
    E=BLH.Encoding(data,d,g)
    PE=BLH.Perturb(E,d,epsilon,g)
    Freq_BLH=BLH.Aggregate(PE,d,epsilon,g)
    Var_emp[1,id]=np.mean((Freq_BLH-Freq_ori)**2)
    Var_ana[1,id]=100*((np.exp(epsilon)+1)/(np.exp(epsilon)-1))**2
    #Optimal Local Hashing
    g=np.ceil(np.exp(epsilon)+1)
    E=OLH.Encoding(data,d,g)
    PE=OLH.Perturb(E,d,epsilon,g)
    Freq_OLH=OLH.Aggregate(PE,d,epsilon,g)
    Var_emp[2,id]=np.mean((Freq_OLH-Freq_ori)**2)
    Var_ana[2,id]=100*np.exp(epsilon)*4/(np.exp(epsilon)-1)**2
    id+=1
fig,axis=plt.subplots()
axis.plot(np.linspace(2,14,7),np.log10(Var_emp[0,:]),'ro-',label='Empirical SHE')
axis.plot(np.linspace(2,14,7),np.log10(Var_emp[1,:]),'bo--',label='Empirical BLH')
axis.plot(np.linspace(2,14,7),np.log10(Var_emp[2,:]),'go-',label='Empirical OLH')
axis.plot(np.linspace(2,14,7),np.log10(Var_ana[0,:]),'yo-',label='Analytical SHE')
axis.plot(np.linspace(2,14,7),np.log10(Var_ana[1,:]),'ko-',label='Analytical BLH')
axis.plot(np.linspace(2,14,7),np.log10(Var_ana[2,:]),'co-',label='Analytical OLH')
leg=axis.legend()
plt.title('fixing epsilon=4 vary d')
plt.xlabel('d(log2')
plt.ylabel('Variance(log10)')
plt.show()




