import HE
import DE
import BLH
import OLH
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


d=128
epsilon_choice=np.linspace(0.5,5,10)
Var_emp=np.zeros((3,10),dtype='float32')
Var_ana=np.zeros((3,10),dtype='float32')
id=0
for epsilon in epsilon_choice:
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
axis.plot(epsilon_choice,np.log10(Var_emp[0,:]),'ro-',markerfacecolor='none',label='Empirical SHE')
axis.plot(epsilon_choice,np.log10(Var_emp[1,:]),'bo--',markerfacecolor='none',label='Empirical BLH')
axis.plot(epsilon_choice,np.log10(Var_emp[2,:]),'go-',markerfacecolor='none',label='Empirical OLH')
axis.plot(epsilon_choice,np.log10(Var_ana[0,:]),'yo-',markerfacecolor='none',label='Analytical SHE')
axis.plot(epsilon_choice,np.log10(Var_ana[1,:]),'ko-',markerfacecolor='none',label='Analytical BLH')
axis.plot(epsilon_choice,np.log10(Var_ana[2,:]),'co-',markerfacecolor='none',label='Analytical OLH')
leg=axis.legend()
plt.title('fixing epsilon=4 vary d')
plt.xlabel('d(log2)')
plt.ylabel('Variance(log10)')
plt.show()




