import DE
import UE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d=128
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
    #Direct Encoding
    PE=DE.Perturb(DE.Encode(data),d,epsilon)
    Freq_DE=DE.Aggregate(d,epsilon,PE)
    Var_emp[0,id]=np.mean((Freq_DE-Freq_ori)**2)
    Var_ana[0,id]=100*(d-2+np.exp(epsilon))/(np.exp(epsilon)-1)**2
    #Symmetry Unary Encoding
    p_sue=np.exp(epsilon/2)/(1+np.exp(epsilon/2))
    q_sue=1-p_sue
    PE_SUE=UE.Perturb(UE.Encode(data_ori,d),d,p_sue,q_sue)
    Freq_SUE=UE.Aggregate(PE_SUE,p_sue,q_sue)
    Var_emp[1,id]=np.mean((Freq_SUE-Freq_ori)**2)
    Var_ana[1,id]=100*np.exp(epsilon/2)/(np.exp(epsilon/2)-1)**2
    #Optimal Unary Encoding
    p_oue=1/2
    q_oue=1/(1+np.exp(epsilon))
    PE_OUE=UE.Perturb(UE.Encode(data_ori,d),d,p_oue,q_oue)
    Freq_OUE=UE.Aggregate(PE_OUE,p_oue,q_oue)
    Var_emp[2,id]=np.mean((Freq_OUE-Freq_ori)**2)
    Var_ana[2,id]=4*100*np.exp(epsilon)/(np.exp(epsilon)-1)**2
    id+=1
fig,axis=plt.subplots()
axis.plot(np.linspace(2,14,7),np.log10(Var_emp[0,:]),'ro-',label='Empirical DE')
axis.plot(np.linspace(2,14,7),np.log10(Var_emp[1,:]),'bo--',label='Empirical SUE')
axis.plot(np.linspace(2,14,7),np.log10(Var_emp[2,:]),'go-',label='Empirical OUE')
axis.plot(np.linspace(2,14,7),np.log10(Var_ana[0,:]),'yo-',label='Analytical DE')
axis.plot(np.linspace(2,14,7),np.log10(Var_ana[1,:]),'ko-',label='Analytical SUE')
axis.plot(np.linspace(2,14,7),np.log10(Var_ana[2,:]),'co-',label='Analytical OUE')
leg=axis.legend()
plt.title('fixing epsilon=4 vary d')
plt.xlabel('d(log2')
plt.ylabel('Variance(log10)')
plt.show()




