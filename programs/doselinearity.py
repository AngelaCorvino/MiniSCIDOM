import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import logging
import argparse
from scipy import optimize
from scipy.optimize import curve_fit
import csv



def read_dataMC(path):
 #data = pd.read_excel (path)
    ''' read measured rcf  data from execel document'''
    data=pd.read_csv(path,header=None,skiprows=1,delimiter=',')
    x=(data[0])# s
    MCdose=(data[1])
    MCerr=(data[1])*33/1000
    stdv=(data[2])
    err=np.sqrt(MCerr**2+stdv**2)
    return x,MCdose,err

directory='/home/corvin22/Desktop/miniscidom/pictures/2021-08-13/'
filename1=directory+'Dosevstime.csv'
filename2=directory+'Dosevscurrent.csv'

time,MCdose1,err1=read_dataMC(filename1)
current,MCdose2,err2=read_dataMC(filename2)



norm=1
m=0.0508




####Function

def linear(x,a,b):
    return a*x+b


def quadratic(x,a,b,c):
    return (a*(x**2))+b*x+c




#FIT
param_bounds=([-np.inf,-np.inf],[np.inf,np.inf])
popt = (1,0) #paramentro usato nel fit precedente

popt, pcov = curve_fit(linear,time,MCdose1, p0=popt,bounds=param_bounds)
a,b= popt
da,db = np.sqrt(np.diag(pcov))


# #####PLOT DOSE vs TIME

f, ( ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

print(f' a : {a:.2f}')
print(f' b : {b:.2f}')
print( 'Fit model : Linear')
#Residui + Chi2
res_par = (MCdose1 - linear(time,*popt))
chi2 = (res_par**2).sum()
chi2_ridotto= chi2/(len(MCdose1)-len(popt))



#ax1=plt.subplots(111)
ax1.errorbar( time,                      MCdose1*norm ,
                                                                  yerr=err1*norm,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='green',
                                                                        markersize=15,
                                                                   ecolor='green',
                                                                elinewidth=None,
                                                                label='MC measurements')

ax1.fill_between(time,
                                                             MCdose1*norm-err1*norm,
                                                           MCdose1*norm + err1*norm,
                                                        color='green', alpha=0.2)

ax1.plot(time,linear(time,*popt),                               color='darkblue',
                                                                    markersize=15,
                                                                    label='Linear Fit')
#plt.title(f'Fit estimation: a = {a:.2f} $\pm$ {da:.2f}, b = {b:.3f} $\pm$ {db:.3f}')
ax1.legend( title=f'Linear model estimation: a = {a:.1f} $\pm$ {da:.1f}, b = {b:.0f} $\pm$ {db:.0f}',fontsize='18')
ax1.set_title('Dose VS Exposure time',
                                                                 fontsize=22,
                                                                 loc='center',
                                                                       pad=None)


ax1.set_ylim([20,900])
#ax1.set_xlim([0.5,25.5])
#ax1.set_xlabel('MC exposure time[s] ',fontsize=20)
ax1.set_ylabel('Dose[mGy]',fontsize=20)
ax1.grid(b=True,color='k',linestyle='dotted',alpha=0.2)

#ax2 = plt.subplot(212, sharex=ax1)

plt.ylabel(" Residuals", size=20)
plt.minorticks_on()
plt.grid()



ax2.plot(time, res_par, color='black', marker='.',
                                                                markersize=15,
                                                                linestyle='--')
#plt.errorbar(c2, res_7par,yerr=dh2, color='black', marker='.', linestyle='none', label='7 par')
ax2.grid(b=True,color='k',linestyle='dotted',alpha=0.2)
ax2.set_xlabel('MC exposure time[s] ',fontsize=20)


plt.legend(fontsize=10)

plt.show()








# ##### FIT DOSE vs CURRENT
param_bounds=([-np.inf,-np.inf],[np.inf,np.inf])
popt = (1,0) #paramentro usato nel fit precedente

popt, pcov = curve_fit(linear,current,MCdose2, p0=popt,bounds=param_bounds)
a,b= popt
da,db = np.sqrt(np.diag(pcov))




#PLOT
f, ( ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})

print(f' a : {a:.2f}')
print(f' b : {b:.2f}')
print( 'Fit model : Linear')
#Residui + Chi2
res_par = (MCdose2 - linear(current,*popt))
chi2 = (res_par**2).sum()
chi2_ridotto= chi2/(len(MCdose2)-len(popt))



#ax1=plt.subplots(111)
ax1.errorbar( current,                      MCdose2*norm ,
                                                                  yerr=err2*norm,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='green',
                                                                        markersize=15,
                                                                   ecolor='green',
                                                                elinewidth=None,
                                                                label='MC measurements')

ax1.fill_between(current,
                                                             MCdose2*norm-err2*norm,
                                                           MCdose2*norm + err2*norm,
                                                        color='green', alpha=0.2)

ax1.plot(current,linear(current,*popt),                               color='darkblue',
                                                                    markersize=15,
                                                                    label='Linear Fit')
#plt.title(f'Fit estimation: a = {a:.2f} $\pm$ {da:.2f}, b = {b:.3f} $\pm$ {db:.3f}')
ax1.legend( title=f'Linear model estimation: a = {a:.1f} $\pm$ {da:.1f}, b = {b:.0f} $\pm$ {db:.0f}',fontsize='18')
ax1.set_title('Dose VS Tube current',
                                                                 fontsize=22,
                                                                 loc='center',
                                                                       pad=None)


ax1.set_ylim([20,900])
#ax1.set_xlim([0.5,25.5])
#ax1.set_xlabel('MC exposure time[s] ',fontsize=20)
ax1.set_ylabel('Dose[mGy]',fontsize=20)
ax1.grid(b=True,color='k',linestyle='dotted',alpha=0.2)

#ax2 = plt.subplot(212, sharex=ax1)

plt.ylabel(" Residuals", size=20)
plt.minorticks_on()
plt.grid()



ax2.plot(current, res_par, color='black', marker='.',
                                                                markersize=15,
                                                                linestyle='--')
#plt.errorbar(c2, res_7par,yerr=dh2, color='black', marker='.', linestyle='none', label='7 par')
ax2.grid(b=True,color='k',linestyle='dotted',alpha=0.2)
ax2.set_xlabel('Tube current[mA] ',fontsize=20)


plt.legend(fontsize=10)

plt.show()
