
'''
COMPARISON BETWEEN MINISCIDOM RECONSTRUCTION  and RCF MEASUREMENTS
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
from scipy import optimize
from scipy.optimize import curve_fit
import csv
import pandas as pd
from readcsv import read_datarcf
from readcsv import read_data_let
from readcsv import read_data_let_scintillator
from readcsv import read_data_let_mini
from readcsv import read_data_scintillatorsimulateddose
from readcsv import read_data_scintillatorsimulateddose_it
from readcsv import read_data_mini
from readcsv import read_datarcf_simulation
from Birkmodel import lightcorrection
from Birkmodel import dosecorrection
from readnpy import read_dose
from readnpy import read_doserr
###############################################################################

#DATA


directory='pictures/2021-08-13/'
#filename='RCF85P_3.csv' #half cylinder
filename='RCF85O_2.csv' #all cylinder
rcfdepth,rcfdose,rcferr,area_rcf,rcfname=read_datarcf(directory,filename)

rcfdepth=rcfdepth/1.3
area_rcf=np.trapz(rcfdose[0:len(rcfdose)-1], rcfdepth[0:len(rcfdepth)-1])


filename='X_Ray_spectrum_200kV.csv'
data=pd.read_csv(filename,header=None,skiprows=0,delimiter=';')
energy=data[0]
intensity=data[1]
intensity_interpolated=np.interp(np.arange(0,200,0.5),energy,intensity)


s=0.0508 #[mm/pixel ]
directory=directory+'notnormalized/'
#dose,depth,tof,number=read_dose(directory,'notnormalizedmean_array137.npy',s) #half cylinder
dose,depth,tof,number=read_dose(directory,'notnormalizedmean_array34.npy',s)

#err=read_doserr(directory,'notnormalizederr137.npy') #half cylinder#standard deviation divided by radical n
err=read_doserr(directory,'notnormalizederr34.npy')
area=np.trapz(dose[6:len(dose)-40], depth[6:len(depth)-40])
norm= 1/dose.max()
norm_rcf=1/rcfdose.max()
#norm= 1/area
#norm_rcf=1/area_rcf
#norm=area_rcf/area
#norm_rcf=1

x= np.arange(0,len(dose),1)*s


fig, ax = plt.subplots()
ax2 = ax.twinx()


ax.errorbar( rcfdepth,                                          rcfdose*norm_rcf,
                                                            yerr=rcferr*norm_rcf,
                                                                      xerr=None,
                                                                        fmt='.',
                                          color=sns.color_palette(  "Paired")[1],
                                                                    markersize=8,
                                         ecolor=sns.color_palette(  "Paired")[1],
                                                                 elinewidth=None,
                                                 label=' RCF{}  V=250V  I=16mA  t=40s'.format(rcfname))

ax.fill_between(rcfdepth,
                                                rcfdose*norm_rcf-rcferr*norm_rcf,
                                                rcfdose*norm_rcf+rcferr*norm_rcf,
                                          color=sns.color_palette(  "Paired")[1],
                                                                       alpha=0.1)


"""
ax.errorbar( rcfdepth,                                          rcfdose1*norm_rcf1,
                                                            yerr=rcferr1*norm_rcf1,
                                                                      xerr=None,
                                                                        fmt='.',
                                          color=sns.color_palette(  "Paired")[2],
                                                                    markersize=8,
                                         ecolor=sns.color_palette(  "Paired")[2],
                                                                 elinewidth=None,
                                                 label=' RCF85O_3   V=200V  I=20mA  t=60s')

ax.fill_between(rcfdepth,
                                                rcfdose1*norm_rcf1-rcferr1*norm_rcf1,
                                                rcfdose1*norm_rcf1+rcferr1*norm_rcf1,
                                          color=sns.color_palette(  "Paired")[2],
                                                                       alpha=0.1)


"""







ax.errorbar(  np.arange(0,len(dose),1)*s,                              dose*norm,
                                                                   yerr=err*norm,
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[3],
                                         ecolor=sns.color_palette(  "Paired")[3],
                                                                 elinewidth=None,
                                             label=' Miniscidom reconstruction ')

ax.fill_between(np.arange(0,len(dose),1)*s,
                                                             dose*norm-err*norm,
                                                           dose*norm + err*norm,
                                          color=sns.color_palette(  "Paired")[3],
                                                                      alpha=0.5)




plt.title('Depth-dose distribution shot {} without depth coversion'.format(number),
                                                                  fontdict=None,
                                                                  fontsize=22,
                                                                  loc='center',
                                                                       pad=None)





#ax2.hlines(5, 0,10, colors='darkorange', linestyles='solid', label='5%')
#ax2.hlines(-5, 0,10, colors='darkorange', linestyles='solid', label='-5%')
ax.set_xlabel('Depth[mm]',fontsize=20)
ax.set_ylabel('Absolute Dose [Gy]',fontsize=20)
ax.legend( title='',fontsize=20,loc='upper right')#loc=3)
ax.grid(b=True,color='k',linestyle='dotted',alpha=0.2)

ax.tick_params(axis='x', which='major', labelsize=16)
ax.tick_params(axis='y', which='major', labelsize=16)
#ax2.tick_params(axis='y', which='major',colors='orange', labelsize=16)
plt.show()





####################PSTAR DATA
dscintillator= 1.023 #[g/cm^3] scintillator density
pstarPVT_data=np.loadtxt('PVTattenuationcoefficient.txt',skiprows=3,comments='#')
pstarPVT_data = np.array(pstarPVT_data)
mu_PVT=pstarPVT_data[:,1] #[g/cmquadro]
energyPVT=pstarPVT_data[:,0]
int_mu_PVT=np.interp(np.arange(0,230,0.3)/1000,energyPVT,mu_PVT)
#E_PVT_in=np.interp(np.arange(0,len(dose),1)*s,mu_PVT,energyPVT)


dRCF= 1.2 #[g/cm^3] scintillator density
pstarRCF_data=np.loadtxt('Gafchromicattenuationcoefficient.txt',skiprows=3,comments='#')
pstarRCF_data = np.array(pstarRCF_data)
mu_RCF=pstarRCF_data[:,1] #[cmquadro/g]
energyRCF=pstarRCF_data[:,0]


int_mu_RCF=np.interp(np.arange(0,230,0.3)/1000,energyRCF,mu_RCF)
#E_RCF_in=np.interp(np.arange(0,len(dose),1)*s,mu_RCF,energyRCF)


intensity_interpolated=np.interp(np.arange(0,230,0.3),energy,intensity)

plt.figure(2)
plt.subplot(211)

plt.plot(energy,intensity,'.',markersize=12, color=sns.color_palette(  "Paired")[8],
label=' Specrtum ')
plt.plot(np.arange(0,230,0.3),intensity_interpolated,'-',markersize=11,
                                                                        linewidth=3,
                                        color=sns.color_palette(  "Paired")[9],
                                                        label='  Interpolation ')




plt.title('X-Ray Tube Isovolt 320',
                                                                  fontdict=None,
                                                                  fontsize=22,
                                                                  loc='center',
                                                                       pad=None)

plt.yscale('linear')
plt.xscale('linear')
plt.grid(b=True,color='k',linestyle='dotted',alpha=0.2)
plt.tick_params(axis='x', which='major', labelsize=22)
plt.tick_params(axis='y', which='major', labelsize=22)
plt.xlabel('Energy[KeV]',fontsize=22)
plt.ylabel('Photon Flux(arbitr. units)', fontsize=22)
plt.legend( title='',fontsize=20,loc='upper right',markerscale=3)#loc=3)


plt.subplot(212)
plt.plot(np.arange(0,230,0.3),int_mu_PVT
                                                                ,'.',
                                                                markersize=11,
                                                                 color=sns.color_palette(  "Paired")[0],
                                                            label='PVT interpolation ')
plt.plot(energyPVT*1000,mu_PVT
                                                                ,'.',markersize=13,
                                                                color=sns.color_palette(  "Paired")[1],
                                                                label='PVT ')
#ax2.plot(xvals,interpolate_mu_PVT
#                                                                ,'.',label='interpolation PVT ')
plt.axvspan(energyRCF[8]*1000,energyRCF[18]*1000,
                                                              facecolor=sns.color_palette(  "Paired")[8],
                                                                      alpha=0.3,
                                                                    label=' Tube spectrum')

plt.plot(np.arange(0,230,0.3),int_mu_RCF
                                                                ,'.',
                                                                markersize=11,
                                                                 color=sns.color_palette(  "Paired")[2],
                                                            label='RCF interpolation ')


plt.plot(energyRCF*1000,mu_RCF
                                                                ,'.',
                                                                markersize=13,
                                                                 color=sns.color_palette(  "Paired")[3],
                                                                label='RCF ')

plt.yscale('log')
plt.xscale('log')
plt.ylabel('Attenuation coefficient[cm²/g] ',fontsize=22)
plt.grid(b=True,color='k',linestyle='dotted',alpha=0.2)
plt.tick_params(axis='x', which='major', labelsize=22)
plt.tick_params(axis='y', which='major', labelsize=22)
plt.xlabel('Energy[KeV]',fontsize=22)
plt.legend( title='',fontsize=20,loc='upper right',markerscale=3)#loc=3)
plt.show()
##############################################################################

ratio=int_mu_RCF/int_mu_PVT


intensity_interpolated=intensity_interpolated/intensity_interpolated.max()


mean_ratio=np.average(ratio, weights=intensity_interpolated)
print('Mean ratio:',mean_ratio)


rcfdepth=rcfdepth*mean_ratio
area_rcf=np.trapz(rcfdose[0:len(rcfdose)], rcfdepth[0:len(rcfdepth)])
area=np.trapz(dose[6:len(dose)-25], depth[6:len(depth)-25])


def linear(x,a,b):
    return a*x+b

param_bounds=([0,0],[np.inf,np.inf])
popt = (-1,0) #initial values
popt1, pcov1 = curve_fit(linear, rcfdepth[0:len( rcfdepth)], rcfdose[0:len(rcfdose)])#, p0=popt,bounds=param_bounds)
a1,b1= popt1
da1,db1 = np.sqrt(np.diag(pcov1))
areafit=np.trapz((rcfdepth*popt1[0]+popt1[1])[0:len(rcfdepth)], rcfdepth[0:len(rcfdepth)])

norm_rcf=1/(rcfdepth*popt1[0]+popt1[1]).max()
norm_fit=norm_rcf
#norm_fit=1/(rcfdepth*popt1[0]+popt1[1]).max()







fig, ax = plt.subplots()



ax.errorbar( rcfdepth,                                          rcfdose*norm_rcf,
                                                            yerr=rcferr*norm_rcf,
                                                                      xerr=None,
                                                                        fmt='.',
                                          color=sns.color_palette(  "Paired")[1],
                                                                    markersize=13,
                                         ecolor=sns.color_palette(  "Paired")[1],
                                                                 elinewidth=None,
                                                 label='$ D_{RCF}$')

ax.fill_between(rcfdepth,
                                                rcfdose*norm_rcf-rcferr*norm_rcf,
                                                rcfdose*norm_rcf+rcferr*norm_rcf,
                                          color=sns.color_palette(  "Paired")[1],
                                                                       alpha=0.1)

ax.plot(rcfdepth,(rcfdepth*popt1[0]+popt1[1])*norm_fit,
                                                                            '-',
                                                color='darkblue',
                                                                  linewidth=3,
                                                              label='RCF linear fit')

ax.errorbar(  np.arange(0,len(dose),1)*s,                              dose*norm,
                                                                   yerr=err*norm,
                                                                       xerr=None,
                                                                         fmt='.',
                                                                         markersize=13,
                                          color=sns.color_palette(  "Paired")[3],
                                         ecolor=sns.color_palette(  "Paired")[3],
                                                                 elinewidth=None,
                                             label='$D_{MS}$')

ax.fill_between(np.arange(0,len(dose),1)*s,
                                                             dose*norm-err*norm,
                                                           dose*norm + err*norm,
                                          color=sns.color_palette(  "Paired")[3],
                                                                      alpha=0.5)




plt.title('Depth dose distribution ',
                                                                  fontdict=None,
                                                                  fontsize=24,
                                                                  loc='center',
                                                                       pad=None)







ax.set_xlabel('Depth[mm]',fontsize=22)
ax.set_ylabel('Relative Dose  ',fontsize=22)
ax.legend( title='X-Ray Tube : V=250V  I=16mA  t=35s',fontsize=20,
                                                            loc='upper right',
                                                            title_fontsize=20,
                                                            markerscale=3)
ax.grid(b=True,color='k',linestyle='dotted',alpha=0.2)
ax.tick_params(axis='x', which='major', labelsize=22)
ax.tick_params(axis='y', which='major', labelsize=22)
#ax2.tick_params(axis='y', which='major',colors='orange', labelsize=16)
plt.show()
