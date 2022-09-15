'''
COMPARISON BETWEEN RCF AND SCINTILLATOR MEASUREMENTS HIGH DOSE SETTING
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from scipy import optimize
from scipy.optimize import curve_fit
import csv
import pandas as pd
from readcsv import read_datarcf
from readcsv import read_data_let
from readcsv import read_data_let_scintillator


################################################################################



#DATA

#high dose
#directory='pictures/2020-09-24/notnormalized/'
directory='pictures/2020-11-05/highdose/notnormalized/'
filename=directory+'notnormalizedmean_array1.npy'
data= np.load(filename)
dose=data[0:len(data)-3]
tof=data[len(data)-3:len(data)]

filename1=directory+'notnormalizederr1.npy'
data1= np.load(filename1)
err=data1[0:len(data1)]

print(len(data))
filename2=directory+'1Dtopprojection/'+'top1Dmean_array1.npy'
top_projection_dose= np.load(filename2)
#tof=data[len(data)-3:len(data)]
tofrcf=[310,110] #KBKG
path='pictures/2020-11-05/highdose/RCF20KBKG.csv'
#tofrcf=[320,100] #LLLM
#path='pictures/2020-11-05/highdose/RCF20LLLM.csv'
depth,rcfdose=read_datarcf(path)

rcferr=rcfdose*3/100

#norm=(rcfdose[0]*tof[0])/(tofrcf[0]*dose[0])
norm=(rcfdose[0])/dose[0]
#norm1D=(rcfdose[0]*tof[0])/(tofrcf[0]*top_projection_dose[0])
norm1D=rcfdose[0]/top_projection_dose[0]










################################################################################
#PLOT RAW DATA


plt.figure(1)
plt.plot( np.arange(0,len(dose),1)*0.0634,
                                                     top_projection_dose*norm1D,
                                                                           '.',
                                                                  Markersize=11,
                                               label='Top projection measured ')
#plt.plot(                            np.arange(0,len(dose),1)*0.0634,
#                                                          dose[::-1],
#                                                                 '.',
#                                                        markersize=7,
#                                              label='{}'.format(tof))
plt.errorbar(  np.arange(0,len(dose),1)*0.0634,                  dose*norm ,
                                                                  yerr=err*norm,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                        label='{} reconstructed'.format(tof))
plt.fill_between(np.arange(0,len(dose),1)*0.0634,
                                                             dose*norm-err*norm,
                                                           dose*norm + err*norm,
                                                        color='gray', alpha=0.5)


plt.grid(True)

plt.errorbar( depth ,rcfdose,                                       #wepl =1.28 already there
                                                                    yerr=rcferr,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                ecolor='orange',
                                                                elinewidth=None,
                                              label='RCF LLLM {}'.format(tofrcf))
plt.legend( title='Unidose',fontsize='large')
plt.title('Depth-dose distribution high dose setting 11-05',
                                                                  fontdict=None,
                                                                  loc='center',
                                                                       pad=None)
plt.xlabel('Depth[mm]')
plt.ylabel('Dose[Gy]')










################################################################################
#LET correction simulations in RCF


#Data

dS=0 #theoretical values
dscintillator= 1.023 #[g/cm^3] scintillator density
dactivelayer=1.2 #[g/cm^3]
k=207/10000 #[g/MeV cm^2]
a=0.9 #scintillator efficiency
dx= 65 #µm scintillator soatial resolution
ddx=1 #spatial resolution error



#SIMULATED LET data  ( we expect a function LET(depth))

path='pictures/2020-11-05/highdose/LET20KBKG.csv'  #highdose
#path='pictures/2020-11-05/lowdose/LET20EQ.csv'
#path='pictures/2020-11-05/highdose/LET20LLLM_september.csv'  #highdose september
#path='pictures/2020-11-05/highdose/LET20LLLM.csv'
xs,ys_iterative,ys_iterative_upper,ys_iterative_lower,ys_ana,ys_ana_upper,ys_ana_lower,sim_redose=read_data_let(path)  #LET simulated in rcf

dy_iterative=(ys_iterative_upper-ys_iterative_lower)/2
dy_ana=(ys_ana_upper-ys_ana_lower)/2


xs=xs*1.03#simulated depth IN scintillator
ys_iterative_upper=ys_iterative_upper*(dscintillator/dactivelayer) #conversion
ys_iterative_lower=ys_iterative_lower*(dscintillator/dactivelayer)
ys_iterative=ys_iterative*(dscintillator/dactivelayer)

ys_ana=ys_ana*(dscintillator/dactivelayer)
ys_ana_lower=ys_ana_lower*(dscintillator/dactivelayer)
ys_ana_upper=ys_ana_upper*(dscintillator/dactivelayer)

#INTERPOLATION
S_i=np.interp(np.arange(0,len(dose),1)*0.0634,xs,ys_iterative)
S_i_low=np.interp(np.arange(0,len(dose),1)*0.0634,xs,ys_iterative_lower)
S_i_up=np.interp(np.arange(0,len(dose),1)*0.0634,xs,ys_iterative_upper)
dS_i=(S_i_up-S_i_low)/2

S_a=np.interp(np.arange(0,len(dose),1)*0.0634,xs,ys_ana)
S_a_low=np.interp(np.arange(0,len(dose),1)*0.0634,xs,ys_ana_lower)
S_a_up=np.interp(np.arange(0,len(dose),1)*0.0634,xs,ys_ana_upper)
dS_a=(S_a_up-S_a_low)/2

#SIMULATED LET IN Scintillator


#path='pictures/2020-11-05/lowdose/LET20EQ_simulatedinscintillator.csv'#low dose
#path='pictures/2020-11-05/highdose/LET20FL_simulatedinscintillator.csv'#highdose
#rcfdose,ys_iterative_sci,lightratio=read_data_let_scintillator(path)
#S_i_sci=np.interp(np.arange(0,len(dose),1)*0.0634,xs,ys_iterative_sci)


###############################################################################
#FUNCTION

def lightout(S,a,k,dx):
    return ((a*S)/(1+(k*S)))*dx             #Birk's Model
#def lightout_err(S,dS,a,k,dx,ddx):
#    return (a/(1+(k*S))*np.sqrt((dx**2)*(dS**2))+((S**2)*(ddx**2)))    #we must propagate the error on birk model

def lightcorrection(S,a,k,dx):
    return lightout(S,a,k,dx)/lightout(S,a,0,dx)  #Birk's Model quenching correction / linear trend

#def lightcorrection_err(S,dS,a,k,dx,ddx):
#    return np.abs(lightcorrection(S,a,k,dx))*np.sqrt(( (lightout_err(S,dS,a,k,dx,ddx)/lightout(S,a,k,dx))**2)+((lightout_err(S,dS,a,0,dx,ddx)/lightout(S,a,0,dx))**2))
#

def dosecorrection(S,a,k,dx):
    return (dose/lightcorrection(S,a,k,dx) )  #S1 should be a function of depth, i can not make a fit but a ican interpolate, data are not normalized

#def dosecorrection_err(S,dS,a,k,dx,ddx):
#    return (np.abs(dosecorrection(S,a,k,dx))
#            *np.sqrt(((err/dose)**2)+
#                    ((lightcorrection_err(S,dS,a,k,dx,ddx)/lightcorrection(S,a,k,dx))**2)))



#NORMALIZATION
#S=S*dscintillator /10 #[keV/micormeters]

k=k/dscintillator*10#[micrometers/kev]


norm_i=((rcfdose.max()*tof[0])/(tof[0]*dosecorrection(S_i,a,k,dx).max())) #high
norm_a=((rcfdose.max()*tof[0])/(tof[0]*dosecorrection(S_a,a,k,dx).max())) #high
#norm_i_sci=((rcfdose.max()*tof[0])/((tof[0]*dosecorrection(S_i_sci,a,k,dx).max())))
norm=(rcfdose.max()*tof[0])/(tof[0]*dose.max())

#CORRECTED DOSE
D_i=dosecorrection(S_i,a,k,dx)*norm_i
D_i_up=dosecorrection(S_i_up,a,k,dx)*norm_i
D_i_low=dosecorrection(S_i_low,a,k,dx)*norm_i
D_a=dosecorrection(S_a,a,k,dx)*norm_a
D_a_up=dosecorrection(S_a_up,a,k,dx)*norm_a
D_a_low= dosecorrection(S_a_low,a,k,dx)*norm_a
#D_i_sci=dosecorrection(S_i_sci,a,k,dx)*norm_i_sci


################################################################################
#PLOT


plt.figure(2)
plt.plot(xs,ys_iterative,'.',
                                              label=' iterative deconvolution ',
                                                                  Markersize=14,
                                                                  color='blue',)
plt.fill_between(xs,
                                                           ys_iterative_lower,
                                                                 ys_iterative,
                                                      color='blue', alpha=0.1)

plt.fill_between(xs,
                                                                 ys_iterative,
                                                            ys_iterative_upper,
                                                       color='blue', alpha=0.1)
plt.plot(xs,ys_ana,'.',
                                             label=' analytical deconvolution ',
                                                   Markersize=14,color='orange')
plt.fill_between(xs,
                                                                        ys_ana,
                                                                  ys_ana_upper,
                                                     color='orange', alpha=0.1)
plt.fill_between(xs,
                                                                   ys_ana_lower,
                                                                         ys_ana,
                                                      color='orange', alpha=0.1)

plt.title('LET simulation ',fontsize=18)
plt.ylabel('Stopping Power [KeV/µm]',fontsize=16)
plt.xlabel(' Depth in PVT scintillator[mm] ',fontsize=16)
plt.grid(True)
plt.legend()

'''
plt.figure(3)
plt.subplot(121)

plt.plot(S,lightout(S,a,k,dx),'.',label='quenching correction',Markersize=14)
plt.title("Birks' Model",fontsize=18)
plt.xlabel('Stopping Power [KeV/µm]',fontsize=16)
plt.ylabel('Light output ',fontsize=16)
plt.xlim([0,8])
plt.ylim([0,300])
plt.plot(S,lightout(S,a,0,dx),'',label='kB=0 linear trend')
plt.legend()
plt.grid(True)

plt.subplot(122)

plt.plot(S,lightcorrection(S,a,k,dx),'.',label='Ratio',Markersize=14)
plt.xlim([0,8])
plt.xlabel('Stopping Power [KeV/µm]',fontsize=16)
plt.title('Real light output vs ideal light output',fontsize=18)
plt.tick_params(axis='x', which='major', labelsize=18)
plt.tick_params(axis='y', which='major', labelsize=18)
plt.legend()
plt.grid(True)
'''
plt.figure(4)
#plt.plot( np.arange(0,len(dose),1)*0.0634*1.03,top_projection_dose*norm_i,'.'
#                                          ,Markersize=11,label='Top projection')
plt.errorbar(  np.arange(0,len(dose),1)*0.0634,                      dose*norm_a ,
                                                                  yerr=err*norm_a,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                  ecolor='gray',
                                                                elinewidth=None, #11/05
                                            label='Measured dose TOF[300,100]'
#                                              label='Measured dose TOF[434,114]'  #09/24
                                                                ,Markersize=11)



plt.plot(  np.arange(0,len(dose),1)*0.0634,
                                                                           D_i,
                                                                            '.',
                                                                      color='m',
               label='Quenching corrected dose  iterative method',
                                                                 Markersize=11)

plt.fill_between( np.arange(0,len(dose),1)*0.0634,
                                                                            D_i,
                                                                         D_i_up,
                                                                      color='m',
                                                                      alpha=0.2)
plt.fill_between( np.arange(0,len(dose),1)*0.0634,
                                                                        D_i_low,
                                                                            D_i,
                                                                      color='m',
                                                                      alpha=0.2)

plt.plot(  np.arange(0,len(dose),1)*0.0634,
                                                                           D_a,
                                                                            '.',
                                                                 color='orange',
              label='Quenching corrected dose  analytical method',
                                                                  Markersize=11)
plt.fill_between( np.arange(0,len(dose),1)*0.0634,
                                                                            D_a,
                                                                         D_a_up,
                                                                 color='orange',
                                                                      alpha=0.2)
plt.fill_between( np.arange(0,len(dose),1)*0.0634,
                                                                        D_a_low,
                                                                            D_a,
                                                                 color='orange',
                                                                      alpha=0.2)

#plt.plot(  np.arange(0,len(dose),1)*0.0634,
#                                                                        D_i_sci,
#                                                                            '.',
#                                                                     color='red',
#label='Quenching corrected dose  with simulation in scintillator iterative method',
#                                                                  Markersize=11)

#plt.plot( depth ,rcfdose_7mm,
#                                                                           '.' ,                     #wepl =1.28 already there
##                                                                    color='y',
#                                                       label='RCF EQ 7mm TOF[]',
#                                                                  Markersize=11)
plt.plot( depth ,rcfdose,
                                                                           '.' ,                     #wepl =1.28 already there
                                                                     color='g',
#                                                label='RCF 20LLLM TOF[320,100]',
                                                label='RCF 20KBKG TOF[310,110]',
                                                                Markersize=11)
plt.fill_between(depth,
                                                                rcfdose-rcferr,
                                                              rcfdose + rcferr,
                                                 color='lightgreen', alpha=0.5)
plt.legend( title='',fontsize='large')

plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=18)
plt.tick_params(axis='y', which='major', labelsize=18)
plt.ylabel('Dose [Gy]',fontsize=18)
plt.xlabel('Depth in PVT[mm]',fontsize=18)
plt.title('Scintillator shape comparison  05-11 high dose shot n.65',fontsize=22)
#plt.title('Scintillator shape comparison  24-09 high dose shot n.84',fontsize=22)
'''
plt.figure(5)

nor=lightratio.max()/lightcorrection(ys_iterative,a,k,dx).max()

plt.plot( xs,lightratio,
                                                                            '.',
                                                                  markersize=12,
                        label='ideal/measured (LET simulated in scintillator) ')
plt.plot(xs,lightcorrection(ys_iterative,a,k,dx),
                                                                           '.',
                                                                  markersize=12,
label='ideal/measured (LET simulated in rcf aand then converted ) not interpolated')
plt.plot( np.arange(0,len(dose),1)*0.0634,
                                                    lightcorrection(S_i,a,k,dx),
                                                                            '.',
             label='ideal/measured (LET simulated in rcf and then converted ) ')
plt.title('Light correction high dose setting',fontsize=20)
plt.ylabel('ratio',fontsize=18)
plt.xlabel('Depth in PVT[mm]',fontsize=18)
plt.legend(fontsize=16)
'''
plt.figure(6)
plt.subplot(121)

norm_sim=dose.max()/sim_redose.max()
norm_1D=dose.max()/top_projection_dose.max()
plt.errorbar(  np.arange(0,len(dose),1)*0.0634,
                                                                      dose*0.9 ,
                                                                   yerr=err*0.9,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                  ecolor='gray',
                                                                elinewidth=None,
                                         label='Reconstructed dose TOF[300,100]'
#                                        label='Reconstructed dose TOF[434,114]'
                                                                 ,Markersize=11)
plt.plot(depth,                                             sim_redose*norm_sim,
                                                                            '.',
                                                                    color='red',
                                             label='Simulated dose TOF[310,110] '
                                                                ,Markersize=11)
plt.plot( np.arange(0,len(dose),1)*0.0634,
                                                   top_projection_dose*norm_1D,
                                                                           '.',
                                                                 Markersize=11,
                                 label='Top projection measuredTOF[300,100] ')
plt.legend()
plt.xlabel('Depth in PVT[mm]',fontsize=18)
plt.ylabel('Intensity',fontsize=18)
plt.title('Shape comparison  05-11 high dose shot n65 background sub',fontsize=18)

#plt.title('Scintillator shape comparison  25-09 shot n84 background sub.',fontsize=18)
plt.grid()
plt.subplot(122)

sim=np.interp(np.arange(0,len(dose),1)*0.0634,depth,sim_redose)
plt.plot(np.arange(0,len(dose),1)*0.0634,
                                                             dose- sim*norm_sim,
                                                                            '.',
                                                                    color='red',
                                                  label=' measured - simulated '
                                                                ,Markersize=11)

plt.legend()
plt.xlabel('Depth in PVT[mm]',fontsize=18)
plt.ylabel('Intensity',fontsize=18)
plt.title('Background estimation  05-11 high dose shot n65 ',fontsize=18)
plt.title('Scintillator shape comparison  25-09 shot n84 background sub.',
                                                                    fontsize=18)
plt.grid()









plt.show()
