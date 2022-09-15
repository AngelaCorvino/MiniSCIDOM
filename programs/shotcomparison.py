import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import logging
import argparse
from scipy import optimize
from scipy.optimize import curve_fit
import csv
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    "-log",
    "--log",
    default="warning",
    help=(
        "Provide logging level. "
        "Example --log debug', default='warning'"),
    )

options = parser.parse_args()
levels = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}
level = levels.get(options.log.lower())
if level is None:
    raise ValueError(
        f"log level given: {options.log}"
        f" -- must be one of: {' | '.join(levels.keys())}")
logging.basicConfig(level=level)
logger = logging.getLogger(__name__)


#high dose

directory='pictures/2020-11-05/lowdose/notnormalized/'



data= np.load(directory+'notnormalizedmean_array43.npy')
dose=data[0:len(data)-3]

#tof1=data1[len(data1)-3:len(data1)]
unidose=0
data0= np.load(directory+'notnormalizederr43.npy')
err=data0[0:len(data0)]
#top_projection_dose= np.load(directory+'1Dtopprojection/'+'top1Dmean_array1.npy')




filename1=directory+'notnormalizedmean_array42.npy'
data1= np.load(filename1)
dose1=data1[0:len(data1)-3]

#tof1=data1[len(data1)-3:len(data1)]
unidose1=0
data2= np.load(directory+'notnormalizederr42.npy')
err1=data2[0:len(data2)]
top_projection_dose1= np.load(directory+'1Dtopprojection/'+'top1Dmean_array42.npy')


data4= np.load(directory+'notnormalizedmean_array44.npy')
dose2=data4[0:len(data4)-3]

#tof2=data4[len(data4)-3:len(data4)]
unidose2=0
data5= np.load(directory+'notnormalizederr44.npy')
err2=data5[0:len(data5)]
top_projection_dose2= np.load(directory+'1Dtopprojection/'+'top1Dmean_array44.npy')



data6= np.load(directory+'notnormalizedmean_array45.npy')
dose3=data6[0:len(data6)-3]

#tof2=data4[len(data4)-3:len(data4)]
unidose3=0
data7= np.load(directory+'notnormalizederr45.npy')
err3=data7[0:len(data7)]
top_projection_dose3= np.load(directory+'1Dtopprojection/'+'top1Dmean_array45.npy')





norm=dose2.max()/dose.max()
norm1=dose2.max()/dose1.max()
norm3=dose2.max()/dose3.max()


















'''
def read_data_let_scintillator(path):

    #read simulation in scintillator data from execel document

    data=pd.read_csv(path,header=None,skiprows=1,delimiter=',')
    xs=np.array(data[0])
    dose_ana_sci=np.array(data[1])  # depth in scintillator
    sim_redose=np.array(data[2])
    ys_ana_sci=np.array(data[3]) # tof simulation in scintillator
    lightratio=np.array(data[4])

    return xs,dose_ana_sci,sim_redose,ys_ana_sci,lightratio
'''
plt.figure(1)
#plt.plot( np.arange(0,len(dose),1)*0.0634,
#                                                           top_projection_dose,
#                                                                           '.',
#                                                                  Markersize=11,
#                                               label='Top projection measured ')
#plt.plot( np.arange(0,len(dose),1)*0.0634,
#                                                                    dose[::-1],
#                                                                           '.',
#                                                                  markersize=7,
#                                                        label='{}'.format(tof))
plt.errorbar(  np.arange(0,len(dose2),1)*0.0634,                         dose/dose.max() ,
                                                                      yerr=err/dose.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                label='{} reconstructed shot 43 '.format(unidose))
plt.errorbar(  np.arange(0,len(dose2),1)*0.0634,                         dose2/dose2.max() ,
                                                                      yerr=err2/dose2.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                label='{} reconstructed shot 44 '.format(unidose2))
plt.errorbar(  np.arange(0,len(dose1),1)*0.0634,                  dose1/dose1.max(),
                                                                yerr=err1/dose1.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                label='{} reconstructed shot 42 '.format(unidose1))
#plt.fill_between(np.arange(0,len(dose1),1)*0.0634,
#                                                         dose1-err1,
#                                                       dose1 + err1,
#                                                        color='gray', alpha=0.5)

plt.errorbar(  np.arange(0,len(dose2),1)*0.0634,                         dose3/dose3.max() ,
                                                                      yerr=err3/dose3.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                label='{} reconstructed shot 45 '.format(unidose3))

plt.ylim([0,1.1])
plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()

plt.legend( title='Unidose',fontsize='large')
plt.title('Depth-dose distribution low dose setting shot 42,43, 44, 45',
                                                                  fontdict=None,
                                                                  loc='center',
                                                                       pad=None)
plt.xlabel('Depth[mm]')
plt.ylabel('Intensity')
plt.legend()







'''
plt.figure(2)
path='pictures/2020-09-24/Scintillator_quenching_2.csv'
xs,dose_ana_sci,sim_redose,ys_ana_sci,lightratio=read_data_let_scintillator(path)
#norm_sim=(dose2[-1]/sim_redose[-1])
norm_sim=(dose2.max()/sim_redose.max())
plt.errorbar(  np.arange(0,len(dose2),1)*0.0634,                          dose2,
                                                                      yerr=err2,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                  ecolor='gray',
                                                                elinewidth=None,
                                         label='Reconstructed dose TOF[434,114]'
                                                                ,Markersize=11)
plt.plot(xs,                                              sim_redose*norm_sim ,
                                                                            '.',
                                                                  color='red',
                                                        label='Simulated dose '
                                                                ,Markersize=11)

plt.legend(fontsize='large')
plt.xlabel('Depth in PVT[mm]',fontsize=18)
plt.ylabel('Intensity',fontsize=18)
plt.title('Scintillator shape comparison 09-24 reconstruction vs simulation shot n84 ',
                                                                    fontsize=22)
plt.grid()
'''
plt.show()













plt.show()
