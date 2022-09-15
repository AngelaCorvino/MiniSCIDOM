'''
COMPARISON BETWEEN TOF SIMULATED DATA IN SCINTILLATOR  AND SCINTILLATOR MEASUREMENTS
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy import optimize
from scipy.optimize import curve_fit
import csv
import pandas as pd
from readcsv import read_spectrum
from readcsv import read_tof
from readcsv import read_datarcf
from readcsv import read_data_let
from readcsv import read_data_let_scintillator
from readcsv import read_data_let_mini
from readcsv import read_data_scintillatorsimulateddose
from readcsv import read_data_scintillatorsimulateddose_it
from readcsv import read_data_mini
from Birkmodel import lightcorrection
from Birkmodel import dosecorrection
from Birkmodel import letcorrection
from readnpy import read_dose
from readnpy import read_doserr
import seaborn as sns

#DATA


#low dose september
directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/'
filename='RCF20KDKE.csv' #all cylinder
rcfdepth,rcfdose,rcferr,area_rcf,rcfname=read_datarcf(directory,filename)
norm_rcf=1/rcfdose.max()

s=0.0634

directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/notnormalized/'
dose1,depth1,tof1,number1=read_dose(directory,'notnormalizedmean_array91.npy',s)
err1=read_doserr(directory,'notnormalizederr91.npy')
area1=np.trapz(dose1[3:len(dose1)], depth1[3:len(depth1)])

dose2,depth2,tof2,number2=read_dose(directory,'notnormalizedmean_array92.npy',s)
err2=read_doserr(directory,'notnormalizederr92.npy')
area2=np.trapz(dose2[3:len(dose2)], depth2[3:len(depth2)])


dose3,depth3,tof3,number3=read_dose(directory,'notnormalizedmean_array93.npy',s)
err3=read_doserr(directory,'notnormalizederr93.npy')
area3=np.trapz(dose3[3:len(dose3)], depth3[3:len(depth3)])




"TOF SIMULATED DATA IN SCINTILLATOR and  RCF MEASUREMENTS analitical "
path1='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/TOFinscintillator_91_ana.csv'
path2='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/TOFinscintillator_92_ana.csv'
path3='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/TOFinscintillator_93_ana.csv'
path4='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/oldtof/TOFinscintillator_91_ana.csv'
depth_sci1,dose_sci1,dose_sci_upper1,dose_sci_lower1,ys_ana_mini1,ys_ana_upper_mini1,ys_ana_lower_mini1=read_tof(path1)
depth_sci2,dose_sci2,dose_sci_upper2,dose_sci_lower2,ys_ana_mini2,ys_ana_upper_mini2,ys_ana_lower_mini2=read_tof(path2)
depth_sci3,dose_sci3,dose_sci_upper3,dose_sci_lower3,ys_ana_mini3,ys_ana_upper_mini3,ys_ana_lower_mini3=read_tof(path3)


area_sci1=np.trapz(dose_sci1[0:len(dose_sci1)-3], depth_sci1[0:len(depth_sci1)-3])
area_sci2=np.trapz(dose_sci2[0:len(dose_sci2)-3], depth_sci2[0:len(depth_sci2)-3])
area_sci3=np.trapz(dose_sci3[0:len(dose_sci3)-3], depth_sci3[0:len(depth_sci3)-3])

#norm1=area_sci1/area1
#norm2=area_sci2/area2
#norm3=area_sci3/area3

#norm1=1/dose1.max()
#norm2=1/dose2.max()
#norm3=1/dose3.max()
#norm4=1/dose4.max()

#norm_sim1=1/dose_sci1.max()
#norm_sim2=1/dose_sci2.max()
#norm_sim3=1/dose_sci3.max()



"""LET CORRECTION"""
D_a_mini1,D_a_up_mini1,D_a_low_mini1,area_corrected1,S_a_mini1=letcorrection(depth_sci1,dose1,ys_ana_mini1,ys_ana_lower_mini1,ys_ana_upper_mini1,s)
D_a_mini2,D_a_up_mini2,D_a_low_mini2,area_corrected2,S_a_mini2=letcorrection(depth_sci2,dose2,ys_ana_mini2,ys_ana_lower_mini2,ys_ana_upper_mini2,s)
D_a_mini3,D_a_up_mini3,D_a_low_mini3,area_corrected3,S_a_mini3=letcorrection(depth_sci3,dose3,ys_ana_mini3,ys_ana_lower_mini3,ys_ana_upper_mini3,s)
###############################################################################

#norm_a_mini1=area_sci1/area_corrected1





def plotfunction(dose,err,norm,color1,color2,depth_sci,dose_sci,norm_sim,dose_sci_lower,dose_sci_upper,D_a_mini,norm_a_mini,number,rcfdepth,rcfdose,norm_rcf,s):
    plt.errorbar( np.arange(0,len(dose),1)*s,                      dose*norm ,
                                                                      yerr=err*norm,
                                                                          xerr=None,
                                                                            fmt='.',
                                                                     markersize=12,
                                                                     color=color1,
                                                                     ecolor=color1,
                                                                    elinewidth=None,
                                                                    label='Dose Miniscidom {}'.format(number),
                                                                    zorder=1)

    plt.fill_between(np.arange(0,len(dose),1)*s,
                                                                 dose*norm-err*norm,
                                                               dose*norm+ err*norm,
                                                                    color=color1,
                                                                            alpha=0.2)


    """
    plt.plot( depth_sci,dose_sci*norm_sim,
                                                                                '.',
                                                                      markersize=12,
                                                                      color=color2,
                                                                   label='ToF pre. {}'.format(number),
                                                                   zorder=2)
    plt.plot( np.arange(0,len(dose),1)*s,D_a_mini*norm_a_mini,
        '.',
                                                                          markersize=12,
                                                                          color='orange',
                                        label='LET correction {}'.format(number),
                                                                      zorder=2)
    """

    plt.errorbar( rcfdepth,                                          rcfdose*norm_rcf,
                                                                yerr=rcferr*norm_rcf,
                                                                          xerr=None,
                                                                            fmt='.',
                                              color=sns.color_palette(  "Paired")[1],
                                                                        markersize=12,
                                             ecolor=sns.color_palette(  "Paired")[1],
                                                                     elinewidth=None,
                                                     label=' Dose {} '.format(rcfname))

    plt.fill_between(rcfdepth,
                                                    rcfdose*norm_rcf-rcferr*norm_rcf,
                                                    rcfdose*norm_rcf+rcferr*norm_rcf,
                                              color=sns.color_palette(  "Paired")[1],
                                                                           alpha=0.1)

    return







plt.figure(1)

norm1=1/dose1.max()
norm3=1/dose3.max()
norm_a_mini1=1/D_a_mini1.max()
norm_a_mini3=1/D_a_mini3.max()
norm_sim1=1/dose_sci1.max()
norm_sim3=1/dose_sci3.max()
#plotfunction(dose1,err1,norm1,sns.color_palette("Paired")[8],sns.color_palette( "Paired")[9],depth_sci1,dose_sci1,norm_sim1,dose_sci_lower1,dose_sci_upper1,D_a_mini1,norm_a_mini1,number1,rcfdepth,rcfdose,norm_rcf,s)
#plotfunction(dose2,err2,area_sci2/area2,sns.color_palette("Paired")[2],sns.color_palette( "Paired")[3],depth_sci2,dose_sci2,1,dose_sci_lower2,dose_sci_upper2,number2,s)
plotfunction(dose3,err3,norm3,sns.color_palette("Paired")[8],sns.color_palette( "Paired")[9],depth_sci3,dose_sci3,norm_sim3,dose_sci_lower3,dose_sci_upper3,D_a_mini3,norm_a_mini3,number3,rcfdepth,rcfdose,norm_rcf,s)
#plotfunction(dose4,err4,norm4,sns.color_palette("Paired")[4],sns.color_palette( "Paired")[5],depth_sci4,dose_sci4,1,dose_sci_lower4,dose_sci_upper4,number4,s)


plt.legend(title='',fontsize=22,loc=1,markerscale=2)
plt.title('ToF predicted dose inside scintillator ',fontsize=24,
                                                                  fontdict=None,
                                                                  loc='center',
                                                                  pad=None)

plt.xlabel('Depth[mm]',fontsize=24)
plt.ylabel('Absolute Dose [Gy]',fontsize=24)
plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=22)
plt.tick_params(axis='y', which='major', labelsize=22)











"""
"PIXEL MEAN VALUE "


P_a_mini=dosecorrection(dose,S_a_mini,a,k,dx)      #mean pixel value corrected withLET value
P_a_up_mini=dosecorrection(dose,S_a_up_mini,a,k,dx)
P_a_low_mini=dosecorrection(dose,S_a_low_mini,a,k,dx)


"CALIBRATION FACTOR "
#Step 1 : interpolate dosce_sci in order to divide it by P_a_mini
dose_sci_interp=np.interp(np.arange(0,len(dose),1)*s,depth_sci,dose_sci)
#Step 2: divide
c=dose_sci_interp/P_a_mini         #this is an arry (simulate dose in scintillatore/ mean pixel value corrected withLET value )
#Step 3 : average the vector
c_mean=np.mean(c)

print(c_mean)
###############################################################################
#PLOT

# create figure and axis objects with subplots()





fig, ax = plt.subplots()
ax2 = ax.twinx()



ax.errorbar(  np.arange(0,len(dose),1)*s,                           dose ,
                                                                       yerr=err,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                       label='Reconstruction mean pixel value  ')

ax.plot(  np.arange(0,len(dose),1)*s,
                                                                        P_a_mini,
                                                                            '.',
                                                                 color='orange',
                              label='LET corrected mean pixel value Miniscidom',
                                                                  Markersize=11)

ax.fill_between( np.arange(0,len(dose),1)*s,
                                                                        P_a_mini,
                                                                     P_a_up_mini,
                                                                 color='orange',
                                                                      alpha=0.3)

ax.fill_between( np.arange(0,len(dose),1)*s,
                                                                    P_a_low_mini,
                                                                        P_a_mini,
                                                                 color='orange',
                                                                      alpha=0.3)


ax2.plot( depth_sci,dose_sci,
                                                                            'g.',
                                                                  Markersize=11,
                                                    label='Simulated Dose TOF ')
ax2.fill_between(depth_sci,
                                                        dose_sci_lower*norm_sim,
                                                        dose_sci_upper*norm_sim,
                                                color='lightgreen', alpha=0.2)






#ax2.set_ylim([0,dose_sci.max()])
#ax.set_ylim([0,P_a_mini.max()])
ax.set_ylabel("Mean Pixel  [a.u.]",color="blue",fontsize=16)
ax2.set_ylabel("Dose [Gy]",color="Green",fontsize=16)
plt.title('Mean pixel value  12-03 low dose shot n3',fontsize=18)

ax.legend( title='',fontsize='large',loc=2)
ax2.legend( title='',fontsize='large',loc=1)

ax2.grid(b=True,color='k',linestyle='-',alpha=0.2)
ax.grid(b=True,color='k',linestyle='dotted',alpha=0.2)

ax.tick_params(axis='x', which='major', labelsize=16)
ax.tick_params(axis='y', which='major',colors='blue', labelsize=16)
ax2.tick_params(axis='y', which='major',colors='green', labelsize=16)
"""

"TOF SIMULATED DATA IN SCINTILLATOR with Quenching effect "

path1='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/ToF_FUKA_sims_dose_quench_miniSCI_25_09_20_shot_91.csv'
path2='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/ToF_FUKA_sims_dose_quench_miniSCI_25_09_20_shot_92.csv'
path3='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/ToF_FUKA_sims_dose_quench_miniSCI_25_09_20_shot_93.csv'

depth_sim,realdose1,quenchedose1,ys1=read_data_mini(path1)
depth_sim,realdose2,quenchedose2,ys2=read_data_mini(path2)
depth_sim,realdose3,quenchedose3,ys3=read_data_mini(path3)


D_a_mini1,D_a_up_mini1,D_a_low_mini1,area_corrected1,S_a_mini1=letcorrection(depth_sci1,dose1,ys_ana_mini1,ys_ana_lower_mini1,ys_ana_upper_mini1,s)



def letcorrection(depth_sci,dose,ys_ana_mini,ys_ana_lower_mini,ys_ana_upper_mini,s):

    dS=0 #theoretical values
    dscintillator= 1.023 #[g/cm^3] scintillator density
    dactivelayer=1.2 #[g/cm^3]
    k=207/10000 #[g/MeV cm^2]
    a=0.9 #scintillator efficiency
    dx= 65 #µm scintillator spatial resolution
    ddx=1 #spatial resolution error

    x=np.arange(0,len(dose),1)*s
    S_a_mini=np.interp(np.arange(0,len(dose),1)*s,depth_sci,ys_ana_mini)
    S_a_low_mini=np.interp(np.arange(0,len(dose),1)*s,depth_sci,ys_ana_lower_mini)
    S_a_up_mini=np.interp(np.arange(0,len(dose),1)*s,depth_sci,ys_ana_upper_mini)


    #CORRECTED DOSE

    D_a_mini=dosecorrection(dose,S_a_mini,a,k,dx)
    D_a_up_mini=dosecorrection(dose,S_a_up_mini,a,k,dx)
    D_a_low_mini=dosecorrection(dose,S_a_low_mini,a,k,dx)

    #NORMALIZATION
    area_corrected=np.trapz(D_a_mini[3:len(D_a_mini)],x[3:len(x)])


    return D_a_mini,D_a_up_mini,D_a_low_mini,area_corrected,S_a_mini









D_a_mini,D_a_up_mini,D_a_low_mini,area_corrected,S_a_mini=letcorrection(depth_sim,quenchedose1,ys1,ys1,ys1,s)
plt.figure(5)
plt.errorbar(  np.arange(0,len(dose1),1)*s,                       dose1/dose1.max(),
                                                                  yerr=err1/dose1.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                                             label=' measured')
plt.plot(  np.arange(0,len(dose1),1)*s,                       D_a_mini1/D_a_mini1.max(),

                                                            color='black',
                                                             label='let correction of miniscidom ')
plt.plot(  np.arange(0,len(quenchedose1),1)*s,                       D_a_mini/D_a_mini.max(),

                                                            color='black',
                                                             label='let correction of quenched dose')


plt.plot( depth_sim,quenchedose1,
                                                                            'g.',
                                                                   markersize=8,
                                              label=' simulated quenched curve')
plt.plot( depth_sim,realdose1,
                                                                            'r.',
                                                                    markersize=8,
                                                      label=' simulated  curve')

plt.title('Shape comparsison low dose setting 12-03 shot n3',
                                                                  fontdict=None,
                                                                   loc='center',
                                                                       pad=None)
plt.xlabel('Depth[mm]')
plt.ylabel('Dose[Gy]')
plt.grid(True)
plt.legend()


















plt.show()
