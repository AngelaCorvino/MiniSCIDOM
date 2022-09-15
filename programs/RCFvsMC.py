"""RCF an MARCUS CHAMBER at ONCORAY"""
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
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
######################################################################### Data

calib1=21.49 #mGy/MU
calib2=24.06 #mGy/MU
depthMC=1.06 #mm
MU21CR=185.9
MU21CT=185.9
MU21CQ=186.1
MU21CP=46.44
MU21CY=1.30
MU21CW=1.30
MU21CX=1.3

calib6PC1PMMA=3078.69 #mGy/MU
calib6PC=2225.86 #mGy/MU
calib6PC_second=2195.05
calib7PC=4506.53
calib12PC= 22.66
calib10PC=24.13

dose121CR=calib1*MU21CR/1000
dose221CR=calib2*MU21CR/1000
dose21CT=calib12PC*MU21CT/1000
dose21CQ=calib10PC*MU21CQ/1000
dose21CP=calib10PC*MU21CP/1000

#print(dose121CR,dose221CR)

dose21CX=calib6PC*MU21CX/1000
dose21CX_second=calib6PC_second*MU21CX/1000
dose21CY=calib7PC*MU21CY/1000
dose21CW=calib6PC1PMMA*MU21CW/1000

directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-09-02/'

filename='RCF21CW.csv'
rcfdepth,rcfdose,rcferr,area_rcf,rcfname=read_datarcf(directory,filename)
normrcf=1/rcfdose.max()
#doserate=[4.78,9.80,19.74,39.87,80.11,158.72]
#doseMC=np.array([4,4.04,4.11,4.29,4.59,5.14])

###########################################################################
#DOSE RATE
ratio=rcfdose.max()/dose21CW
#So the task is now to multiply the MU dose with 0.7 and divide it by the irradiation time as a estimation of the Bragg peak dose rate in the miniSCIDOM.
MU=np.array([1.301,1.313,1.337,1.385,1.486,1.669])
doseMU=np.array([4.01,4.04,4.12,4.26,4.57,5.14])
irrtime=np.array([50.37,24.75,12.53,6.433,3.453,1.942])
doserate=doseMU*ratio/irrtime
#minisicdom data

s=0.073825 #[mm/pixel ]
directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-09-02/notnormalized/'
filename0 ='notnormalizedmean_array1.npy'
dose0,depth,tof,numbe0r=read_dose(directory,filename0,s)

err0=read_doserr(directory,'notnormalizederr1.npy') #standard deviation divided by radical n
area0=np.trapz(dose0[3:len(dose0)], depth[3:len(depth)])
#norm0=area_rcf/area0
norm0=1/dose0.max()
doserate0=doserate[0]

filename1 ='notnormalizedmean_array3.npy'
dose1,depth,tof1,number1=read_dose(directory,filename1,s)

err1=read_doserr(directory,'notnormalizederr3.npy') #standard deviation divided by radical n
area1=np.trapz(dose1[3:len(dose1)], depth[3:len(depth)])
#norm1=area_rcf/area1
norm1=1/dose1.max()
doserate1=doserate[1]



filename2 ='notnormalizedmean_array5.npy'
dose2,depth,tof2,number2=read_dose(directory,filename2,s)

err2=read_doserr(directory,'notnormalizederr5.npy') #standard deviation divided by radical n
area2=np.trapz(dose2[3:len(dose2)], depth[3:len(depth)])
#norm2=area_rcf/area2
norm2=1/dose2.max()
doserate2=doserate[2]



filename3 ='notnormalizedmean_array7.npy'
dose3,depth,tof3,number3=read_dose(directory,filename3,s)

err3=read_doserr(directory,'notnormalizederr7.npy') #standard deviation divided by radical n
area3=np.trapz(dose3[3:len(dose3)], depth[3:len(depth)])
#norm3=area_rcf/area3
norm3=1/dose3.max()
doserate3=doserate[3]


filename4 ='notnormalizedmean_array9.npy'
dose4,depth,tof4,number4=read_dose(directory,filename4,s)

err4=read_doserr(directory,'notnormalizederr9.npy') #standard deviation divided by radical n
area4=np.trapz(dose4[4:len(dose4)], depth[4:len(depth)])
#norm4=area_rcf/area4
norm4=1/dose4.max()
doserate4=doserate[4]

err4=read_doserr(directory,'notnormalizederr9.npy') #standard deviation divided by radical n
area4=np.trapz(dose4[4:len(dose4)], depth[4:len(depth)])
#norm4=area_rcf/area4
norm4=1/dose4.max()
doserate4=doserate[4]

filename5 ='notnormalizedmean_array15.npy'
dose5,depth,tof5,number5=read_dose(directory,filename5,s)

err5=read_doserr(directory,'notnormalizederr15.npy') #standard deviation divided by radical n
area5=np.trapz(dose5[4:len(dose5)], depth[4:len(depth)])
#norm5=area_rcf/area5
norm5=1/dose5.max()
doserate5=doserate[5]




def function():
    fig, ax = plt.subplots()
    plt.title('Does quenching depend on dose rate ?(6PC 1PMMA)',fontsize=24)



    ax.errorbar(np.arange(0,len(dose0),1)*s,                              dose0*norm0,
                                                                   yerr=err0*norm0,
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[3],
                                         ecolor=sns.color_palette(  "Paired")[3],
                                                                 elinewidth=None,
                                             label=f' Doserate={doserate0:.2f}Gy/s')

    ax.errorbar(np.arange(0,len(dose1),1)*s,                              dose1*norm1,
                                                                   yerr=err1*norm1,
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[4],
                                         ecolor=sns.color_palette(  "Paired")[4],
                                                                 elinewidth=None,
                                             label=f' Doserate={doserate1:.2f}Gy/s')


    ax.errorbar(np.arange(0,len(dose2),1)*s,                              dose2*norm2,
                                                                   yerr=err2*norm2,
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[5],
                                         ecolor=sns.color_palette(  "Paired")[5],
                                                                 elinewidth=None,
                                             label=f'Doserate={doserate2:.2f}Gy/s')



    ax.errorbar(np.arange(0,len(dose3),1)*s,                              dose3*norm3,
                                                                   yerr=err3*norm3,
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[6],
                                         ecolor=sns.color_palette(  "Paired")[6],
                                                                 elinewidth=None,
                                             label=f' Doserate={doserate3:.2f}Gy/s')


    ax.errorbar(np.arange(0,len(dose4),1)*s,                              dose4*norm4,
                                                                   yerr=err4*norm4,
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[7],
                                         ecolor=sns.color_palette(  "Paired")[7],
                                                                 elinewidth=None,
                                             label=f' Doserate= {doserate4:.2f}Gy/s')


    ax.errorbar(np.arange(0,len(dose5),1)*s,                              dose5*norm5,
                                                                   yerr=err5*norm5,
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[8],
                                         ecolor=sns.color_palette(  "Paired")[8],
                                                                 elinewidth=None,
                                             label=f' Doserate= {doserate5:.2f}Gy/s')







    ax.errorbar( rcfdepth,                      rcfdose*normrcf ,
                                                                  yerr=rcferr*normrcf ,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='orange',
                                                                        markersize=14,
                                                                   ecolor='orange',
                                                                elinewidth=None,
                                                                label=' RCF 21CW measured dose')

    ax.fill_between(rcfdepth,
                                                                rcfdose*normrcf -rcferr*normrcf ,
                                                                rcfdose*normrcf +rcferr*normrcf ,
                                                        color='orange', alpha=0.2)





    sub_axes = plt.axes([.6, .6, .25, .25])

    sub_axes.errorbar( np.arange(0,len(dose0),1)[10:40]*s,
                                                            (dose0*norm0)[10:40],
                                                        yerr=(err0*norm0)[10:40],
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[3],
                                         ecolor=sns.color_palette(  "Paired")[3],
                                                                 elinewidth=None)






    sub_axes.errorbar( np.arange(0,len(dose1),1)[10:40]*s,
                                                            (dose1*norm1)[10:40],
                                                        yerr=(err1*norm1)[10:40],
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[4],
                                         ecolor=sns.color_palette(  "Paired")[4],
                                                                 elinewidth=None)

    sub_axes.errorbar( np.arange(0,len(dose2),1)[10:40]*s,
                                                            (dose2*norm2)[10:40],
                                                        yerr=(err2*norm2)[10:40],
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[5],
                                         ecolor=sns.color_palette(  "Paired")[5],
                                                                 elinewidth=None)

    sub_axes.errorbar( np.arange(0,len(dose3),1)[10:40]*s,
                                                            (dose3*norm3)[10:40],
                                                        yerr=(err3*norm3)[10:40],
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[6],
                                         ecolor=sns.color_palette(  "Paired")[6],
                                                                 elinewidth=None)

    sub_axes.errorbar( np.arange(0,len(dose4),1)[10:40]*s,
                                                            (dose4*norm4)[10:40],
                                                        yerr=(err4*norm4)[10:40],
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[7],
                                         ecolor=sns.color_palette(  "Paired")[7],
                                                                 elinewidth=None)




    sub_axes.errorbar( np.arange(0,len(dose5),1)[10:40]*s,
                                                            (dose5*norm5)[10:40],
                                                        yerr=(err5*norm5)[10:40],
                                                                       xerr=None,
                                                                         fmt='.',
                                          color=sns.color_palette(  "Paired")[8],
                                         ecolor=sns.color_palette(  "Paired")[8],
                                                                 elinewidth=None)




    sub_axes.grid(b=True,color='k',linestyle='-',alpha=0.2)
    mark_inset(ax, sub_axes, loc1=2, loc2=3, fc="none", ec="0.5")

    #ax.set_ylim([0.3,1.2])
    ax.set_xlabel('mm',fontsize=20)
    ax.set_ylabel('Relative Dose',fontsize=20)
    ax.grid(b=True,color='k',linestyle='dotted',alpha=0.2)
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.legend( title='doserate=doseMU*0.7/irrtime',fontsize=20,loc=3,markerscale=2,title_fontsize=24)
    plt.show()

    return

plt.figure(1)


k=dose21CW/rcfdose.max()
print(len(depth))
print(len(rcfdose))
plt.errorbar(depthMC,dose21CW,
                                                        yerr=dose121CR*(2/100),
                                                                     xerr=None,
                                                                     fmt='.',
                                                                     color='green',
                                                                     markersize=14,
                                                                label='$D_{MC} $')



plt.errorbar( rcfdepth,                      rcfdose ,
                                                                  yerr=rcferr,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='orange',
                                                                        markersize=14,
                                                                   ecolor='orange',
                                                                elinewidth=None,
                                                             label='$ D_{RCF}$')

plt.fill_between(rcfdepth,
                                                                rcfdose-rcferr,
                                                                rcfdose+rcferr,
                                                        color='orange', alpha=0.2)



#plt.errorbar( depth,                      rcfdose1 ,
#                                                                  yerr=rcferr1,
#                                                                      xerr=None,
#                                                                        fmt='.',
#                                                                        markersize=14,
#                                                                   ecolor='orange',
#                                                                elinewidth=None,
#                                                                label=' RCF 21CP measured dose')
#
#plt.fill_between(depth,
#                                                                rcfdose1-rcferr1,
#                                                                rcfdose1+rcferr1,
#                                                        color='orange', alpha=0.5)


plt.title('Depth dose profile 6PC 1PMMA',fontsize=24)
plt.xlabel('mm',fontsize=20)
plt.ylabel('Dose[Gy]',fontsize=20)
plt.legend( title='',fontsize=20,markerscale=3)
#plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
#plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)
plt.show()

###################################################################
"""
directory='pictures/2021-09-01/'
filename='RCF21CT.csv'
rcfdepth,rcfdose,rcferr,area_rcf,rcfname=read_datarcf(directory,filename)
normrcf=1/rcfdose.max()


plt.figure(2)



plt.errorbar(depthMC,dose21CT,
                                                        yerr=dose21CT*(2/100),
                                                                     xerr=None,
                                                                     fmt='.',
                                                                     color='green',
                                                                     markersize=14,
                                                                    label="calib(marcus chamber)")


plt.errorbar( rcfdepth,                      rcfdose ,
                                                                  yerr=rcferr,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='orange',
                                                                        markersize=14,
                                                                   ecolor='orange',
                                                                elinewidth=None,
                                                                label=' $D_{RCF}$')

plt.fill_between(rcfdepth,
                                                                rcfdose-rcferr,
                                                                rcfdose+rcferr,
                                                        color='orange', alpha=0.2)



plt.title('Depth dose profile 12PC ',fontsize=24)
plt.xlabel('mm',fontsize=20)
plt.ylabel('Dose[Gy]',fontsize=20)
plt.legend( title='',fontsize=20)
#plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
#plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)
plt.show()





directory='pictures/2021-09-01/'
filename='RCF21CQ.csv'

rcfdepth,rcfdose,rcferr,area_rcf,rcfname=read_datarcf(directory,filename)
filename1='RCF21CP.csv'
rcfdepth,rcfdose1,rcferr1,area_rcf1,rcfname1=read_datarcf(directory,filename1)

plt.figure(3)



plt.errorbar(depthMC,dose21CQ,
                                                        yerr=dose21CQ*(2/100),
                                                                     xerr=None,
                                                                     fmt='.',
                                                                     markersize=14,
                                                                    label="$D_{MC}$")


plt.errorbar( rcfdepth,                      rcfdose ,
                                                                  yerr=rcferr,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        markersize=14,
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                                                label=' $D_{RCF}$')

plt.fill_between(rcfdepth,
                                                                rcfdose-rcferr,
                                                                rcfdose+rcferr,
                                                        color='gray', alpha=0.5)

plt.errorbar( rcfdepth,                      rcfdose1 ,
                                                                  yerr=rcferr1,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        markersize=14,
                                                                   ecolor='orange',
                                                                elinewidth=None,
                                                                label=' RCF 21CP measured dose')

plt.fill_between(rcfdepth,
                                                                rcfdose1-rcferr1,
                                                                rcfdose1+rcferr1,
                                                        color='orange', alpha=0.5)
plt.errorbar(depthMC,dose21CP,
                                                        yerr=dose21CP*(2/100),
                                                                     xerr=None,
                                                                     fmt='.',
                                                                     markersize=14,
                                                                    label="calib(marcus chamber)")

plt.title('Depth dose profile 10PC ',fontsize=24)
plt.xlabel('mm',fontsize=20)
plt.ylabel('Dose[Gy]',fontsize=20)
plt.legend( title='',fontsize=20)
plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)
plt.show()



"""


directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-09-02/'
filename='RCF21CX.csv'
rcfdepth,rcfdose,rcferr,area_rcf,rcfname=read_datarcf(directory,filename)
normrcf=1/rcfdose.max()
plt.figure(4)


plt.errorbar(depthMC,dose21CX_second,
                                                        yerr=dose21CX_second*(2/100),
                                                                     xerr=None,
                                                                     fmt='.',
                                                                     color='green',
                                                                     markersize=14,
                                                                    label="$D_{MC} calib2$")

plt.errorbar(depthMC,dose21CX,
                                                        yerr=dose21CX*(2/100),
                                                                     xerr=None,
                                                                     fmt='.',
                                                                     markersize=14,
                                                                    label="$D_{MC} calib1$")


plt.errorbar(rcfdepth,                                                   rcfdose,
                                                                     yerr=rcferr,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='darkorange',
                                                                        markersize=14,
                                                                   ecolor='orange',
                                                                elinewidth=None,
                                                                label='$D_{RCF} calib$')

plt.fill_between(rcfdepth,
                                                                rcfdose-rcferr,
                                                                rcfdose+rcferr,
                                                        color='orange', alpha=0.2)


plt.ylim([1,4])
plt.title('Depth dose profile 6PC ',fontsize=24)
plt.xlabel('mm',fontsize=20)
plt.ylabel('Dose[Gy]',fontsize=20)
plt.legend( title='',fontsize=22,markerscale=3)
#plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
#plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)
plt.show()



directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-09-02/'
filename='RCF21CY.csv'

rcfdepth,rcfdose,rcferr,area_rcf,rcfname=read_datarcf(directory,filename)
normrcf=1/rcfdose.max()
plt.figure(5)



plt.errorbar(depthMC,dose21CY,
                                                        yerr=dose21CY*(2/100),
                                                                     xerr=None,
                                                                     fmt='.',
                                                                     color='green',
                                                                     markersize=14,
                                                                    label="$D_{MC}$")


plt.errorbar(rcfdepth,                      rcfdose ,
                                                                  yerr=rcferr,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='darkorange',
                                                                        markersize=14,
                                                                   ecolor='orange',
                                                                elinewidth=None,
                                                                label=' $D_{RCF}$')

plt.fill_between(rcfdepth,
                                                                rcfdose-rcferr,
                                                                rcfdose+rcferr,
                                                        color='orange', alpha=0.2)



plt.title('Depth dose profile 7PC ',fontsize=24)
plt.xlabel('mm',fontsize=20)
plt.ylabel('Dose[Gy]',fontsize=20)
plt.legend( title='',fontsize=22,markerscale=3)
#plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)
plt.show()



directory='/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2021-09-02/'
filename='RCF21CW.csv'
rcfdepth,rcfdose,rcferr,area_rcf,rcfname=read_datarcf(directory,filename)
normrcf=1/rcfdose.max()
plt.figure(6)



plt.errorbar(depthMC,dose21CW,
                                                        yerr=dose21CW*(2/100),
                                                                     xerr=None,
                                                                     fmt='.',
                                                                     color='green',
                                                                     markersize=14,
                                                                    label="calib(marcus chamber)")


plt.errorbar(rcfdepth,                      rcfdose ,
                                                                  yerr=rcferr,
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='darkorange',
                                                                        markersize=14,
                                                                   ecolor='orange',
                                                                elinewidth=None,
                                                                label=' RCF 21CW measured dose')

plt.fill_between(rcfdepth,
                                                                rcfdose-rcferr,
                                                                rcfdose+rcferr,
                                                        color='orange', alpha=0.2)



plt.title('Depth dose profile 6PC+1PMMA ',fontsize=24)
plt.xlabel('mm',fontsize=20)
plt.ylabel('Dose[Gy]',fontsize=20)
plt.legend( title='',fontsize=22,markerscale=3)
#plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
#plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)


plt.show()
