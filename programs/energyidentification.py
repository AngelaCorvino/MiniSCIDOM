#Extraxction opf main energy components of proton beam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


treshold=np.array([0.6 , 0.7 ,0.8, 0.9,1])
directory='pictures/munichdata/2021-11-03/notnormalized/'

data= np.load(directory+'notnormalizedmean_array77.npy')
dose=data[0:len(data)-3]
dose=dose[::-1]
r=np.arange(0,len(dose),1)*0.0634
r0=np.zeros(5)
mask=[]

for i in range(0,5,1):
    mask=(dose/dose.max()>=treshold[i])
    print(mask)
    r0[i]=r[mask][-1]

data0= np.load(directory+'notnormalizederr77.npy')
err=data0[0:len(data0)]




filename1=directory+'notnormalizedmean_array100.npy'
data1= np.load(filename1)
dose1=data1[0:len(data1)-3]
dose1=dose1[::-1]

r1=np.zeros(5)
mask1=[]
for i in range(0,5,1):
    mask1=(dose1/dose1.max()>=treshold[i] )
    r1[i]=r[mask1][-1]


#tof1=data1[len(data1)-3:len(data1)]
unidose1=0
data2= np.load(directory+'notnormalizederr100.npy')
err1=data2[0:len(data2)]
top_projection_dose1= np.load(directory+'1Dtopprojection/'+'top1Dmean_array100.npy')


data4= np.load(directory+'notnormalizedmean_array108.npy')
dose2=data4[0:len(data4)-3]
dose2=dose2[::-1]
r2=np.zeros(5)
mask2=[]
for i in range(0,5,1):
    mask2=(dose2/dose2.max()>=treshold[i] )
    r2[i]=r[mask2][-1]
#tof2=data4[len(data4)-3:len(data4)]
unidose2=0
data5= np.load(directory+'notnormalizederr108.npy')
err2=data5[0:len(data5)]
top_projection_dose2= np.load(directory+'1Dtopprojection/'+'top1Dmean_array108.npy')



data6= np.load(directory+'notnormalizedmean_array134.npy')
dose3=data6[0:len(data6)-3]
dose3=dose3[::-1]
r3=np.zeros(5)
mask3=[]
for i in range(0,5,1):
    mask3=(dose3/dose3.max()>=treshold[i] )
    r3[i]=r[mask3][-1]
#tof2=data4[len(data4)-3:len(data4)]
unidose3=0
data7= np.load(directory+'notnormalizederr134.npy')
err3=data7[0:len(data7)]
top_projection_dose3= np.load(directory+'1Dtopprojection/'+'top1Dmean_array134.npy')


data8= np.load(directory+'notnormalizedmean_array135.npy')
dose4=data8[0:len(data8)-3]
dose4=dose4[::-1]
r4=np.zeros(5)
mask4=[]
for i in range(0,5,1):
    mask4=(dose4/dose4.max()>=treshold[i] )
    r4[i]=r[mask4][-1]
#tof2=data4[len(data4)-3:len(data4)]
unidose4=0
data9= np.load(directory+'notnormalizederr135.npy')
err4=data9[0:len(data9)]
top_projection_dose4= np.load(directory+'1Dtopprojection/'+'top1Dmean_array135.npy')


#PSTAR DATA
dscintillator= 1.023 #[g/cm^3] scintillator density
pstar_data=np.loadtxt('PSTARvinyltoluene-based.txt',skiprows=4,comments='#')
pstar_data = np.array(pstar_data)
Range=pstar_data[:,1] #[g/cmquadro]
Range=Range*dscintillator*10 #[mm]
energy=pstar_data[:,0]

E_in=np.interp(r,Range,energy)

#Finding energy values
energy_new=energy[np.where((Range>0.5 ) & (Range <2))] #cm
range_new=Range[np.where((Range>0.5) & (Range <2))]
xvals = np.linspace(energy_new.min(), energy_new.max(), 50)

interpolate_range=np.interp(xvals,energy_new,range_new)
R=np.array([r0[2],r1[2],r2[2],r3[2],r4[2]])  #e obtained if e´we chose the treshold=0.8
monocromaticbeam_energy=np.zeros(5)
for i in range(0,5,1):
    monocromaticbeam_energy[i]=np.interp(R[i],Range,energy)







x_array=np.array([x_1,x_2])
y_array=np.array([y_1,y_2])

Distal edge: y1>0.8 and y2 =<0.8

range = np.interp(      0.8,
                            y_array,
                            x_array )

plt.figure(1)
plt.axhline(y=0.8,linestyle='dotted',color='black')
plt.vlines( R[1:5], 0, 0.8,
                                colors=['red','orange','green','magenta'],
                                                             linestyles='dotted',
 label='Estimated range with 0.8 threshold {}mm '.format(R[1:5]))
'''
plt.errorbar(  np.arange(0,len(dose2),1)*0.0634,
                                                               dose/dose.max() ,
                                                             yerr=err/dose.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                    color='blue',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                label='{} reconstructed shot 77 '.format(unidose))
plt.axhline(y=0.9,linestyle='dotted',color='black')

plt.axvline(x=r2[3], linestyle='dotted',color='orange')
plt.axvline(x=r4[3], linestyle='dotted',color='magenta')
plt.axvline(x=r3[3], linestyle='dotted',color='green')
plt.axvline(x=r0[3], linestyle='dotted',color='blue')
plt.axvline(x=r1[3], linestyle='dotted',color='red')
'''
plt.errorbar(  np.arange(0,len(dose1),1)*0.0634,                  dose1/dose1.max(),
                                                                yerr=err1/dose1.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                        color='red',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                label='shot 100:{:.2f}  MeV'.format(monocromaticbeam_energy[1]))
plt.errorbar(  np.arange(0,len(dose2),1)*0.0634,
                                                             dose2/dose2.max() ,
                                                         yerr=err2/dose2.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                    ecolor='gray',
                                                                   color='orange',
                                                                elinewidth=None,
                                label='shot 108:{:.2f} MeV'.format(monocromaticbeam_energy[2]))

plt.errorbar(  np.arange(0,len(dose2),1)*0.0634,
                                                               dose3/dose3.max(),
                                                            yerr=err3/dose3.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                    color='green',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                                label='shot 134:{:.2f} MeV '.format(monocromaticbeam_energy[3]))
plt.errorbar(  np.arange(0,len(dose2),1)*0.0634,
                                                               dose4/dose4.max(),
                                                          yerr=err4/dose4.max(),
                                                                      xerr=None,
                                                                        fmt='.',
                                                                color='magenta',
                                                                   ecolor='gray',
                                                                elinewidth=None,
                            label='shot 135: {:.2f} MeV '.format(monocromaticbeam_energy[4]))
plt.grid(True)


plt.legend( title='Unidose',fontsize='large')
plt.title('Range Comparison ',
                                                                  fontdict=None,
                                                                  loc='center',
                                                                       pad=None)
plt.xlim([0,4])
plt.xlabel('Depth[mm]')
plt.ylabel('Normalized amplitude')
plt.legend()




plt.figure(2)
plt.subplot(1,2,1)
plt.plot(treshold,r0,'.',color='blue',label=' shot 77(12 MeV) ')
plt.plot(treshold,r1,'.',color='red',label=' shot 100 (12MeV)')
plt.plot(treshold,r2,'.',color='orange',label='shot 108 (14MeV)')
plt.plot(treshold,r3,'.',color='green',label='shot 134 (13MeV)')
plt.plot(treshold,r4,'.',color='magenta',label='shot 135 (13MeV)')
plt.title('Proton range Vs Relative Energy treshold')
plt.ylabel('Proton range[mm]')
plt.xlabel('Relative Energy  treshold')
plt.minorticks_on()
plt.grid()
plt.legend()

plt.subplot(1,2,2)

plt.plot(E_in,dose/dose.max() ,label=' shot 77(12 MeV) ')
plt.plot(  E_in,dose1/dose1.max(),
                                                                    color='red',
                                                      label=' shot 100 (12MeV)')
plt.plot(  E_in,dose2/dose2.max(),
                                                                 color='orange',
                                                       label=' shot 108 (14MeV)')
plt.plot(  E_in,dose3/dose3.max(),
                                                                  color='green',
                                                      label=' shot 134 (13MeV)')
plt.plot(  E_in,dose4/dose4.max(),
                                                                 color='magenta',
                                                       label=' shot 135 (13MeV)')
plt.vlines( monocromaticbeam_energy, 0, 0.85,
                                colors=['blue','red','orange','green','magenta'],
                                                             linestyles='dotted',
 label='{} MeV Estimated energy from PSTAR data '.format(monocromaticbeam_energy))
plt.title('Dose vs Energy')
plt.xlabel('Energy[MeV]')
plt.ylabel('Intensity')
plt.xlim([0,20])
plt.legend()
plt.minorticks_on()
plt.grid()

plt.show()
