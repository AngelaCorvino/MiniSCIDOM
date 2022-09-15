#Extraxction opf main energy components of proton beam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


treshold=np.array([0.6 , 0.7 ,0.8, 0.9,1])
directory='pictures/munichdata/2021-11-03/notnormalized/'

filename1=directory+'notnormalizedmean_array100.npy'
data1= np.load(filename1)
dose1=data1[0:len(data1)-3]
dose1=dose1[::-1]

data2= np.load(directory+'notnormalizederr100.npy')
err1=data2[0:len(data2)]
err1=err1/dose1.max()
top_projection_dose1= np.load(directory+'1Dtopprojection/'+'top1Dmean_array100.npy')


dose1=dose1/dose1.max() #normalization
r=np.arange(0,len(dose1),1)*0.0634
#Finding dose values
dose_new=dose1[np.where((r>0.65) & (r <0.8))]
range_new=r[np.where((r>0.65 ) & (r <0.8))]
xval = np.linspace(range_new.min(), range_new.max(), 30)
dose_interpolation=np.interp(xval,range_new,dose_new)
mask1=(dose_interpolation>=0.8)
r1=xval[mask1][-1]





data4= np.load(directory+'notnormalizedmean_array108.npy')
dose2=data4[0:len(data4)-3]
dose2=dose2[::-1]

data5= np.load(directory+'notnormalizederr108.npy')
err2=data5[0:len(data5)]
err2=err2/dose2.max()
top_projection_dose2= np.load(directory+'1Dtopprojection/'+'top1Dmean_array108.npy')
dose2=dose2/dose2.max() #normalization

dose_new2=dose2[np.where((r>1.55 ) & (r <1.75))]
range_new2=r[np.where((r>1.55 ) & (r <1.75))]
xval2 = np.linspace(range_new2.min(), range_new2.max(), 30)
dose_interpolation2=np.interp(xval2,range_new2,dose_new2)
mask2=(dose_interpolation2>=0.8)
r2=xval2[mask2][-1]



data6= np.load(directory+'notnormalizedmean_array134.npy')
dose3=data6[0:len(data6)-3]
dose3=dose3[::-1]


data7= np.load(directory+'notnormalizederr134.npy')
err3=data7[0:len(data7)]
err3=err3/dose3.max()
top_projection_dose3= np.load(directory+'1Dtopprojection/'+'top1Dmean_array134.npy')
dose3=dose3/dose3.max() #normalization

dose_new3=dose3[np.where((r>1.3 ) & (r <1.5))]
range_new3=r[np.where((r>1.3) & (r <1.5))]
xval3 = np.linspace(range_new3.min(), range_new3.max(), 30)
dose_interpolation3=np.interp(xval3,range_new3,dose_new3)
mask3=(dose_interpolation3>=0.8)
r3=xval3[mask3][-1]





data8= np.load(directory+'notnormalizedmean_array135.npy')
dose4=data8[0:len(data8)-3]
dose4=dose4[::-1]


data9= np.load(directory+'notnormalizederr135.npy')
err4=data9[0:len(data9)]
err4=err4/dose4.max()
top_projection_dose4= np.load(directory+'1Dtopprojection/'+'top1Dmean_array135.npy')

dose4=dose4/dose4.max() #normalization

dose_new4=dose4[np.where((r>1.35 ) & (r <1.55))]
range_new4=r[np.where((r>1.35) & (r <1.55))]
xval4= np.linspace(range_new4.min(), range_new4.max(), 30)
dose_interpolation4=np.interp(xval4,range_new4,dose_new4)
mask4=(dose_interpolation4>=0.8)
r4=xval4[mask4][-1]

#PSTAR DATA
dscintillator= 1.023 #[g/cm^3] scintillator density
pstar_data=np.loadtxt('PSTARvinyltoluene-based.txt',skiprows=4,comments='#')
pstar_data = np.array(pstar_data)
Range=pstar_data[:,1] #[g/cmquadro]
Range=Range*dscintillator*10 #[mm]
energy=pstar_data[:,0]

E_in=np.interp(r,Range,energy)
R=np.array([r1,r2,r3,r4])  #e obtained if e´we chose the treshold=0.8
monocromaticbeam_energy=np.zeros(5)
for i in range(0,4,1):
    monocromaticbeam_energy[i]=np.interp(R[i],Range,energy)







plt.figure(1)
'''
plt.axhline(y=0.8,color='black',linestyle='dashed',linewidth=3)
plt.plot(xval,dose_interpolation,'d',color='darkred',markersize=4)
plt.plot(xval4,dose_interpolation4,'d',color='darkmagenta',markersize=4)
plt.plot(xval3,dose_interpolation3,'d',color='lightgreen',markersize=4)
plt.plot(xval2,dose_interpolation2,'d',color='goldenrod',markersize=4)
'''
plt.axhline(y=0.8,color='black',linestyle='dashed',linewidth=3)


plt.vlines( R, 0, 0.8,
                               colors=['red','orange','green','magenta'],
                                                             linestyles='dashed',
                                                        linewidth=3    )

plt.plot(  np.arange(0,len(dose1),1)*0.0634,                  dose1,


                                                                        '',
                                                                        markersize=11,
                                                                        linewidth=3,
                                                                        color='red',

                                label='Shot 100:  {:.2f}  MeV, Range ={:.2f}mm'.format(monocromaticbeam_energy[0],R[0]))

plt.plot(  np.arange(0,len(dose2),1)*0.0634,
                                                             dose2/dose2.max() ,


                                                                        '',
                                                                        markersize=11,
                                                                        linewidth=3,
                                                                   color='orange',
                                label='Shot 108:  {:.2f} MeV, Range ={:.2f}mm'.format(monocromaticbeam_energy[1],R[1]))

plt.plot(  np.arange(0,len(dose2),1)*0.0634,
                                                               dose3/dose3.max(),

                                                                        '',
                                                                        markersize=11,
                                                                        linewidth=3,
                                                                    color='green',

                                label='Shot 134:  {:.2f} MeV,  Range ={:.2f}mm '.format(monocromaticbeam_energy[2],R[2]))
plt.plot(  np.arange(0,len(dose2),1)*0.0634,
                                                               dose4/dose4.max(),


                                                                        '',
                                                                        markersize=11,
                                                                        linewidth=3,
                                                                color='magenta',

                            label='Shot 135:  {:.2f} MeV,  Range ={:.2f}mm '.format(monocromaticbeam_energy[3],R[3]))



plt.legend(fontsize='18')
plt.title('Range Comparison ',
                                                                  fontdict=None,
                                                                  loc='center',
                                                                       pad=None,
                                                                    fontsize=26)
plt.xlim([0,4])
plt.ylim([0,1.1])
plt.xlabel('Depth[mm]',fontsize=22)
plt.ylabel('Normalized amplitude',fontsize=22)
plt.xticks(size = 22)
plt.yticks(size = 22)





plt.show()
