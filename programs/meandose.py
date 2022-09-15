import numpy as np
import matplotlib.pyplot as plt
import os
import glob



#2020-09-22 high dose
tofhighpeak=np.array([281,245,177,228,238,186,141,147,192,234,212])
toflowpeak=np.array([85,52,48,82,67,47,63,37,45,49,39])
peakratio=np.array([3.30,4.68,3.73,2.78,3.58,3.94,2.23,3.98,4.27,4.80,5.44])

'''
#202-09-24 high dose
tofhighpeak=np.array([428,435,434,462,392,389])
toflowpeak=np.array([124,146,114,120,97,115])
peakratio=np.array([3.44,2.98,3.82,3.85,4.06,3.38])
'''
'''
#202-09-25 low dose
unidose=np.array([3.30,4.00,4.40]) #nC

'''




#shotnumber = []

filelist = glob.glob('pictures/2020-09-22/centered-highdose/notnormalized/*')
#for filename in filelist:
#    _, n=(filename.split('array',1))
#    i = int(n[:-4])
#    shotnumber.append(i)
#    print(shotnumber)


for filename in filelist:
    if '.npy' in filename:
        data= np.load(filename)
        dose=data[1:len(data)-3]
        tof=data[len(data)-3:len(data)]
        print(dose)
        print(tof)

        plt.plot(                            np.arange(0,len(dose),1)*0.0634,
                                                                  dose[::-1],
                                                                         '.',
                                                                markersize=7,
                                                      label='{}'.format(tof))


        plt.grid(True)

        #legend1=plt.legend(unidose[shotnumber], title='Unidose [nC]',loc='upper right',fontsize='large')
        #legend1=plt.legend(tofhighpeak[shotnumber], title='TOF High energy peak [mV]',loc='upper right',fontsize='large')
        #legend2=plt.legend(toflowpeak[shotnumber],title='Low energy peak[mV]',loc=4,fontsize='large')
        #legend3=plt.legend(peakratio[shotnumber],title='Ratio ',loc=3,fontsize='large')
        #plt.gca().add_artist(legend1)
        #plt.gca().add_artist(legend2)
        #plt.gca().add_artist(legend3)
    else:
        print('not found')

plt.legend( title='TOF High and Low energy peak [mV]',fontsize='large')
plt.title('Depth-dose distribution high dose setting 09-22', fontdict=None, loc='center', pad=None)
plt.xlabel('Depth[mm]')
plt.ylabel('Intensity')
plt.show()
