import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
path='20EB-Lateral-2\ResultsStack20EB-5mm-x-1.csv'

def read_data(path):
	#data = pd.read_excel (path)
	data=pd.read_csv(path,header=None,skiprows=1,delimiter=',')
				#print(data)
	x=(data[0]) # pixel
	y=(data[1]) # intensity
	return x,y
x,y=read_data(path)

xcm_rcf=np.average(x,weights=y)
print(xcm_rcf)
plt.plot(x,y,'.')
plt.xlim([0,60])
plt.ylim([3,4])
plt.show()
'''
path='pictures/2020-09-25/20KD-20KE-Lateral (ca.7mm)/20KD-Lateral (ca.7mm)/rcflayer_14.csv'

def read_data(path):
    #data = pd.read_excel (path)
    data=pd.read_csv(path,header=None,skiprows=1,delimiter=',')
    x=(data[0]) # pixel
    y=(data[1]) # intensity
    return x,y
x,y=read_data(path)
plt.plot(x,y,'.')
plt.show()
