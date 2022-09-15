import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.optimize import curve_fit

import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-log",
    "--log",
    default="warning",
    type=str,
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

area=200
time1=np.array([300,400,500,600,700,800,900])
time=np.array([10,25,50,75,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900])
mean1=np.array([24374,33977,44056,53993,63946,65446,65472])


mean=np.array([929,2209,4340,6478,8541,12725,16860,20830,24814,28909,33471,38349,43276,48197,53097,
               57842,62243,64633,65300,65444,65472])
std=np.array([180,247,400,499,641,947,1214,1493,1767,2064,2327,2577,2755,3068,3302,3375,3172,1923,828,235,0])


std1=np.array([1658,2150,2627,3059,2653,275,0])
err1=std1/np.sqrt(150) *(2) #2sigma
err=std/np.sqrt(area) *(2)



#fit function
def test_func(x, a, b):
    return a*x+b

params, pcov = curve_fit(test_func, time[:11],       mean[:11],
                                              sigma= err[:11],
                                                   p0=[0, 0],
                bounds=([-np.inf,-np.inf] ,[ np.inf, np.inf]))
a,b=params
da,db=np.sqrt(np.diag(pcov))

logging.info("Fit estimation a %f", a)
logging.info(" da %f", da)
logging.info("Fit estimation b %f", b)
logging.info(" db %f", db)

plt.figure(figsize=(6, 4))
#Data

#plt.errorbar(time1,mean1,yerr=err1,fmt='.' ,markersize=10,label='Data')
plt.errorbar(time,mean,yerr=err,fmt='.' ,markersize=10,label='Data')


plt.fill_between(time,
                                                mean-err,
                                              mean + err,
                                 color='gray', alpha=0.5)
#Fit
plt.plot(np.arange(-1,900), test_func(np.arange(-1,900),
                                                params[0], params[1]),
                                               label=(f'y=ax+b a={a:.1f} \u00B1 {da:.1f},b={b:.0f} \u00B1 {db:.0f}  '))
plt.fill_between(time,
                                                  mean-err,
                                                mean + err,
                                    color='gray', alpha=0.5)

plt.plot(np.arange(-1,900),np.ones(901)*65472,'--',label='camera saturation')
plt.legend(loc=2,fontsize='large')
plt.title(' Balser Camera Exposure ',fontsize=18)
plt.xlabel('Time[ms]',fontsize=16)
plt.ylabel('Mean Intensity',fontsize=16)
plt.xlim([-1,950])
plt.ylim([-100,70000])
plt.grid(b=True, which='major', color='k', linestyle='-',alpha=0.2)
plt.grid(b=True, which='minor', color='k', linestyle='-', alpha=0.2)
plt.minorticks_on()

#plt.plot(np.arange(0,900),)

plt.show()
