##################################################
## Author: Marvin Reimold                       ##
## Copyright: -, RCF calib plotter              ##
## Credits: [Marvin Reimold]                    ##
## License: -                                   ##
## Version: 0.1.0                               ##
## Maintainer: Marvin Reimold                   ##
## Email: m.reimold@hzdr.de                     ##
## Status: Development                          ##
##################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

############################### read in data ###########################################

#Path csv file
path="csv_rcf_calib.txt"
rcf_calib_data=pd.read_csv(path,delimiter=',',skiprows=1,header=None)
netOD = np.array(rcf_calib_data[0])
netOD_err = np.array(rcf_calib_data[1])
dose = np.array(rcf_calib_data[2])
dose_err = np.array(rcf_calib_data[3])
print((dose_err /dose)*100)


########################################################## data Oncoray
#
# path="csv_rcf_caliboncoray.txt"
# rcf_calib_data=pd.read_csv(path,delimiter=',',skiprows=1,header=None)
# netODoncoray = np.array(rcf_calib_data[0])
# netODoncoray_err = (netOD/100)*2
# doseoncoray = np.array(rcf_calib_data[1])
# doseoncoray_err = (dose_err/100)*2
# #print(dose)

########################################################################################
############################### Fit parameters #########################################

# p1 = -1
# p1_err= 0
# p2 = -0.03684
# p2_err = 1.49*10**(-3)
# p3 = -0.20782
# p3_err = 7.03*10**(-3)
# netOD_fit = np.arange(0.038,0.96,0.01)
# ############################### Fit parameters #########################################
#                                                                                        #
# p1oncoray = -1
# p1oncoray_err = 0
# p2oncoray = -0.0372
# p2oncoray_err = 1.15*10**(-3)
# p3oncoray = -0.20562
# p3oncoray_err = 6.48*10**(-3)
#
# netODoncoray_fit = np.arange(0.037,0.96,0.01)
# def  par(p1,p2,p3,netOD_fit):
#                                                                                      #
#     dose_fit = (p1+10**(-netOD_fit))/(p2+p3*10**(-netOD_fit))
#                                                                                        #
# ########################################################################################
#
# ############################### uncertainty parameters #################################
#                                                                                        #
# #err_dose_fit_1 = ((p1+p1_err+10**(-netOD_fit))/(p2+p2_err+(p3+p3_err)*10**(-netOD_fit))-\
# #               (p1-p1_err+10**(-netOD_fit))/(p2-p2_err+(p3-p3_err)*10**(-netOD_fit)))/2
#                                                                                        #
# #err_dose_fit_2 = ((p1+p1_err+10**(-netOD))/(p2+p2_err+(p3+p3_err)*10**(-netOD))-\
# #               (p1-p1_err+10**(-netOD))/(p2-p2_err+(p3-p3_err)*10**(-netOD)))/2
#                                                                                        #
#     err_dose_fit_1 = np.abs((p1+10**(-netOD_fit))*p2_err/(p2+p3*10**(-netOD_fit))**2)+\
#                  np.abs((p1+10**(-netOD_fit))*p3_err*10**(-netOD_fit)/\
#                  (p2+p3*10**(-netOD_fit))**2)
#                                                                                        #
#     err_dose_fit_2 = np.abs((p1+10**(-netOD))*p2_err/(p2+p3*10**(-netOD))**2)+\
#                  np.abs((p1+10**(-netOD))*p3_err*10**(-netOD)/\
#                  (p2+p3*10**(-netOD))**2)
#                                                                                        #
#     return(dose_fit,err_dose_fit_1,err_dose_fit_2)
#
# dose_fit,err_dose_fit_1,err_dose_fit_2=par(p1,p2,p3,netOD_fit)
# doseoncoray_fit,err_doseoncoray_fit_1,err_doseoncoray_fit_2=par(p1oncoray,p2oncoray,p3oncoray,netODoncoray_fit)
#
# dose_mc_rel_err = 0.04373
# x1 = 0.01916
# x1_err = 0.02054
# x2 = -0.02704
# x2_err = 0.18024
# x3 = 0.13282
# x3_err = 0.41382
# x4 = 0.28405
# x4_err = 0.27212
#                                                                                        #
# dose_netOD_err_1 = x1+netOD_fit*x2+netOD_fit**2*x3+netOD_fit**3*x4
# dose_netOD_err_2 = x1+netOD*x2+netOD**2*x3+netOD**3*x4
#
# #print(netOD)
# dose_tot_err_1 = np.sqrt(dose_netOD_err_1**2+err_dose_fit_1**2+dose_fit**2*dose_mc_rel_err**2)
# dose_tot_err_2 = np.sqrt(dose_netOD_err_2**2+err_dose_fit_2**2+dose**2*dose_mc_rel_err**2)
#
# #dose_tot_err = np.sqrt(dose**2*dose_mc_rel_err**2+dose_netOD_err**2+err_dose_fit**2)
# #dose_tot_err = np.sqrt(dose**2*dose_mc_rel_err**2+dose_netOD_err**2)+err_dose_fit
#
# plt.plot(dose_fit,dose_tot_err_1/dose_fit*100,label='$2\sigma$ uncertainty')
# plt.scatter(dose,dose_tot_err_2/dose*100,color='black',label='Measured values')
# plt.plot([4,4],[0,5.6],linestyle='--',color='black',label='Estimated value: {}%'.format(5.6)+ ' $(2\sigma)$')
# plt.plot([0,4],[5.6,5.6],linestyle='--',color='black')
# plt.xlim(0,16)
# plt.ylim(5,8)
# plt.ylabel(r'$\frac{|{\Delta}D_{RCF,calib}|}{D_{RCF}}$ in % (2$\sigma$)', size =25)
# plt.xlabel(r'$D_{RCF}$ in Gy', size =25)
# plt.tick_params('both', labelsize= 20)
# plt.legend(prop={'size': 12})
# plt.tight_layout()
# plt.show()
#                                                                                        #
# ########################################################################################
#
#
#
# ################################# Plot_data ############################################
#                                                                                        #
# plt.errorbar(         netOD,
#                        dose,
#              xerr=netOD_err,
#               yerr=dose_err,
#                     fmt='.',
#                   capsize=4,
#                  capthick=2,
#               color='black',
#               label='Measured values protons')
# plt.plot(netOD_fit,
#           dose_fit,
#           color='black',
#          label= r'$D_{RCF,p}=\frac{p1+10^{netOD}}{p2+p3*10^{netOD}}$'+\
#                 '\n$p1=-1\pm0$'+\
#                 '\n$p2=-0.0368\pm0.0015$'+\
#                 '\n$p3=-0.207\pm0.007$'  )
# plt.fill_between(               netOD_fit,
#                   dose_fit-dose_tot_err_1,
#                   dose_fit+dose_tot_err_1,
#                              color='blue',
#                               alpha = 0.3,
#                    label = '95% confidence band' )
#
# """
# plt.errorbar(         netODoncoray,
#                        doseoncoray,
#              xerr=netODoncoray_err,
#               yerr=doseoncoray_err,
#                     fmt='.',
#                   capsize=4,
#                  capthick=2,
#               color='blue',
#               label='Measured values photons')
# plt.plot(netODoncoray_fit,
#           doseoncoray_fit,
#           color='blue',
#          label= r'$D_{RCF,X}=\frac{p1+10^{netOD}}{p2+p3*10^{netOD}}$'+\
#                 '\n$p1=-1\pm0$'+\
#                 '\n$p2=-0.0372\pm0.0012$'+\
#                 '\n$p3=-0.206\pm0.007$'  )
# """
#
#
# plt.xlim(0,1)
# plt.ylim(0,18)
# plt.ylabel(r'$D_{RCF}$ in Gy', size =25)
# plt.xlabel('netOD', size =25)
# plt.tick_params('both', labelsize= 20)
# plt.legend(prop={'size': 12})
# plt.tight_layout()
# plt.show()
