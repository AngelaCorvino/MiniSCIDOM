import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Birkmodel import letcorrection
#from readcsv import read_tof
from readnpy import read_dose
from readnpy import read_doserr



##########################
def read_data_mini(path):
    """read simulation in scintillator data from execel document"""
    data = pd.read_csv(path, header=None, skiprows=1, delimiter=",")
    depth_sim = data[0]  # depth in scintillator
    realdose = data[1]
    quenchedose = data[2]
    ys = data[3]  # LET ToF in kev/um

    return depth_sim, realdose, quenchedose, ys

    # the following data are simulated directly in the scintillator strarting from tof measurements


def read_tof(path):
    """read dose simulated in scintillator  from execel document"""
    data = pd.read_csv(path, header=None, skiprows=1, delimiter=",")
    depth_sci = data[0]  # depth in scintillator
    dose_sci = data[1]
    dose_sci_upper = data[6]
    dose_sci_lower = data[7]

    # ys_ana_mini=(data[8])
    # ys_ana_upper_mini=(data[9])# #dose weighted LET ToF in kev/um analitical method
    # ys_ana_lower_mini=(data[10])

    ys_ana_mini = data[11]  # fluence weighted LET ToF in kev/um analitical method
    ys_ana_upper_mini = data[12]
    ys_ana_lower_mini = data[13]

    return (
        depth_sci,
        dose_sci,
        dose_sci_upper,
        dose_sci_lower,
        ys_ana_mini,
        ys_ana_upper_mini,
        ys_ana_lower_mini,
    )


#################################################
"""TOF SIMULATED DATA """
path1 = "/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/ToF_FUKA_sims_dose_quench_miniSCI_25_09_20_shot_93.csv"
path2 = "/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/ToF_FUKA_sims_dose_quench_miniSCI_25_09_20_shot_92.csv"
path3 = "/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/ToF_FUKA_sims_dose_quench_miniSCI_25_09_20_shot_91.csv"

depth_sim, realdose1, quenchedose1, ys1 = read_data_mini(path1)
depth_sim, realdose2, quenchedose2, ys2 = read_data_mini(path2)
depth_sim, realdose3, quenchedose3, ys3 = read_data_mini(path3)


"TOF PREDICTED  DATA IN SCINTILLATOR and  RCF MEASUREMENTS analitical "
path1 = "/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/TOFinscintillator_93_ana.csv"
path2 = "/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/TOFinscintillator_92_ana.csv"
path3 = "/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/TOFinscintillator_91_ana.csv"
(
    depth_sci1,
    dose_sci1,
    dose_sci_upper1,
    dose_sci_lower1,
    ys_ana_mini1,
    ys_ana_upper_mini1,
    ys_ana_lower_mini1,
) = read_tof(path1)
(
    depth_sci2,
    dose_sci2,
    dose_sci_upper2,
    dose_sci_lower2,
    ys_ana_mini2,
    ys_ana_upper_mini2,
    ys_ana_lower_mini2,
) = read_tof(path2)
(
    depth_sci3,
    dose_sci3,
    dose_sci_upper3,
    dose_sci_lower3,
    ys_ana_mini3,
    ys_ana_upper_mini3,
    ys_ana_lower_mini3,
) = read_tof(path3)

# low dose September
s = 0.0634
directory = "/Users/angelacorvino/Documents/GitHub/miniscidom/pictures/2020-09-25/lowdose/notnormalized/"
dose1, depth1, tof1, number1 = read_dose(directory, "notnormalizedmean_array93.npy", s)
err1 = read_doserr(directory, "notnormalizederr93.npy")
area1 = np.trapz(dose1[3 : len(dose1)], depth1[3 : len(depth1)])

dose2, depth2, tof2, number2 = read_dose(directory, "notnormalizedmean_array92.npy", s)
err2 = read_doserr(directory, "notnormalizederr92.npy")
area2 = np.trapz(dose2[3 : len(dose2)], depth2[3 : len(depth2)])


dose3, depth3, tof3, number3 = read_dose(directory, "notnormalizedmean_array91.npy", s)
err3 = read_doserr(directory, "notnormalizederr91.npy")
area3 = np.trapz(dose3[3 : len(dose3)], depth3[3 : len(depth3)])
###############################################################################
"""NORMALIZATION"""
norm1 = 1 / dose1.max()
norm2 = 1 / dose2.max()
norm3 = 1 / dose3.max()


normquen1 = 1 / quenchedose1.max()
normquen2 = 1 / quenchedose2.max()
normquen3 = 1 / quenchedose3.max()
normreal1 = 1 / realdose1.max()

###############################################################################
"""LET CORRECTION"""
D_a_mini1, D_a_up_mini1, D_a_low_mini1, area_corrected1, S_a_mini1 = letcorrection(
    depth_sci1, dose1, ys_ana_mini1, ys_ana_lower_mini1, ys_ana_upper_mini1, s
)
D_a_mini2, D_a_up_mini2, D_a_low_mini2, area_corrected2, S_a_mini2 = letcorrection(
    depth_sci2, dose2, ys_ana_mini2, ys_ana_lower_mini2, ys_ana_upper_mini2, s
)
D_a_mini3, D_a_up_mini3, D_a_low_mini3, area_corrected3, S_a_mini3 = letcorrection(
    depth_sci3, dose3, ys_ana_mini3, ys_ana_lower_mini3, ys_ana_upper_mini3, s
)


(
    D_a_mini_quenched,
    D_a_up_mini_quenched,
    D_a_low_mini_quenched,
    area_corrected_quenched,
    S_a_mini_quenched,
) = letcorrection(depth_sim, quenchedose3, ys3, ys3, ys3, s)
##################################################################################

fig, ax = plt.subplots()
ax2 = ax.twinx()

ax2.plot(
    depth_sci1,
    ys_ana_mini1,
    "-",
    drawstyle="steps-mid",
    linewidth=2,
    color="orange")


ax.plot(
    np.arange(0, len(dose1), 1) * 0.0634,
    dose1 * norm1,
    color=sns.color_palette("Paired")[1],
    linewidth=2,
    label="$D_{MS}$" ,
)
ax.plot(
    depth_sci1,
    dose_sci1 / dose_sci1.max(),
    "-",
    drawstyle="steps-mid",
    linewidth=2,
    color="green",
    label="$D_{ToFpredicted}$",
)


ax.plot(
    depth_sim,
    quenchedose1 * normquen1,
    "r-",
    drawstyle="steps-mid",
    linewidth=2,
    label="$D_{ToFsimulated,quenched}$",
)
ax.plot(
    np.arange(0, len(dose1), 1) * s,
    D_a_mini1 / D_a_mini1.max(),
    color=sns.color_palette("Paired")[0],
    linewidth=2,
    label="$D_{MS,LETcorrected}$",
)






#plt.title(
#    "Shape Comparison shot n{}".format(number1),
#    fontdict=None,
#    loc="center",
#    pad=None,
#    fontsize=24,
#)

ax.set_xlabel("Depth in PVT in mm ", fontsize=20)
# set y-axis label
ax.set_ylabel(" Relative Dose", fontsize=20)
ax2.set_ylabel(r"Fluence weighted LET in $\frac{keV}{µm}$", color="orange", fontsize=20)
ax.legend(title="", fontsize=13, loc='upper right',fancybox=True, shadow=True)
#ax2.grid(b=True, color="k", linestyle="dashed", alpha=0.2)
#ax.grid(b=True, color="k", linestyle="-", alpha=0.2)
ax.tick_params(axis="x", which="major", labelsize=20)
ax.tick_params(axis="y", which="major", labelsize=20)
ax2.tick_params(axis="y", which="major", colors="orange", labelsize=20)
ax.set_xlim(0)
ax.set_ylim(0,1.5)
ax2.set_xlim(0)
ax2.set_ylim(0,20)

plt.tight_layout()
plt.show()
