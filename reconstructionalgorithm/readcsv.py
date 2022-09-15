import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_datarcf(directory, filename):
    # data = pd.read_excel (path)
    """read measured rcf  data from execel document"""
    path = directory + filename
    data = pd.read_csv(path, header=None, skiprows=1, delimiter=",")
    depth = (data[0]) * 1.3  # pixel
    rcfdose = data[1]
    rcferr = rcfdose * (546 / 10000)
    area_rcf = np.trapz(rcfdose[0 : len(rcfdose)], depth[0 : len(depth)])
    (rcfname, extension) = filename.split(".csv")
    return depth, rcfdose, rcferr, area_rcf, rcfname


def read_datarcf_simulation(path):
    # data = pd.read_excel (path)
    """read measured rcf  data from execel document"""
    data = pd.read_csv(path, header=None, skiprows=1, delimiter=",")
    depth = (data[0]) * 1.28  # waterequivalentdepth
    rcfdose = data[1]
    rcferr = data[2]  # calibration eroor
    dosesim = data[3]  # simulated dose in rcf
    dosesim_upper = data[8]
    dosesim_lower = data[9]
    return depth, rcfdose, rcferr, dosesim, dosesim_lower, dosesim_upper


def read_data_let(path):
    """read simulation data from execel document"""
    data = pd.read_csv(path, header=None, skiprows=1, delimiter=",")
    xs = (data[0]) * 1.28  # dfepht in rcf converted in depth in water
    ys_iterative = data[1]
    ys_iterative_upper = data[2]  # intensity
    ys_iterative_lower = data[3]
    ys_ana = data[4]
    ys_ana_upper = data[5]
    ys_ana_lower = data[6]
    # sim_redose=(data[7])
    return (
        xs,
        ys_iterative,
        ys_iterative_upper,
        ys_iterative_lower,
        ys_ana,
        ys_ana_upper,
        ys_ana_lower,
    )


def read_data_let_scintillator(path):
    """read simulation in scintillator data from execel document"""
    data = pd.read_csv(path, header=None, skiprows=1, delimiter=",")
    rcfdose = data[1]
    ys_iterative_sci = data[4]  # tof simulation in scintillator
    lightratio = data[5]

    return rcfdose, ys_iterative_sci, lightratio


# the following data are simulated strarting from a shot measured by the miniscidom
def read_data_let_mini(path):
    """read simulation in scintillator data from execel document"""
    data = pd.read_csv(path, header=None, skiprows=1, delimiter=",")
    depth_sci = data[0]  # depth in scintillator
    ys_iterative_mini = data[1]
    ys_iterative_upper_mini = data[
        2
    ]  # #Dose weighted LET ToF in kev/um analitical method
    ys_iterative_lower_mini = data[3]
    ys_ana_mini = data[4]
    ys_ana_upper_mini = data[5]
    ys_ana_lower_mini = data[6]
    tof_curve = data[7]

    return (
        depth_sci,
        ys_iterative_mini,
        ys_iterative_upper_mini,
        ys_iterative_lower_mini,
        ys_ana_mini,
        ys_ana_upper_mini,
        ys_ana_lower_mini,
        tof_curve,
    )


# the following data are simulated directly in the scintillator strarting from tof measurements
def read_data_scintillatorsimulateddose(path):
    """read dose simulated in scintillator  from execel document"""
    data = pd.read_csv(path, header=None, skiprows=1, delimiter=",")
    depth_sci = data[1]  # depth in scintillator
    depth = (data[0]) * 1.28
    rcfdose = data[2]
    rcferr = data[3]
    # ys_iterative_mini=(data[1])
    # ys_iterative_upper_mini=(data[2])# intensity
    dose_sci = data[4]
    dose_sci_upper = data[9]
    dose_sci_lower = data[10]

    # ys_ana_mini=(data[11])
    # ys_ana_upper_mini=(data[12])# #dose weighted LET ToF in kev/um analitical method
    # ys_ana_lower_mini=(data[13])

    ys_ana_mini = data[14]  # Fluence weighted LET ToF in kev/um analitical method
    ys_ana_upper_mini = data[15]
    ys_ana_lower_mini = data[16]

    # tof_curve=(data[7])

    return (
        depth,
        depth_sci,
        rcfdose,
        rcferr,
        dose_sci,
        dose_sci_upper,
        dose_sci_lower,
        ys_ana_mini,
        ys_ana_upper_mini,
        ys_ana_lower_mini,
    )

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

    # tof_curve=(data[7])

    return (
        depth_sci,
        dose_sci,
        dose_sci_upper,
        dose_sci_lower,
        ys_ana_mini,
        ys_ana_upper_mini,
        ys_ana_lower_mini,
    )


def read_spectrum(path):
    """read dose simulated in scintillator  from execel document"""
    data = pd.read_csv(path, header=None, skiprows=1, delimiter=",")
    energy = data[0]  # depth in scintillator
    nproton = data[1]
    xerr_upper = data[4]
    xerr_lower = data[5]
    yerr_upper = data[6]
    yerr_lower = data[7]

    return energy, nproton, xerr_upper, xerr_lower, yerr_upper, yerr_lower


# the following data are simulated directly in the scintillator strarting from tof measurements
def read_data_scintillatorsimulateddose_it(path):
    """read dose simulated in scintillator  from execel document"""
    data = pd.read_csv(path, header=None, skiprows=1, delimiter=",")
    depth = (data[0]) * 1.28  # water equivalent depth
    depth_sci = data[1]  # depth in scintillator= water equivalent depth

    rcfdose = data[2]
    rcferr = data[3]
    # ys_iterative_mini=(data[1])
    # ys_iterative_upper_mini=(data[2])# intensity
    dose_sci = data[4]
    dose_sci_upper = data[9]
    dose_sci_lower = data[10]
    ys_iterative_mini = data[11]
    ys_iterative_upper_mini = data[
        12
    ]  # #dose weighted LET ToF in kev/um analitical method
    ys_iterative_lower_mini = data[13]

    # ys_iterative_mini=(data[14])
    # ys_iterative_upper_mini=(data[15])# #fluence weighted LET ToF in kev/um analitical method
    # ys_iterative_lower_mini=(data[16])
    # tof_curve=(data[7])

    return (
        depth,
        depth_sci,
        rcfdose,
        rcferr,
        dose_sci,
        dose_sci_upper,
        dose_sci_lower,
        ys_iterative_mini,
        ys_iterative_upper_mini,
        ys_iterative_lower_mini,
    )


def read_data_mini(path):
    """read simulation in scintillator data from execel document"""
    data = pd.read_csv(path, header=None, skiprows=1, delimiter=",")
    depth_sim = data[0]  # depth in scintillator
    realdose = data[1]
    quenchedose = data[2]
    ys = data[3]  # LET ToF in kev/um

    return depth_sim, realdose, quenchedose, ys
