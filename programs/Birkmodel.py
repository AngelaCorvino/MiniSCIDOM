import numpy as np

"LET correction simulations in scintillator"


def lightout(S, a, k, dx):
    """Birks'Model"""
    return ((a * S) / (1 + (k * S))) * dx


def lightcorrection(S, a, k, dx):
    """Birk's Model quenching correction / linear trend"""
    return np.divide(
        lightout(S, a, k, dx),
        lightout(S, a, 0, dx),
        out=np.zeros_like(lightout(S, a, k, dx)),
        where=lightout(S, a, 0, dx) != 0,
    )


def dosecorrection(dose, S, a, k, dx):
    return np.divide(
        dose,
        lightcorrection(S, a, k, dx),
        out=np.zeros_like(dose),
        where=lightcorrection(S, a, k, dx) != 0,
    )  # S1 should be a function of depth, i can not make a fit but a ican interpolate, data are not normalized


"LET correction prediction in scintillator"


def letcorrection(
    depth_sci, dose, ys_ana_mini, ys_ana_lower_mini, ys_ana_upper_mini, s
):

    dS = 0  # theoretical values
    dscintillator = 1.023  # [g/cm^3] scintillator density
    dactivelayer = 1.2  # [g/cm^3]
    k = 207.0 / 10000  # [g/MeV cm^2]
    a = 0.9  # scintillator efficiency
    dx = 65  # µm scintillator spatial resolution
    ddx = 1  # spatial resolution error
    k = k / dscintillator * 10  # [micrometers/kev]

    x = np.arange(0, len(dose), 1) * s
    S_a_mini = np.interp(np.arange(0, len(dose), 1) * s, depth_sci, ys_ana_mini)
    S_a_low_mini = np.interp(
        np.arange(0, len(dose), 1) * s, depth_sci, ys_ana_lower_mini
    )
    S_a_up_mini = np.interp(
        np.arange(0, len(dose), 1) * s, depth_sci, ys_ana_upper_mini
    )

    # CORRECTED DOSE

    D_a_mini = dosecorrection(dose, S_a_mini, a, k, dx)
    D_a_up_mini = dosecorrection(dose, S_a_up_mini, a, k, dx)
    D_a_low_mini = dosecorrection(dose, S_a_low_mini, a, k, dx)

    # NORMALIZATION
    area_corrected = np.trapz(D_a_mini[3 : len(D_a_mini)], x[3 : len(x)])

    return D_a_mini, D_a_up_mini, D_a_low_mini, area_corrected, S_a_mini
