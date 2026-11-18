import numpy as np

def calculateRMST(survivalFunction, horizon=36):
    curve = survivalFunction.loc[survivalFunction.index <= horizon]

    times = np.r_[0, curve.index.to_numpy()]

    probabilites = np.r_[1, curve.to_numpy()]

    return np.trapezoid(probabilites, times)