import numpy as np
import pandas as pd

def calculateSMD(data, treatmentCol, covariates):

    treated = data[data[treatmentCol] == 1]
    control = data[data[treatmentCol] == 0]

    results = {}

    for col in covariates:

        meanT = treated[col].mean()
        meanC = control[col].mean()

        varT = treated[col].var()
        varC = control[col].var()
        
        pooledSD = np.sqrt((varT + varC) / 2)

        if pooledSD == 0:
            results[col] = np.nan
        else:
            results[col] = (meanT - meanC) / pooledSD

    return pd.Series(results, name="SMD")