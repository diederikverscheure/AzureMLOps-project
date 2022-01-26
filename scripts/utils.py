import numpy as np
from typing import List
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def getTargets(df):
    return np.hstack((df['presence_living'].values,df['presence_bedroom'].values))

def smooth(index,y,derivative=1):
    dt = (index-index[0]).total_seconds()
    dtr = np.linspace(dt[0],dt[-1],len(dt))
    f = interp1d(dt,y,'linear')
    dxr = savgol_filter(f(dtr),101,1,derivative)
    fr = interp1d(dtr,dxr,'linear')
    dx = fr(dt)
    return dx
    
def getFeatures(df):
    df['delta_living']= df['co2_living']-df['co2_outside']
    df['delta_bedroom'] = df['co2_bedroom']-df['co2_outside']
    df['deriv_living'] = smooth(df.index,df['co2_living'],derivative=1)
    df['deriv_bedroom'] = smooth(df.index,df['co2_bedroom'],derivative=1)
    return np.vstack((df[['delta_living','deriv_living']].values,
                      df[['delta_bedroom','deriv_bedroom']].values))

