import numpy as np
from typing import List
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def getTargets(df):
    return np.hstack((df['presence_living'].values,df['presence_bedroom'].values))

def smooth(index,y,derivative=1):
    dt = (index[1:]-index[0:-1]).total_seconds()
    dx = savgol_filter(y,11,1,derivative,delta=np.mean(dt),mode='nearest')
    return dx

def getScoreFeatures(df):
    df['deriv'] = smooth(df.index,df.iloc[:,0]-400,derivative=1)
    return df.iloc[:,0:2].values
    
def getFeatures(df):
    df['delta_living']= df['co2_living']-df['co2_outside']
    df['delta_bedroom'] = df['co2_bedroom']-df['co2_outside']
    df['deriv_living'] = smooth(df.index,df['co2_living'],derivative=1)
    df['deriv_bedroom'] = smooth(df.index,df['co2_bedroom'],derivative=1)
    return np.vstack((df[['delta_living','deriv_living']].values,
                      df[['delta_bedroom','deriv_bedroom']].values))

