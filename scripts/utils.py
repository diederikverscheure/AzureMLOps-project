import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

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


def buildModel(inputShape: tuple, classes: int) -> Sequential:
    model = Sequential()
    height, width, depth = inputShape
    inputShape = (height, width, depth)
    chanDim = -1

    # CONV => RELU => POOL layer set              # first CONV layer has 32 filters of size 3x3
    model.add(Conv2D(32, (3, 3), padding="same", name='conv_32_1', input_shape=inputShape))
    model.add(Activation("relu"))                 # ReLU (Rectified Linear Unit) activation function
    model.add(BatchNormalization(axis=chanDim))   # normalize activations of input volume before passing to next layer
    model.add(MaxPooling2D(pool_size=(2, 2)))     # progressively reduce spatial size (width and height) of input 
    model.add(Dropout(0.25))                      # disconnecting random neurons between layers, reduce overfitting

    # (CONV => RELU) * 2 => POOL layer set          # filter dimensions remain the same (3x3)
    model.add(Conv2D(64, (3, 3), padding="same", name='conv_64_1'))   # increase total number of filters learned (from 32 to 64)
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same", name='conv_64_2'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 3 => POOL layer set
    model.add(Conv2D(128, (3, 3), padding="same", name='conv_128_1'))   # total number of filters learned by CONV layers has doubled (128)
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same", name='conv_128_2'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same", name='conv_128_3'))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # first (and only) set of fully connected layer (FC) => RELU layers
    model.add(Flatten())
    model.add(Dense(512, name='fc_1'))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # softmax classifier
    model.add(Dense(classes, name='output'))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

# model = buildModel((64, 64, 3), len(LABELS))