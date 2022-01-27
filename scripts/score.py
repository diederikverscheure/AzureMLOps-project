import os
import numpy as np
import pandas as pd
import json
import pickle
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def init():
    global model

    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_path = os.path.join(os.environ.get('AZUREML_MODEL_DIR'), 'model.pkl')
    print(os.listdir(os.environ.get('AZUREML_MODEL_DIR')))
    print(model_path)
    with open(model_path,'rb') as f:
        model = pickle.load(f)
    #run('{"dt": ["2020-01-01 09:00:00","2020-01-01 09:00:05","2020-01-01 09:00:10","2020-01-01 09:00:15","2020-01-01 09:00:20","2020-01-01 09:00:25","2020-01-01 09:00:30","2020-01-01 09:00:35","2020-01-01 09:00:40","2020-01-01 09:00:45"], "co2": [100,200,300,400,500,600,700,800,900,1000]}')        

def run(json_data):
    data = json.loads(json_data)
    df = pd.DataFrame(data)
    df.loc[:,df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df.set_index(df.columns[0],inplace=True)
    X = getScoreFeatures(df)
    likelyhood = model.predict_proba(X)
    return 'Presence likelihood = ' + str(likelyhood[:,1])

def plot_decision_boundary():
    xx, yy = np.mgrid[0:3000, -5:5:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)
    
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                          vmin=0, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])
    
    ax.set(xlim=(0, 3000), ylim=(-5, 5),
           xlabel="CO2", ylabel="CO2 derivative")


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


