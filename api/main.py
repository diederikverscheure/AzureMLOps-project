import numpy as np
from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pickle
import json
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dfl = pd.DataFrame(columns=['t','co2']).set_index('t')
dfb = pd.DataFrame(columns=['t','co2']).set_index('t')

dfl.to_csv('outputs/living.csv')
dfb.to_csv('outputs/bedroom.csv')

with open('outputs/model.pkl','rb') as f:
    model = pickle.load(f)

def smooth(index,y,derivative=1):
    dt = (index[1:]-index[0:-1]).total_seconds()
    dx = savgol_filter(y,11,1,derivative,delta=np.mean(dt),mode='nearest')
    return dx

def getScoreFeatures(df):
    df['deriv'] = smooth(df.index,df.iloc[:,0]-400,derivative=1)
    return df.iloc[:,0:2].values
    
@app.post('/likelyhood')
async def likelyhood(json_data: str):
    data = json.loads(json_data)
    df = pd.DataFrame(data)
    df.loc[:,df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df.set_index(df.columns[0],inplace=True)
    X = getScoreFeatures(df)
    likelyhood = model.predict_proba(X)
    return 'Presence likelihood = ' + str(likelyhood[:,1])    
  
@app.get('/rooms/{room}')
async def show_room(room: str):
    df = pd.read_csv('outputs/' + room + '.csv',index_col=[0],parse_dates=[0])
    return df.to_string()

@app.post('/rooms/{room}')
async def add_measurement(room: str, co2: float):
    df = pd.read_csv('outputs/' + room + '.csv',index_col=[0],parse_dates=[0])
    dfn = pd.DataFrame({'t': pd.Timestamp.now(), 'co2': co2},index=[0]).set_index('t')
    df = pd.concat([df,dfn])
    df.to_csv('outputs/' + room + '.csv')
    return df.to_string()

@app.get('/rooms/{room}/presence')
async def show_room(room: str):
    df = pd.read_csv('outputs/' + room + '.csv',index_col=[0],parse_dates=[0])
    X = getScoreFeatures(df)
    likelyhood = model.predict_proba(X)
    return likelyhood[-1,1]
