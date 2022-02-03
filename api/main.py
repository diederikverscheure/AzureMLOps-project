import numpy as np
from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pickle


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def likelyhood(str: data):
    data = json.loads(json_data)
    df = pd.DataFrame(data)
    df.loc[:,df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df.set_index(df.columns[0],inplace=True)
    X = getScoreFeatures(df)
    likelyhood = model.predict_proba(X)
    return 'Presence likelihood = ' + str(likelyhood[:,1])    
  
