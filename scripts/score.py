import os
import numpy as np
import json
import pickle
from sklearn.linear_model import LogisticRegression

def init():
    global model

    # The AZUREML_MODEL_DIR environment variable indicates
    # a directory containing the model file you registered.
    model_path = os.path.join(os.environ.get('AZUREML_MODEL_DIR'), 'model.pkl')
    print(os.listdir(os.environ.get('AZUREML_MODEL_DIR')))
    print(model_path)
    with open(model_path,'rb') as f:
        model = pickle.load(f)

def run(json_data):
    data = json.loads(json_data)
    X = np.asarray(data['data']).reshape(-1,2)
    likelyhood = model.predict_proba(X)
    return 'Presence likelihood = ' + str(likelyhood[0,1])

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

