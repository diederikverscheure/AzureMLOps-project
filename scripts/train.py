import argparse
import os
from glob import glob
import random
import numpy as np
import pandas as pd
import stat

# This time we will need our Tensorflow Keras libraries, as we will be working with the AI training now
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression


# This AzureML package will allow to log our metrics etc.
from azureml.core import Run

# Important to load in the utils as well!
from utils import getFeatures, getTargets
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--training-folder', type=str, dest='training_folder', help='training folder mounting point.',default='../training_data')
parser.add_argument('--testing-folder', type=str, dest='testing_folder', help='testing folder mounting point.',default='../testing_data')
parser.add_argument('--max-epochs', type=int, dest='max_epochs', help='The maximum epochs to train.')
parser.add_argument('--seed', type=int, dest='seed', help='The random seed to use.')
parser.add_argument('--initial-learning-rate', type=float, dest='initial_lr', help='The initial learning rate to use.')
parser.add_argument('--batch-size', type=int, dest='batch_size', help='The batch size to use during training.')
parser.add_argument('--patience', type=int, dest='patience', help='The patience for the Early Stopping.')
parser.add_argument('--model-type', type=str, dest='model_type', help='The name of the model to use.')
parser.add_argument('--model-name', type=str, dest='model_name', help='The name of the model to use.')
args = parser.parse_args()


training_folder = args.training_folder
print('Training folder:', training_folder)

testing_folder = args.testing_folder
print('Testing folder:', testing_folder)

MAX_EPOCHS = args.max_epochs # Int
SEED = args.seed # Int
INITIAL_LEARNING_RATE = args.initial_lr # Float
BATCH_SIZE = args.batch_size # Int
PATIENCE = args.patience # Int
MODEL_TYPE = args.model_type # String
MODEL_NAME = args.model_name # String

df_train = pd.read_csv(training_folder + '/dataset.csv',index_col=0,parse_dates=[0])
df_test = pd.read_csv(testing_folder + '/dataset.csv',index_col=0,parse_dates=[0])

print("Training samples:", df_train.shape[0])
print("Testing samples:", df_test.shape[0])


# Parse to Features and Targets for both Training and Testing. Refer to the Utils package for more information
X_train = getFeatures(df_train)
y_train = getTargets(df_train)

X_test = getFeatures(df_test)
y_test = getTargets(df_test)

print('Shapes:')
print(X_train.shape)
print(X_test.shape)
print(len(y_train))
print(len(y_test))


# Create an output directory where our AI model will be saved to.
# Everything inside the `outputs` directory will be logged and kept aside for later usage.
model_path = os.path.join('outputs', MODEL_NAME)
os.makedirs('outputs', exist_ok=True,mode=0o777)

## START OUR RUN context.
## We can now log interesting information to Azure, by using these methods.
run = Run.get_context()

if MODEL_TYPE == 'logreg':
    model = LogisticRegression(random_state=SEED).fit(X_train, y_train)

print("Evaluating model")
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)
print(classification_report(y_test, y_pred_test, target_names=['not present','present'])) 

cf_matrix = confusion_matrix(y_test, y_pred_test)
print(cf_matrix)

# We could use this, but we are logging realtime with the callback!
run.log('accuracy_train',accuracy_score(y_train, y_pred_train) )
run.log('accuracy_test',accuracy_score(y_test, y_pred_test) )
run.log('f1_train',f1_score(y_train, y_pred_train) )
run.log('f1_test',f1_score(y_test, y_pred_test) )


## Log Confusion matrix , see https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run.run?view=azure-ml-py#log-confusion-matrix-name--value--description----
cmtx = {
    "schema_type": "confusion_matrix",
    # "parameters": params,
    "data": {
        "class_labels": ['not present','present'],   # ["0", "1"]
        "matrix": [[int(y) for y in x] for x in cf_matrix]
    }
}

run.log_confusion_matrix('Confusion matrix - error rate', cmtx)

# Save the confusion matrix to the outputs.
np.save('outputs/confusion_matrix.npy', cf_matrix)

st = os.stat('outputs')
oct_perm = oct(st.st_mode)
print(oct_perm)

print('Saving model to ',model_path)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print("DONE TRAINING. AI model has been saved to the outputs.")
