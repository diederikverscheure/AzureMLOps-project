# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 10:27:44 2022

@author: dve
"""

import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from steps.utils import connectWithAzure
from azureml.core import Dataset
from azureml.data.datapath import DataPath
from sklearn.model_selection import TimeSeriesSplit

load_dotenv('.env',override=True)
SEED = int(os.environ.get('RANDOM_SEED'))
TRAIN_TEST_SPLIT_FACTOR = float(os.environ.get('TRAIN_TEST_SPLIT_FACTOR'))

def prepareDataset(ws):
    df = pd.read_csv('./data/states.csv')
    # Set correct types
    df = df.convert_dtypes()
    df.loc[:,'created'] = pd.to_datetime(df.loc[:,'created'])
    df.loc[:,'last_changed'] = pd.to_datetime(df.loc[:,'last_changed'])
    df.loc[:,'last_updated'] = pd.to_datetime(df.loc[:,'last_updated'])
    # Drop old data
    df = df.loc[df['last_changed']>=pd.Timestamp('2022-01-01 06:00:00')]

    # Clean states of trackers
    device_trackers = ['device_tracker.m2007j20cg',
                       'device_tracker.pocophone_f1',
                       'device_tracker.xiaomi_communications_89_8b_3e',
                       'device_tracker.xiaomi_communications_b1_9e_1c']
    for d in device_trackers:
        print(d)
        print(df.loc[df.entity_id==d].state.value_counts())
        df.loc[:,'state'] = df.loc[:,'state'].replace('Geofence','home')
        df.loc[(df.entity_id==d)&(df.state!='home')&(df.state!='unavailable'),'state'] = 'not_home'
    
    # Clean binary sensors
    binary_sensors = ['binary_sensor.shellydw2_aac009_door',
                      'binary_sensor.otgw_central_heating']
    for b in binary_sensors:
        print(b)
        print(df.loc[df.entity_id==b].state.value_counts())
        df.loc[(df.entity_id==b)&(df.state!='on'),'state'] = 'off'

    # Sensors
    sensors = ['sensor.co2_074185ff',
               'sensor.co2_0741823e',
               'sensor.shellydw2_aac009_temperature',
               'sensor.otgw_room_temperature',
               'sensor.otgw_outside_temperature',
               'sensor.openweathermap_temperature',
               'sensor.shellyht_adba7f_temperature',
               'sensor.shellyht_9552b5_temperature',
               'sensor.shellyht_955ae2_temperature']
    entities = device_trackers + binary_sensors + sensors
    # Replace strings by values
    dfe = df.loc[df.entity_id.isin(entities)]
    dfe.loc[:,'state'] = dfe.loc[:,'state'].replace(['home','not_home','on','off'],['1','0','1','0'])
    dfe.loc[:,'state'] = pd.to_numeric(dfe.loc[:,'state'],errors='coerce')
    dfi = pd.pivot_table(dfe,values=['state'],index=['last_changed'],columns=['entity_id'],aggfunc=np.median)
    dfi = dfi.droplevel(0,axis=1)
    dfi.sort_index(inplace=True)
    
    # Correction for specific day
    dfi.loc[(dfi.index>=pd.Timestamp('2022-01-16 08:00:00'))&(dfi.index<=pd.Timestamp('2022-01-21 16:00:00')),'device_tracker.pocophone_f1'] = 0
    dfi.loc[(dfi.index>=pd.Timestamp('2022-01-16 08:00:00'))&(dfi.index<=pd.Timestamp('2022-01-16 12:00:00')),'device_tracker.m2007j20cg'] = 0
    dfi.loc[(dfi.index>=pd.Timestamp('2022-01-16 14:15:00'))&(dfi.index<=pd.Timestamp('2022-01-16 17:00:00')),'device_tracker.m2007j20cg'] = 0
    dfi.loc[(dfi.index>=pd.Timestamp('2022-01-16 17:30:00'))&(dfi.index<=pd.Timestamp('2022-01-16 19:30:00')),'device_tracker.m2007j20cg'] = 0
    
    # Interpolate sensors
    dfi.loc[:,sensors] = dfi.loc[:,sensors].interpolate(method='values') #.interpolate(method='nearest')

    # Process nmap trackers
    # dt_range = pd.date_range(dfi.index[0],dfi.index[-1],freq='60s')
    # for d in ['device_tracker.xiaomi_communications_89_8b_3e','device_tracker.xiaomi_communications_b1_9e_1c']:
    #     dft = dfi[d].reindex(dfi.index.union(dt_range),method='nearest').fillna(0)
    #     dft = dft.ewm(halflife=60).mean()
    #     dft = dft/dft.max()
    #     dfi[d+'_processed'] = dft.reindex(dfi.index,method='nearest')
    
    # Correct door sensor and device trackers (fill non-empties)
    for b in binary_sensors+['device_tracker.m2007j20cg','device_tracker.pocophone_f1']:
        print(b)
        if b.find('door') >= 0:
            max_duration= 3600
        elif b.find('otgw') >= 0:
            max_duration = 3600*6
        elif b.find('device_tracker') >= 0:
            max_duration = 3600*12
        ix = np.where(~np.isnan(dfi[b].values))
        ix = np.append(ix[0],dfi.shape[0])
        # Use first non-empty to fill all before
        dfi[b].iloc[0:ix[0]+1].fillna(value=0.0,axis=0,inplace=True)
        for i in range(len(ix)-1):
            if dfi[b].iloc[ix[i]] == 0.0:
                # Front fill until next event
                dfi[b].iloc[ix[i]:ix[i+1]].fillna(method='ffill',axis=0,inplace=True)
            elif (dfi[b].iloc[ix[i]] == 1.0) and (ix[i+1] < dfi.shape[0]):
                if ((dfi.index[ix[i+1]]-dfi.index[ix[i]]).seconds < max_duration):
                    # Front fill until next event                                  
                    dfi[b].iloc[ix[i]:ix[i+1]].fillna(method='ffill',axis=0,inplace=True)
                else:
                    print(dfi.index[ix[i]],' - ', dfi.index[ix[i+1]],' - ', (dfi.index[ix[i+1]]-dfi.index[ix[i]]).seconds,' s ',dfi[b].iloc[ix[i]], '-', dfi[b].iloc[ix[i+1]])
            else:
                # Front fill with zeros until next event
                dfi[b].iloc[ix[i]+1:ix[i+1]].fillna(value=0.0,axis=0,inplace=True)
    
    # Determine bedroom presence
    doy = dfi.index.dayofyear.unique()
    dfi['presence_bedroom'] = 0.0
    for d in doy:
        dfd = dfi.loc[(dfi.index.dayofyear==d)&(dfi['device_tracker.xiaomi_communications_b1_9e_1c']==1)]
        if dfd.shape[0] != 0:
            dt_up = dfd.index[np.argmin(dfd.index)] # Opstaan
            dt_sleep = dfd.index[np.argmax(dfd.index)] # Slapen
            dfi.loc[(dfi.index.dayofyear==d)&(dfi.index<dt_up),'presence_bedroom'] = 1.0
            dfi.loc[(dfi.index.dayofyear==d)&(dfi.index>dt_sleep),'presence_bedroom'] = 1.0
            if dfd.index[0].dayofweek <= 4:
                dfi.loc[(dfi.index.dayofyear==d)&(dfi.index.hour>=8)&(dfi.index.hour<12),'device_tracker.m2007j20cg'] = 0.0
                dfi.loc[(dfi.index.dayofyear==d)&(dfi.index.hour>=13)&(dfi.index.hour<17),'device_tracker.m2007j20cg'] = 0.0
                
    # Determine living presence
    dfi['presence_living'] = (((dfi['device_tracker.pocophone_f1'].astype(bool))|(dfi['device_tracker.m2007j20cg'].astype(bool)))&(~dfi['presence_bedroom'].astype(bool))).astype(float)
    dfi['co2_outside'] = 400.0
    dfi.rename(columns={'sensor.co2_0741823e': 'co2_living','sensor.co2_074185ff': 'co2_bedroom',\
                        'sensor.otgw_room_temperature': 'temperature_living',\
                        'sensor.openweathermap_temperature': 'temperature_outside',\
                        'binary_sensor.shellydw2_aac009_door': 'window_living'},inplace=True)        

    # Plot processed results
    plt.close('all')
    f, ax = plt.subplots(1,1)
    ax.plot(dfi.index,dfi['co2_living']-dfi['co2_outside'])
    ax2 = ax.twinx()
    ax2.plot(dfi.index,dfi['temperature_living']-dfi['temperature_outside'])
    #ax.fill_between(dfi.index,0,3000,dfi['device_tracker.m2007j20cg']==1,color='green',alpha=0.2)
    #ax.fill_between(dfi.index,0,3000,dfi['device_tracker.pocophone_f1']==1,color='red',alpha=0.2)
    #ax.fill_between(dfi.index,0,1500,dfi['device_tracker.xiaomi_communications_89_8b_3e']>0.5,color='green',alpha=0.2)
    #ax.fill_between(dfi.index,1500,3000,dfi['device_tracker.xiaomi_communications_b1_9e_1c']>0.5,color='red',alpha=0.2)    
    ax.fill_between(dfi.index,0,3000,dfi['window_living']>0.5,color='red',alpha=0.2)    
    
    f, ax = plt.subplots(2,1)
    ax[0].plot(dfi.index,dfi['co2_living'])
    ax[0].fill_between(dfi.index,0,3000,dfi['presence_living']>0.5,color='green',alpha=0.2)
    
    ax[1].plot(dfi.index,dfi['co2_bedroom'])
    ax[1].fill_between(dfi.index,0,3000,dfi['presence_bedroom']>0.5,color='red',alpha=0.2)
    
    keep = ['co2_living','temperature_living','presence_living','window_living',\
            'co2_bedroom','presence_bedroom',\
            'co2_outside','temperature_outside']
    dfi[keep].dropna().to_csv('./processed_data/dataset.csv') 

    # Upload the directory as a new dataset
    print('Uploading dataset now ...')
    dataset = Dataset.File.upload_directory(src_dir='./processed_data',
                        target = DataPath(datastore=ws.get_default_datastore(), path_on_datastore='processed_data/'),
                        overwrite=True)

    print('... uploaded file, now creating a dataset ...')

    # Make sure to register the dataset whenever everything is uploaded.
    new_dataset = dataset.register(ws,
                            name='Project_data',
                            description='Timeseries data for my project',
                            tags={'dt_start': str(dfi.index[0]), 
                                  'dt_end': str(dfi.index[-1]),
                                  'GIT-SHA': os.environ.get('GIT_SHA')}, # Optional tags, can always be interesting to keep track of these!
                            create_new_version=True)    

def trainTestSplitData(ws):
    default_datastore = ws.get_default_datastore()
    dataset = Dataset.get_by_name(ws,'Project_data')
    print('Starting to process dataset')
    dataset.download('processed_data')
    dfi = pd.read_csv('./processed_data/dataset.csv',index_col=0,parse_dates=[0])
    tscv = TimeSeriesSplit(n_splits=int(1/TRAIN_TEST_SPLIT_FACTOR))
    for train_index, test_index in tscv.split(dfi.values):
        print("TRAIN:", train_index, "TEST:", test_index)
        dfi_train, dfi_test = dfi.iloc[train_index], dfi.iloc[test_index]
    dfi_train.to_csv('./training_data/dataset.csv')
    dfi_test.to_csv('./testing_data/dataset.csv')
    
    print('Uploading datasets now ...')
    training_dataset = Dataset.File.upload_directory(src_dir='./training_data',
                        target = DataPath(datastore=ws.get_default_datastore(), path_on_datastore='training_data/'),
                        overwrite=True)
    testing_dataset = Dataset.File.upload_directory(src_dir='./testing_data',
                        target = DataPath(datastore=ws.get_default_datastore(), path_on_datastore='testing_data/'),
                        overwrite=True)
    

    training_dataset = training_dataset.register(ws,
        name=os.environ.get('TRAIN_SET_NAME'), # Get from the environment
        description='Timeseries training data',
        tags={'dt_start': str(dfi_train.index[0]), 
              'dt_end': str(dfi_train.index[-1]),
              'Split size': str(1 - TRAIN_TEST_SPLIT_FACTOR),
              'type': 'training',
              'GIT-SHA': os.environ.get('GIT_SHA')},
        create_new_version=True)
    print(f"Training dataset registered: {training_dataset.id} -- {training_dataset.version}")

    testing_dataset = testing_dataset.register(ws,
        name=os.environ.get('TEST_SET_NAME'), # Get from the environment
        description='Timeseries testing data',
        tags={'dt_start': str(dfi_test.index[0]), 
              'dt_end': str(dfi_test.index[-1]),
              'Split size': str(1 - TRAIN_TEST_SPLIT_FACTOR),
              'type': 'testing',
              'GIT-SHA': os.environ.get('GIT_SHA')},
        create_new_version=True)
    print(f"Testing dataset registered: {testing_dataset.id} -- {testing_dataset.version}")

    
if __name__ == '__main__':
    # Load data
    print(os.getcwd())
    ws = connectWithAzure()
    # Set these values to 'false' if you want to skip them.
    print('Process data:',os.environ.get('PROCESS_DATA'))
    if os.environ.get('PROCESS_DATA') == 'true':
        print('Processing the data')
        prepareDataset(ws)
    else:
        print('Skipping the process part')

    if os.environ.get('SPLIT_DATA') == 'true':
        print('Splitting the data')
        trainTestSplitData(ws)
    else:
        print('Skipping the split part')        
    