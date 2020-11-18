# import xgboost as xgb
# from xgboost.sklearn import XGBClassifier

import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, History 

from sklearn import metrics
from sklearn.metrics import roc_auc_score

# import matplotlib.pyplot as plt

# from sklearn.externals import joblib


import pickle

# import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# fix random seed for reproducibility
np.random.seed(7)

import psutil
import time

def create_model(params): 
    
    x_train_temp = train_sample[df_features].copy() 
    y_train_temp = train_sample[df_target].copy()
    
    model = Sequential()
    model.add(Dense(params['units1'], input_shape=(x_train_temp.shape[1],)))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))
    if(params['num_layers'] == 'two_hidden'):
        model.add(Dense(params['units2']))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['dropout2']))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=params['optimizer'])
    
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=params['early_stop_rounds'])
    history = History()
    
    model.fit(x_train_temp, y_train_temp,
              batch_size=params['batch_size'],
              epochs=1,
              callbacks=[early_stop, history],
              verbose=2,
              validation_data=(valid_sample[df_features],valid_sample[df_target])) 
    
    [loss, acc] = model.evaluate(valid_sample[df_features],valid_sample[df_target], verbose=0)
    
    global num
    mem = psutil.virtual_memory()
    
    if(np.isnan(acc)):
        
        print("{}) Validation set Accuracy: NaN".format(num),"\tAvailable Mem:",(mem.available/1024)/1024,"mb")
        
        
        f1 = open("status.txt", "a+")
        f1.write("num:{} , loss:{}, Valid acc: {}, params: {} \n \n".format(num,loss, acc,params))
        f1.close() 
      
        
        num = num + 1
        return {'loss': np.inf, 'status': STATUS_OK, 'paras': params}
    else:
        
        model.fit(x_train_temp, y_train_temp,
                  batch_size=params['batch_size'],
                  epochs=500,
                  callbacks=[early_stop, history],
                  verbose=2,
                  validation_data=(valid_sample[df_features],valid_sample[df_target]))
        
        [loss, acc] = model.evaluate(valid_sample[df_features],valid_sample[df_target], verbose=0)
        
        mem = psutil.virtual_memory()
        if(np.isnan(acc)):
            print("{}) Validation set Accuracy: NaN".format(num),"\tAvailable Mem:",(mem.available/1024)/1024,"mb")
            
            
            f1 = open("status.txt", "a+")
            f1.write("num:{} , loss:{}, Valid acc: {}, params: {} \n \n".format(num,loss, acc,params))
            f1.close() 
            
            num = num + 1
            return {'loss': np.inf, 'status': STATUS_OK, 'params': params}
        print("{}) Validation set Accuracy: {:7.2f}%".format(num,acc*100),"\tAvailable Mem:",(mem.available/1024)/1024,"mb")
        
        f1 = open("status.txt", "a+")
        f1.write("num:{} , loss:{}, Valid acc: {}, params: {} \n \n".format(num,loss, acc,params))
        f1.close() 
        
        num = num + 1
        return {'loss': -acc, 'status': STATUS_OK, 'params': params}
    
    
def get_best_model_nn(space):

    trials=Trials()
    global num
    num = 1
    best = fmin(create_model, space, algo=tpe.suggest, max_evals=50, trials=trials)

    return trials.best_trial['result']

def train_best_model(best_params):

    x_train_temp = train_sample[df_features].copy() 
    y_train_temp = train_sample[df_target].copy()
    model = Sequential()
    model.add(Dense(best_params['units1'], input_shape=(x_train_temp.shape[1],)))
    model.add(Activation(best_params['activation']))
    model.add(Dropout(best_params['dropout1']))
    if(best_params['num_layers'] == 'two_hidden'):
        model.add(Dense(best_params['units2']))
        model.add(Activation(best_params['activation']))
        model.add(Dropout(best_params['dropout2']))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=best_params['optimizer'])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=best_params['early_stop_rounds'])
    history = History()

    model.fit(x_train_temp, y_train_temp,
              batch_size=best_params['batch_size'],
              epochs=500,
              callbacks=[early_stop, history],
              verbose=2,
              validation_data=(valid_sample[df_features],valid_sample[df_target]))
    return model
    
train_o = pd.read_pickle('train_insurance_2017_new.pkl')
valid_o = pd.read_pickle('valid_insurance_2017_new.pkl')
# test_o = pd.read_pickle('../../data/test_insurance_2017.pkl')

# train_o.drop(['StartedFlag'],axis=1,inplace=True)
# valid_o.drop(['StartedFlag'],axis=1,inplace=True)
# test_o.drop(['StartedFlag'],axis=1,inplace=True)

# train, valid, test, df_features, df_target = data(all_data,target='NewEstimateTotal')
# del all_data
df_target = 'StartedFlag'
# df_features = list(set(train_o.columns) - set([df_target]))
df_features = list(train_o.columns)
df_features.remove('NewEstimateTotal')
df_features.remove('StartedFlag')

# Saving df_features
import pickle
with open('df_features.pkl', 'wb') as f:
    pickle.dump(df_features, f)   

train_sample = train_o.head(100000)
valid_sample = valid_o.head(20000)
# del train
# del valid
del train_o
del valid_o
# del test_o
# del test

if __name__ == '__main__':
    start_time = time.time()
    space = {'num_layers': hp.choice('num_layers',['one_hidden', 'two_hidden']),
                    'units1': hp.choice('units1', [32, 64, 128, 256,512]),
                    'units2': hp.choice('units2', [32, 64, 128, 256,512]),
                    'dropout1': hp.uniform('dropout1', .25,.75),
                    'dropout2': hp.uniform('dropout2',  .25,.75),
                    'batch_size' : hp.choice('batch_size', [16,32,64,128]),
#                     'nb_epochs' :  500,
                    'optimizer': hp.choice('optimizer',['rmsprop', 'adam', 'nadam','sgd']),
                    'activation': hp.choice('activation',['relu','sigmoid']),
                    'early_stop_rounds': hp.choice('early_stop_rounds',[10,20,30,40,50]),
                }
    best_trails1 = get_best_model_nn(space)


    f = open("started_flag_100000_hyperopt_keras_2017.txt", "w")
    f.write("Took {} seconds with best trails as: {}".format(time.time()-start_time,best_trails1))
    f.close() 

    # Training with the best params

    best_model = train_best_model(best_trails1['params'])

    # To save model
#     best_model.save('insurance_classify_100000_estimate_total_hyperopt_keras_2017.h5')

    # with open('insurance_classify_100000_estimate_total_hyperopt_2017.pkl', 'wb') as f:
    #     pickle.dump(best_model, f)
    
    # serialize model to JSON
    model_json = best_model.to_json()
    with open("insurance_classify_100000_started_flag_hyperopt_keras_2017_new.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    best_model.save_weights("insurance_classify_100000_started_flag_hyperopt_keras_2017_new.h5")
    print("Saved model to disk")

    #Making Preds on train data
    train_o = pd.read_pickle('train_insurance_2017_new.pkl')
    valid_o = pd.read_pickle('valid_insurance_2017_new.pkl')
    
    training_data = pd.concat([train_o,valid_o])
    
    train_new = pd.DataFrame()
    for i,chunk in enumerate(np.array_split(training_data, 5)):
        print(i)
        pred = best_model.predict(chunk[df_features]).flatten()
        chunk['probability'] = pred
        chunk['preds'] = np.where(pred>=0.5,1,0)
        train_new = train_new.append(chunk)
    train_new.to_pickle('train_insurance_2017_preds_started_flag_hyperopt_keras_new.pkl')
    acc_train = accuracy_score(train_new.StartedFlag, train_new.preds)

    f = open("train_preds_started_flag_100000_hyperopt_keras_2017.txt", "w")
    f.write("Accuracy on train data: {}".format(acc_train))
    f.close()
    
    
    #Making Preds on test data
    
    test_o = pd.read_pickle('test_insurance_2017_new.pkl')
#     test_o.drop(['StartedFlag'],axis=1,inplace=True)
    
    test_new = pd.DataFrame()
    for i,chunk in enumerate(np.array_split(test_o, 5)):
        print(i)
        pred = best_model.predict(chunk[df_features]).flatten()
        chunk['probability'] = pred
        chunk['preds'] = np.where(pred>=0.5,1,0)
        test_new = test_new.append(chunk)
    test_new.to_pickle('test_insurance_2017_preds_started_flag_hyperopt_keras_new.pkl')
    acc_test = accuracy_score(test_new.StartedFlag, test_new.preds)

    f = open("test_preds_started_flag_100000_hyperopt_keras_2017.txt", "w")
    f.write("Accuracy on test data: {}".format(acc_test))
    f.close()
    
#     Saving Actual CSVs
    def reverse_dummies(dummies_df):
        sep = '_'
        dummy_cols = list(set(col.split(sep)[0] for col in dummies_df.columns if sep in col))
        print(dummy_cols)
        other_cols = [col for col in dummies_df.columns if sep not in col]
        print(other_cols)
        dfs = []
        for i, col in enumerate(dummy_cols):
            print(i, col)
            dfs.append(dummies_df.filter(regex=col).rename(columns=lambda name: name.split(sep)[1]).idxmax(axis=1))

        df = pd.concat(dfs + [dummies_df[other_cols]], axis=1)
        df.columns = dummy_cols + other_cols
        return df
    
    training_csv = reverse_dummies(train_new)
    training_csv.to_csv('training_results.csv')
    testing_csv = reverse_dummies(test_new)
    testing_csv.to_csv('testing_results.csv')