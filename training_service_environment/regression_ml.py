# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

import numpy as np
import pandas as pd
import sys
import psutil
import time

from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, History 

from psqlConnector import deleteQuery,insertQuery,getRowCount
from dataTransferConnector import get_credentials




# Data fetch
def data_fetch(csv_name):
    print("Training process has initiated, reviewing training data set...")
    try:
        import subprocess
        sub_output = subprocess.check_output('file -b --mime-encoding datasets/{}.csv'.format(csv_name), shell=True)
        encoding_csv = sub_output.decode('utf-8')
        df = pd.read_csv('datasets/{}.csv'.format(csv_name), encoding = encoding_csv)
        
        global unique_id_column
        unique_id_column = get_index_col(df)
        
        if(df[unique_id_column].nunique()!=df.shape[0]):
            print('There are duplicate values in the index column. \nExiting the script...')
            flush_memory()
            
        df.set_index(unique_id_column,inplace=True)    
        
        global train_columns
        train_columns = df.columns
        return df
    
    except UnicodeError as e: 
        print("The {} csv has wrong encoding. Please use a csv with utf-8 encoding. \nExiting the script...".format(csv_name))
        flush_memory()
        
    except Exception as e:
        print("The {} csv is not present in the datasets folder. \nExiting the script...".format(csv_name))
        flush_memory()

def data_fetch_test_array(csv_name):
    try:
        import subprocess
        sub_output = subprocess.check_output('file -b --mime-encoding datasets/{}.csv'.format(csv_name), shell=True)
        encoding_csv = sub_output.decode('utf-8')
        df = pd.read_csv('datasets/{}.csv'.format(csv_name),encoding = encoding_csv)
        
        if(unique_id_column not in list(df.columns)):
            print('The given index column is not present in the given test csv.')
            flush_memory()
        
        if(df[unique_id_column].nunique()!=df.shape[0]):
            print('There are duplicate values in the index column of the given test csv. \nExiting the script...')
            flush_memory()
        
        df.set_index(unique_id_column,inplace=True)    
        
        return df
    except UnicodeError as e: 
        print("The {} csv has wrong encoding. Please use a csv with utf-8 encoding. \nExiting the script...".format(csv_name))
        flush_memory()
        
    except Exception as e:
        print("The {} csv is not present in the datasets folder. \nExiting the script...".format(csv_name))
        flush_memory()
        
def get_index_col(df):
    
    print('The columns of the given training csv are:')
    print(list(df.columns))
    index_col = input('Enter the name of the unique index column: ')
    
    if(index_col not in list(df.columns)):
        print('The given index column is not present in the training csv.')
        flush_memory()
    
    return index_col
    
# Pre Processing
def get_string_cols(df):    
    string_cols = list(df.select_dtypes(include=['object','category']).columns)
    return string_cols

def get_num_cols(df):    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_cols = list(df.select_dtypes(include=numerics).columns)    
    return num_cols

def pre_process(df,label_col):
    
    try:
        y = df[label_col]
    except Exception as e:
        print("The {} column in not present in the given csv. \nExiting the script...".format(label_col))
        flush_memory()
        
    print("Pre-Processing data set and creating a training data set, validation data set and test data set...")
    
    df = df.drop(label_col,axis=1)
    string_cols = get_string_cols(df)
    num_cols = get_num_cols(df)
    substring = ':string'
    num_cat_cols = []
    for string in num_cols:
        if(substring in string):
            num_cat_cols.append(string)
    for col in num_cat_cols:
        df[col] = df[col].astype('object')
    categorical_cols = string_cols + num_cat_cols
    df = pd.get_dummies(df,columns=categorical_cols,sparse=True)
    global training_dummy_columns
    training_dummy_columns = df.columns
    df.columns = df.columns.str.replace(':string','')
    df[label_col] = y  
    return df

def pre_process_test_csv(df,label_col,training_dummy_columns):
    df_copy = df.copy()

    try:
        y = df_copy[label_col]
    except Exception as e:
        print("The {} column in not present in the given test csv. \nExiting the script...".format(label_col))
        flush_memory()

    y = df_copy[label_col]
    df_copy = df_copy.drop(label_col,axis=1)
    string_cols = get_string_cols(df_copy)
    num_cols = get_num_cols(df_copy)
    substring = ':string'
    num_cat_cols = []
    for string in num_cols:
        if(substring in string):
            num_cat_cols.append(string)
    for col in num_cat_cols:
        df_copy[col] = df_copy[col].astype('object')
    categorical_cols = string_cols + num_cat_cols
    df_copy = pd.get_dummies(df_copy,columns=categorical_cols,sparse=True)
    missing_cols = set( training_dummy_columns ) - set( df_copy.columns )
    for col in missing_cols:
        df_copy.loc[:,col] = 0
    df_copy = df_copy[training_dummy_columns]
    df_copy.columns = df_copy.columns.str.replace(':string','')
    df_copy[label_col] = y
    return df_copy
    
# Train Test Split
def split_train_test(df,label_col,test_size=0.2):    
    X = df.loc[:,df.columns != label_col]
    y = df[label_col]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=42)
    return train_x, test_x, train_y, test_y

# Model Training
def get_search_space():    
    space = {'num_layers': hp.choice('num_layers',['one_hidden', 'two_hidden']),
                'units1': hp.choice('units1', [32, 64, 128, 256,512]),
                'units2': hp.choice('units2', [32, 64, 128, 256,512]),
                'dropout1': hp.uniform('dropout1', .25,.75),
                'dropout2': hp.uniform('dropout2',  .25,.75),
                'batch_size' : hp.choice('batch_size', [16,32,64,128]),
                'nb_epochs' :  500,
                'optimizer': hp.choice('optimizer',['rmsprop', 'adam', 'nadam','sgd']),
                'activation': hp.choice('activation',['relu','sigmoid']),
                'early_stop_rounds': hp.choice('early_stop_rounds',[10,20,30,40,50]),
            }
    return space

def data(csv_name,label_col,test_arg,use_lime,features_subset):    
    pre_processed_data = pre_process(df=input_training_data,label_col=label_col)
    x_train, x_test, y_train, y_test = split_train_test(df=pre_processed_data,label_col=label_col)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    test_arg_processed = []
    if(test_arg!=None):
        for test_csv in test_arg:
            test_dummy = data_fetch_test_array(test_csv)
            test_dummy_processed = pre_process_test_csv(df=test_dummy,label_col=label_col,training_dummy_columns=training_dummy_columns)
            test_arg_processed.append(test_dummy_processed)
    if(use_lime!=None):
        x_train_lime = x_train[features_subset]
        x_test_lime = x_test[features_subset]
        x_valid_lime = x_valid[features_subset]
        y_train_lime = y_train
        y_test_lime = y_test
        y_valid_lime = y_valid
        test_arg_processed_lime = []
        for test_dummy in test_arg_processed:
            y_test_dummy = test_dummy[label_col]
            test_dummy = test_dummy[features_subset] 
            test_dummy_copy = test_dummy.copy()
            test_dummy_copy[label_col] = y_test_dummy
            test_arg_processed_lime.append(test_dummy_copy)
        return input_training_data, x_train_lime, x_test_lime, x_valid_lime, y_train_lime, y_test_lime, y_valid_lime, test_arg_processed_lime
    return input_training_data, x_train, x_test, x_valid, y_train, y_test, y_valid, test_arg_processed

def create_model(params):
    global num
    print("step <{}> of {} in progress".format(num,run_value))
    x_train_temp = x_train.copy() 
    y_train_temp = y_train.copy()
    model = Sequential()
    model.add(Dense(params['units1'], input_shape=(x_train_temp.shape[1],)))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))
    if(params['num_layers'] == 'two_hidden'):
        model.add(Dense(params['units2']))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['dropout2']))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mse', metrics=['mse'],
                  optimizer=params['optimizer'])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=params['early_stop_rounds'])
    terminate_nan = keras.callbacks.TerminateOnNaN()
    history = History()
    model.fit(x_train_temp, y_train_temp,
              batch_size=params['batch_size'],
              epochs=500,
              callbacks=[early_stop, terminate_nan, history],
              verbose=0,
              validation_data=(x_valid,y_valid)) 
    [loss, mse] = model.evaluate(x_valid,y_valid, verbose=0)
    rmse_val = loss ** 0.5
    pct_error_val = (rmse_val / y_valid.mean())*100
    est_time_remaining = ((time.time() - start_time_evals) / num) * (run_value-num)
    m, s = divmod(est_time_remaining, 60)
    h, m = divmod(m, 60)
    mem = psutil.virtual_memory()
    print("Results of Optimization Step <{}>:".format(num))
    if(np.isnan(mse)):
        print("({}) Validation error: NaN".format(num),"|","Remaining Available Mem:",(mem.available/1024)/1024,"mb","|","Estimated time remaining:",int(h),"hours",int(m),"minutes",int(s),"seconds","\n")
        num = num + 1
        return {'loss': np.inf, 'status': STATUS_OK, 'params': params}
    print("({}) Validation error: {:7.2f} %".format(num,pct_error_val),"|","Remaining Available Mem:",(mem.available/1024)/1024,"mb","|","Estimated time remaining:",int(h),"hours",int(m),"minutes",int(s),"seconds","\n")
    num = num + 1
    return {'loss': loss**0.5, 'status': STATUS_OK, 'params': params}

def train_best_model(best_params):   
    print('Neural Network Architecture optimization complete. Initiating training of model...') 
    x_train_temp = x_train.copy() 
    y_train_temp = y_train.copy()
    model = Sequential()
    model.add(Dense(best_params['units1'], input_shape=(x_train_temp.shape[1],)))
    model.add(Activation(best_params['activation']))
    model.add(Dropout(best_params['dropout1']))
    if(best_params['num_layers'] == 'two_hidden'):
        model.add(Dense(best_params['units2']))
        model.add(Activation(best_params['activation']))
        model.add(Dropout(best_params['dropout2']))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mse', metrics=['mse'],
                  optimizer=best_params['optimizer'])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=best_params['early_stop_rounds'])
    history = History()
    model.fit(x_train_temp, y_train_temp,
              batch_size=best_params['batch_size'],
              epochs=500,
              callbacks=[early_stop, history],
              verbose=0,
              validation_data=(x_valid,y_valid)) 

    return model


def get_best_model(csv_name,label_col,test_arg,use_lime=None,features_subset=None):
    global x_train, x_test, x_valid, y_train, y_test, y_valid
    input_df, x_train, x_test, x_valid, y_train, y_test, y_valid, test_arg_processed = data(csv_name=csv_name,label_col=label_col,test_arg=test_arg,use_lime=use_lime,features_subset=features_subset)
    trials=Trials()
    space = get_search_space()
    print("Neural Network Architecture optimization in progress, please be patient...")
    global start_time_evals
    start_time_evals = time.time()
    best = fmin(create_model, space, algo=tpe.suggest, max_evals=run_value, trials=trials)
    best_trials_temp = trials.best_trial['result'] 
    best_model_temp = train_best_model(best_trials_temp['params']) 
    scaled_feature_df = pd.concat([x_train,x_valid,x_test])
    label_df = pd.concat([y_train,y_valid,y_test])
    pred_df = make_predictions(model=best_model_temp,df=scaled_feature_df)
    output_df = pd.merge(input_df,pred_df['predictions'].to_frame(),left_index=True,right_index=True)
    return best_model_temp, output_df, test_arg_processed, best_trials_temp['params']

# Make Predictions
def make_predictions(model,df):    
    predictions = model.predict(df).flatten()
    df['predictions'] = predictions    
    return df

#Check Authenticity
def check_authenticity(df,train_columns,label_col,test_csv_name):
    test_columns = list(df.columns)
    train_columns = list(train_columns)
    try:
        test_columns.remove(label_col)
    except:
        pass
    try:
        train_columns.remove(label_col)
    except:
        pass
    if(list(test_columns) != list(train_columns)):
        print("The columns in the {} test csv do not match with the given model configuration. \nExiting the script...".format(test_csv_name))
        flush_memory()

# Displaying Result
def display_results(model,output_df,label_col,test_arg_processed):    
    print("#####################################################")
    print("Results on the training data:")
    print("Training Size: {} rows".format(x_train.shape[0]))
    print("Testing Size: {} rows".format(x_test.shape[0]))
    loss,mse = model.evaluate(x_test,y_test,verbose=0)
    rmse = loss**0.5
    pct_error = (rmse / y_test.mean())*100
    print("RMSE on the test data: ",rmse)
    print("Percent error on the test data: ", pct_error, "%")
    print("#####################################################")
    for test_df in test_arg_processed:
        loss_temp,mse_temp = model.evaluate(test_df.loc[:, test_df.columns != label_col],test_df[label_col],verbose=0)
        rmse_temp = loss**0.5
        pct_error_temp = (rmse_temp / test_df[label_col].mean())*100
        print("RMSE on the given test csv: ",rmse_temp)
        print("Percent error on the given test csv: ", pct_error_temp, "%")
        print("#####################################################")
    return rmse, pct_error

# Save the model and update the config db
def update_config_db(csv_name,label_col,selected_cols,rmse,pct_error,model_type,source):
    try:
        deleteQuery(csv_name)
    except:
        pass
    values = (csv_name,model_type,str(selected_cols).replace("'",'"'),label_col,rmse,pct_error,source,unique_id_column)
    insertQuery(values)
    
def revert(x):
    if(x not in list(train_columns)):
        if("_" in x):
            return(x.split("_")[0] if len(x.split("_"))==2 else "_".join(x.split("_")[:-1]))
    else:
        return x

def check_duplicacy(csv_name):
    rowcount = getRowCount(csv_name)
    if(rowcount != 0):
        print("Model {} already exists.".format(csv_name))
        while(True):
            continue_or_not = input("Do you still want to train a new model with this data set? You can replace the existing model with the same name once training has completed. Action: y/n - ")
            continue_or_not = continue_or_not.strip(" ")
            try:
                if(continue_or_not not in ['Y','y','N','n']):
                    raise ValueError("Please enter a valid input.")
                else:
                    break
            except Exception as e:
                print(e)
                continue
        if(continue_or_not in ['N','n']):
            print("Exiting the script...")
            flush_memory()
        return 1
    else:
        return 0
    
def save(csv_name,label_col,model,output_df,selected_cols,rmse, pct_error,model_type,whether_duplicate):
    csv_name_old = csv_name
    if(whether_duplicate == 1):
        while(True):
            save_model = input("A model with this name already exists, do you want to re-name this model or replace? Please enter 1 to rename or 2 to replace.")
            save_model = save_model.strip(" ")
            try:
                if(save_model not in ['1','2']):
                    raise ValueError("Please enter a valid input.")
                else:
                    break
            except Exception as e:
                print(e)
                continue
        if(save_model == '1'):
            csv_name = input("Please name this model: ")
    print("Saving your model...")    
    model.save("models/{}.h5".format(csv_name))
    output_df.to_csv("training_predictions/{}_predictions.csv".format(csv_name))
    
    save_feature_list = list(set(pd.Series(list(selected_cols)).apply(lambda x: revert(x))))
    
    import pickle
    with open('models/{}_columns.pkl'.format(csv_name), 'wb') as f:
        pickle.dump(train_columns, f)
    with open('models/{}_dummy_columns.pkl'.format(csv_name), 'wb') as f:
        pickle.dump(training_dummy_columns, f)
    with open('models/{}_selected_columns.pkl'.format(csv_name), 'wb') as f:
        pickle.dump(selected_cols, f)

    source = input("Enter Data source: ")
    update_config_db(csv_name=csv_name,label_col=label_col,selected_cols=save_feature_list,rmse=rmse,pct_error=pct_error,model_type=model_type,source=source)
    print("Your model, {} has been saved.".format(csv_name))

    transfer_results(csv_name_old,csv_name)

def transfer_results(csv_name_old,csv_name):
    user_name,ip,root_folder_path = get_credentials()
    os.system("scp datasets/{}* {}@{}:{}/datasets".format(csv_name_old,user_name,ip,root_folder_path))
    os.system("scp training_predictions/{}* {}@{}:{}/training_predictions".format(csv_name,user_name,ip,root_folder_path))
    os.system("scp models/{}* {}@{}:{}/models".format(csv_name,user_name,ip,root_folder_path))
    
def monitor_ram(threshold):
    print("Checking to see available RAM.")
    mem = psutil.virtual_memory()
    print("Initial RAM available",(mem.available/1024)/1024,"mb")
    if(mem.available > threshold):
        while(mem.available > threshold):
            mem = psutil.virtual_memory()
            if(mem.available <= threshold):
                print('You do not have enough RAM to execute model training, please deploy your Izenda ML Application on a machine with greater than 150MB RAM.')
                print("Exiting the script...")
                flush_memory()
    else:
        flush_memory()        
        
def flush_memory():
    process = psutil.Process(os.getpid())
    print("Session complete")
    os.system("kill -KILL {}".format(process.pid))
    

def start_thread(threshold):
    from threading import Thread
    thread = Thread(target = monitor_ram, args=(threshold,))
    thread.daemon = True
    thread.start()    
    
# Lime functions
def imp_features_lime():
    import lime
    import lime.lime_tabular
    selected_features = training_dummy_columns
    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(x_train), feature_names=selected_features, class_names=[], verbose=False, mode='regression')
    global start_time_lime
    start_time_lime = time.time()
    df1 = get_intensity_dfs(explainer,x_valid)
    features_subset = get_best_cols_lime(df1)
    return features_subset
 
def predict(qc):
    qc = best_model.predict(qc)
    return qc.reshape(qc.shape[0])
    
def get_intensity_dfs(explainer,x_valid):
    x_valid_copy = x_valid.copy()
    x_valid_copy.reset_index(drop=True,inplace=True)
    print('We are calculating the importance of each feature in your training data set, this process will take {} iterations to complete, please be patient...'.format(x_valid_copy.shape[0]))
    for im in range(x_valid_copy.shape[0]):
        print("Iteration {} of {} is in progress...".format(im+1,x_valid_copy.shape[0]))
        exp = explainer.explain_instance(x_valid_copy.loc[im], predict, num_features=x_valid_copy.shape[1])
        name_pos = list(x_valid_copy.columns)
        intansity = [0]*len(name_pos)
        grt = [0]*len(name_pos)
        grt_and_eql = [0]*len(name_pos)
        less = [0]*len(name_pos)
        less_and_eql = [0]*len(name_pos)
        try:
            for i in exp.as_list():
                if i[0].find(' < ') != -1 and i[0].find(' <= ') != -1:
                    grt[name_pos.index(i[0][i[0].find(' < ')+3:i[0].find(' <= ')])] = float(i[0][0:i[0].find(' < ')])
                    less_and_eql[name_pos.index(i[0][i[0].find(' < ')+3:i[0].find(' <= ')])] = float(i[0][i[0].find(' <= ')+4:])
                    intansity[name_pos.index(i[0][i[0].find(' < ')+3:i[0].find(' <= ')])] = i[1]
                elif i[0].find(' <= ') != -1 and i[0].find(' < ') != -1:
                    grt_and_eql[name_pos.index(i[0][i[0].find(' <= ')+4:i[0].find(' < ')])] = float(i[0][0:i[0].find(' <= ')])
                    less[name_pos.index(i[0][i[0].find(' <= ')+4:i[0].find(' < ')])] = float(i[0][i[0].find(' < ')+3:])
                    intansity[name_pos.index(i[0][i[0].find(' <= ')+4:i[0].find(' < ')])] = i[1]
                elif i[0].find(' < ') != -1:
                    less[name_pos.index(i[0][0:i[0].find(' < ')])] = float(i[0][i[0].find(' < ')+3:])
                    intansity[name_pos.index(i[0][0:i[0].find(' < ')])] = i[1]
                elif i[0].find(' <= ') != -1:
                    less_and_eql[name_pos.index(i[0][0:i[0].find(' <= ')])] = float(i[0][i[0].find(' <= ')+4:])
                    intansity[name_pos.index(i[0][0:i[0].find(' <= ')])] = i[1]
                elif i[0].find(' > ') != -1:
                    grt[name_pos.index(i[0][0:i[0].find(' > ')])] = float(i[0][i[0].find(' > ')+3:])
                    intansity[name_pos.index(i[0][0:i[0].find(' > ')])] = i[1]
                elif i[0].find(' >= ') != -1:
                    grt_and_eql[name_pos.index(i[0][0:i[0].find(' >= ')])] = float(i[0][i[0].find(' >= ')+4:])
                    intansity[name_pos.index(i[0][0:i[0].find(' >= ')])] = i[1]
        except:
            pass
        if im == 0:
            intensity_dic = {'feature_name':name_pos, 'intensity0':intansity}
            df_int = pd.DataFrame(intensity_dic)
           
        else:
            df_int['intensity'+str(im)] = intansity
            
        est_time_remaining_lime = ((time.time() - start_time_lime) / (im+1)) * (x_valid_copy.shape[0]-(im+1))
        m1, s1 = divmod(est_time_remaining_lime, 60)
        h1, m1 = divmod(m1, 60)
            
        print("Estimated time to completion:",int(h1),"hours",int(m1),"minutes",int(s1),"seconds")
        if((im+1) != x_valid_copy.shape[0]):
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")
            
    return df_int.T

def train_model_lime(best_params,feature_subset):
    x_train_temp = x_train.copy() 
    y_train_temp = y_train.copy()
    
    x_valid_temp = x_valid.copy() 
    y_valid_temp = y_valid.copy()
    
    x_train_temp = x_train_temp[feature_subset]
    x_valid_temp = x_valid_temp[feature_subset]
    
    model = Sequential()
    model.add(Dense(best_params['units1'], input_shape=(x_train_temp.shape[1],)))
    model.add(Activation(best_params['activation']))
    model.add(Dropout(best_params['dropout1']))
    if(best_params['num_layers'] == 'two_hidden'):
        model.add(Dense(best_params['units2']))
        model.add(Activation(best_params['activation']))
        model.add(Dropout(best_params['dropout2']))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mse', metrics=['mse'],
                  optimizer=best_params['optimizer'])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=best_params['early_stop_rounds'])
    history = History()
    model.fit(x_train_temp, y_train_temp,
              batch_size=best_params['batch_size'],
              epochs=500,
              callbacks=[early_stop, history],
              verbose=0,
              validation_data=(x_valid_temp,y_valid_temp))
    return model

def get_results_per_simulation(best_params,feature_subset):
    model = train_model_lime(best_params,feature_subset)
    x_valid_temp = x_valid.copy() 
    y_valid_temp = y_valid.copy()
    x_valid_temp = x_valid_temp[feature_subset]
    
    loss_v,mse_v = model.evaluate(x_valid_temp,y_valid_temp,verbose=0)
    rmse_v = loss_v ** 0.5
    pct_error_v = (loss_v ** 0.5) / y_valid_temp.mean()*100
    
    return pct_error_v

def simulation_of_features(df,pct_error_val_ori):
    
    df_non_zero = df[df['sum_of_intensities'] != 0]
    df_non_zero_copy = df_non_zero.copy()    
    df_non_zero_copy.sort_values(by=['index'],inplace=True)
    feature_subset_selected = list(df_non_zero_copy['feature_name'])
    feature_subset_unselected = list(df[df['sum_of_intensities'] == 0]['feature_name'])
    pct_error_v = get_results_per_simulation(best_params,feature_subset_selected)
    
    if(pct_error_v > pct_error_val_ori):
        df.sort_values(by=['index'],inplace=True)
        return list(df['feature_name']),[]
    else:

        if(len(df_non_zero)<=100):
            
            for i in range(1,100):
                
                if(i < len(df_non_zero)):
                    
                    pct_error_v_best = pct_error_v
                    good_features_best = feature_subset_selected
                    bad_features_best =  feature_subset_unselected
                    
                    drop_value = i
                    
                    subset_df = df_non_zero.head(len(df_non_zero) - drop_value)
                    subset_df = subset_df.sort_values(by=['index'])
                    
                    feature_subset_selected = list(subset_df['feature_name'])
                    feature_subset_unselected = list(df_non_zero.tail(drop_value)['feature_name'])
                    pct_error_v = get_results_per_simulation(best_params,feature_subset_selected)
                    
                    if(pct_error_v > pct_error_v_best):
                        return good_features_best, list(df[df['sum_of_intensities'] == 0]['feature_name'])+bad_features_best 
                    else:
                        continue
                    

        else:
            for i in range(1,100):
                
                pct_error_v_best = pct_error_v
                good_features_best = feature_subset_selected
                bad_features_best =  feature_subset_unselected
                
                drop_value = int(np.floor((i / 100) * len(df_non_zero)))
                
                subset_df = df_non_zero.head(len(df_non_zero) - drop_value)
                subset_df = subset_df.sort_values(by=['index'])
                
                feature_subset_selected = list(subset_df['feature_name'])
                feature_subset_unselected = list(df_non_zero.tail(drop_value)['feature_name'])
                pct_error_v = get_results_per_simulation(best_params,feature_subset_selected)
                    
                if(pct_error_v > pct_error_v_best):
                    return good_features_best, list(df[df['sum_of_intensities'] == 0]['feature_name'])+bad_features_best 
                else:
                    continue
                

def get_best_cols_lime(intensity_df):
    header = intensity_df.iloc[0]
    intensity_df = intensity_df[1:]
    intensity_df.columns = header
    intensity_df_trans = intensity_df.T
    intensity_df_trans['sum_of_intensities'] = intensity_df_trans.abs().sum(axis=1)
    intensity_df_trans.reset_index(inplace=True)
    intensity_df_trans.reset_index(inplace=True)
    intensity_df_trans.sort_values(by=['sum_of_intensities'],ascending=False,inplace=True)
    
    loss_v,mse_v = best_model.evaluate(x_valid,y_valid,verbose=0)
    rmse_v = loss_v**0.5
    pct_error_v = (rmse_v / y_valid.mean())*100
   
    good_features, bad_features = simulation_of_features(intensity_df_trans,pct_error_v)
    print("Calculations complete. Below is a list of features that have little to no importance on your model: ")
    print(bad_features)
    
    while(True):
        yes_no_lime = input("Do you want to proceed y/n?")
        yes_no_lime = yes_no_lime.strip(" ")
        try:
            if(yes_no_lime not in ['Y','y','N','n']):
                raise ValueError("Please enter a valid input.")
            else:
                break
        except Exception as e:
            print(e)
            continue
    if(yes_no_lime in ['N','n']):
        features_subset = []
        return features_subset
    else:
        no_of_features = len(good_features)
        print('The {} features selected are:\n'.format(no_of_features))
        print(good_features)
        while(True):
            what_to_do = input('Do you wish to proceed with training your model with these features? y/n Enter (1) to exit feature selection.')
            what_to_do = what_to_do.strip(" ")
            try:
                if(what_to_do not in ['Y','y','N','n','1']):
                    raise ValueError("Please enter a valid input.")
                else:
                    break
            except Exception as e:
                print(e)
                continue
        if(what_to_do in ['1','n','N']):
            features_subset = []
        else:
            features_subset = good_features
        return features_subset
    
def driver(csv_name,label_col,test_array,model_type):
    global run_value
    run_value = 2
    threshold = 150 * 1024 * 1024
    start_thread(threshold)
    global num 
    num = 1
    whether_duplicate = check_duplicacy(csv_name)
    global input_training_data
    input_training_data = data_fetch(csv_name)
    if(test_array != [None]):
        test_arg = test_array
        for test_csv in test_array:
            test_dummy = data_fetch_test_array(test_csv)
            check_authenticity(df=test_dummy,train_columns=train_columns,label_col=label_col,test_csv_name=test_csv)
    else:
        test_arg = None
    global best_model
    global best_params
    best_model, output_df, test_arg_processed, best_params = get_best_model(csv_name=csv_name,label_col=label_col,test_arg=test_arg)
    rmse, pct_error = display_results(model=best_model, output_df=output_df, label_col=label_col, test_arg_processed=test_arg_processed)
    pct_error_lime = -1
    while(True):
        feature_selection = input("Would you like to evaluate automatic feature selection? y/n - ")
        feature_selection = feature_selection.strip(" ")
        try:
            if(feature_selection not in ['Y','y','N','n']):
                raise ValueError("Please enter a valid input.")
            else:
                break
        except Exception as e:
            print(e)
            continue
    if(feature_selection in ['Y','y']):
        num = 1
        features_subset = imp_features_lime()
        if(len(features_subset)!=0):
            best_model_lime, output_df_lime, test_arg_processed_lime, best_params_lime = get_best_model(csv_name=csv_name,label_col=label_col,test_arg=test_arg,use_lime=1,features_subset=features_subset)
            rmse_lime, pct_error_lime = display_results(model=best_model_lime, output_df=output_df_lime,label_col=label_col,test_arg_processed=test_arg_processed_lime)
        else:
            pct_error_lime = -1
    elif(feature_selection in ['N','n']):
        pct_error_lime = -1
        
    if(pct_error_lime == -1):
        print("Following are the results of the model trained without dropping any feaures: ")
        print("#####################################################")
        print("Training Size: {} rows".format(x_train.shape[0]))
        print("Testing Size: {} rows".format(x_test.shape[0]))
        print("Percent error on the test data: ", pct_error, "%")
        print("#####################################################")
        while(True):
            save_model = input("Do you want to save this model? y/n - ")
            save_model = save_model.strip(" ")
            try:
                if(save_model not in ['Y','y','N','n']):
                    raise ValueError("Please enter a valid input.")
                else:
                    break
            except Exception as e:
                print(e)
                continue
        if(save_model in ['Y','y']):
            save(csv_name=csv_name,label_col=label_col,model=best_model, output_df=output_df, selected_cols = training_dummy_columns, rmse=rmse, pct_error=pct_error,model_type=model_type,whether_duplicate=whether_duplicate)
            flush_memory()
        else:
            print("Exiting the script...")
            flush_memory()
            
    else:
        print("Here are the results of your models, 1) is a model without dropping any features and 2) is the model optimized based on dropping features deemed to have little or no importance.")
        print("#####################################################")
        print("Model 1")
        print("Training Size: {} rows".format(x_train.shape[0]))
        print("Testing Size: {} rows".format(x_test.shape[0]))
        print("Percent error on the test data: ", pct_error, "%")
        print("#####################################################")
        print("Model 2")
        print("Training Size: {} rows".format(x_train.shape[0]))
        print("Testing Size: {} rows".format(x_test.shape[0]))
        print("Percent error on the test data: ", pct_error_lime, "%")
        print("#####################################################")
        while(True):
            save_model = input("Select which model you wish to save, please enter 1 or 2. Enter 0 to exit training.")
            save_model = save_model.strip(" ")
            try:
                if(save_model not in ['1','2','0']):
                    raise ValueError("Please enter a valid input.")
                else:
                    break
            except Exception as e:
                print(e)
                continue
        if(save_model == '1'):
            save(csv_name=csv_name,label_col=label_col,model=best_model, output_df=output_df, selected_cols = training_dummy_columns, rmse=rmse, pct_error=pct_error,model_type=model_type,whether_duplicate=whether_duplicate)
            flush_memory()
        elif(save_model == '2'):
            save(csv_name=csv_name,label_col=label_col,model=best_model_lime, output_df=output_df_lime, selected_cols = features_subset, rmse=rmse_lime, pct_error=pct_error_lime,model_type=model_type,whether_duplicate=whether_duplicate)
            flush_memory()
        else:
            print("Exiting the script...")
            flush_memory()

def process(lst):
    print('Izenda machine learning application has been activated')
    try:
        if(len(lst) == 4):
            raise Exception("Test data set not provided. No performance metrics will be provided on test data. \n Test csv : None")
    except Exception as e:
        print(e)
        test_array = None
    if(len(lst) == 5):
        test_array = lst[4]
        print("Test data set provided. \n Test csv : {}".format(test_array))
    csv_name = lst[1]
    label_col = lst[2]
    model_type = lst[3]
    test_array = [test_array]
    try:
        driver(csv_name=csv_name,label_col=label_col,test_array=test_array,model_type=model_type)
    except MemoryError as e:
        print(e)
