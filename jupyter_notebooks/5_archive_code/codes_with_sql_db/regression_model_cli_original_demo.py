# coding: utf-8

import numpy as np
import pandas as pd
import sys
import psutil

from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, History 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Data fetch
def data_fetch(csv_name):
    try:
        import subprocess
        sub_output = subprocess.check_output('file -b --mime-encoding datasets/{}.csv'.format(csv_name), shell=True)
        encoding_csv = sub_output.decode('utf-8')
        df = pd.read_csv('datasets/{}.csv'.format(csv_name),index_col=0, encoding = encoding_csv)
        
        if(len(df.index)!=df.shape[0]):
            print('There are duplicate values in the index column. \nExiting the script...')
            flush_memory()
        
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
        df = pd.read_csv('datasets/{}.csv'.format(csv_name),index_col=0,encoding = encoding_csv)
                  
        if(len(df.index)!=df.shape[0]):
            print('There are duplicate values in the index column of the test csv. \nExiting the script...')
            flush_memory()
                  
        return df
    except UnicodeError as e: 
        print("The {} csv has wrong encoding. Please use a csv with utf-8 encoding. \nExiting the script...".format(csv_name))
        flush_memory()
        
    except Exception as e:
        print("The {} csv is not present in the datasets folder. \nExiting the script...".format(csv_name))
        flush_memory()

        
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
        
    print("Pre-Processing the data...")
    
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
    print('Size of df before making dummies: ', df.memory_usage(deep=True).sum()/1024/1024, 'mb') 
    print('Converting categorical columns to dummies...') 
    df = pd.get_dummies(df,columns=categorical_cols,sparse=True)
    print('Size of df after making dummies: ', df.memory_usage(deep=True).sum()/1024/1024, 'mb')
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
    print('Size of df before making dummies: ', df_copy.memory_usage(deep=True).sum()/1024/1024, 'mb') 
    print('Converting categorical columns to dummies...')
    df_copy = pd.get_dummies(df_copy,columns=categorical_cols,sparse=True)
    print('Size of df after making dummies: ', df_copy.memory_usage(deep=True).sum()/1024/1024, 'mb')
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
    print("Reading the data...")
    data = data_fetch(csv_name)
    pre_processed_data = pre_process(df=data,label_col=label_col)
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
        return data, x_train_lime, x_test_lime, x_valid_lime, y_train_lime, y_test_lime, y_valid_lime, test_arg_processed_lime
    return data, x_train, x_test, x_valid, y_train, y_test, y_valid, test_arg_processed

def create_model(params):    
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
    global num
    mem = psutil.virtual_memory()
    if(np.isnan(mse)):
        print("{}) Validation set root mean sq. error: NaN".format(num),"\tAvailable Mem:",(mem.available/1024)/1024,"mb")
        num = num + 1
        return {'loss': np.inf, 'status': STATUS_OK, 'params': params}
    print("{}) Validation set root mean sq. error: {:7.2f}".format(num,mse**0.5),"\tAvailable Mem:",(mem.available/1024)/1024,"mb")
    num = num + 1
    return {'loss': loss**0.5, 'status': STATUS_OK, 'params': params}

def train_best_model(best_params):   
    print('Training the best selected model...') 
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
    print("Selecting the best network architecture specifically for your data...")
    best = fmin(create_model, space, algo=tpe.suggest, max_evals=2, trials=trials)
    best_trials_temp = trials.best_trial['result'] 
    best_model_temp = train_best_model(best_trials_temp['params']) 
    scaled_feature_df = pd.concat([x_train,x_valid,x_test])
    label_df = pd.concat([y_train,y_valid,y_test])
    pred_df = make_predictions(model=best_model_temp,df=scaled_feature_df)
    output_df = pd.merge(input_df,pred_df['predictions'].to_frame(),left_index=True,right_index=True)
    return best_model_temp, output_df, test_arg_processed

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
def update_config_db(csv_name,label_col,selected_cols,rmse,pct_error,model_type):
    import mysql.connector
    class NumpyMySQLConverter(mysql.connector.conversion.MySQLConverter):
        """ A mysql.connector Converter that handles Numpy types """
        def _float32_to_mysql(self, value):
            return float(value)
        def _float64_to_mysql(self, value):
            return float(value)
        def _int32_to_mysql(self, value):
            return int(value)
        def _int64_to_mysql(self, value):
            return int(value)
    mydb = mysql.connector.connect(
    host="db", 
    user="root",
    passwd="root", 
    database="configdb")
    mydb.set_converter_class(NumpyMySQLConverter)
    mycursor = mydb.cursor()
    try:
        delete_query = "DELETE FROM config_table where model_name = '{}';".format(csv_name)
        mycursor.execute(delete_query)
        mydb.commit()
    except:
        pass
    insert_query = "INSERT INTO config_table (model_name, model_type,list_of_features,target_column,RMSE,Percent_Error) VALUES (%s, %s, %s, %s, %s, %s)"
    values = (csv_name,model_type,str(selected_cols),label_col,rmse,pct_error)
    mycursor.execute(insert_query, values)
    mydb.commit()
    print(mycursor.rowcount, "record inserted for", csv_name, "in the config database.")

def check_duplicacy(csv_name):
    import mysql.connector
    mydb = mysql.connector.connect(
    host="db", 
    user="root",
    passwd="root", 
    database="configdb")
    mycursor = mydb.cursor()
    duplicate_query = "SELECT model_name FROM config_table WHERE model_name = '{}';".format(csv_name)
    mycursor.execute(duplicate_query)
    myresult = mycursor.fetchall()
    if(mycursor.rowcount != 0):
        print("Model for that csv already exists.")
        while(True):
            continue_or_not = input("Do you still want to continue? y/n - ")
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
    
def save(csv_name,label_col,model,output_df,selected_cols,rmse, pct_error,model_type):
    print("Saving the model...")    
    model.save("models/{}.h5".format(csv_name))
    print("Saving the output predictions...")
    output_df.to_csv("training_predictions/{}_predictions.csv".format(csv_name))
    print("Pickling necessary data...")
    import pickle
    with open('models/{}_columns.pkl'.format(csv_name), 'wb') as f:
        pickle.dump(train_columns, f)
    with open('models/{}_dummy_columns.pkl'.format(csv_name), 'wb') as f:
        pickle.dump(training_dummy_columns, f)
    with open('models/{}_selected_columns.pkl'.format(csv_name), 'wb') as f:
        pickle.dump(selected_cols, f)
    print("Updating the config database...")
    update_config_db(csv_name=csv_name,label_col=label_col,selected_cols=selected_cols,rmse=rmse,pct_error=pct_error,model_type=model_type)
    print("Model and the config information has been saved.")
    
def monitor_ram(threshold):
    print("Monitoring RAM...")
    mem = psutil.virtual_memory()
    print("Initial RAM available",(mem.available/1024)/1024,"mb")
    if(mem.available > threshold):
        while(mem.available > threshold):
            mem = psutil.virtual_memory()
            if(mem.available <= threshold):
                print("Overflow...")
                print("RAM is full. Please upgrade your machine.")
                print("Exiting the script...")
                flush_memory()
    else:
        flush_memory()        
        
def flush_memory():
    process = psutil.Process(os.getpid())
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
    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(x_train), feature_names=selected_features, class_names=[], verbose=True, mode='regression')
    df1 = get_intensity_dfs(explainer,x_valid)
    features_subset = get_best_cols_lime(df1)
    return features_subset
 
def predict(qc):
    qc = best_model.predict(qc)
    return qc.reshape(qc.shape[0])
    
def get_intensity_dfs(explainer,x_valid):
    print('Generating intensity values for training data...')
    x_valid_copy = x_valid.copy()
    x_valid_copy.reset_index(drop=True,inplace=True)
    print('The LIME iterations will run {} times...'.format(x_valid_copy.shape[0]))
    for im in range(x_valid_copy.shape[0]):
        print('-'*25,im+1,'-'*25)
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
            
    return df_int.T

def get_best_cols_lime(intensity_df):
    header = intensity_df.iloc[0]
    intensity_df = intensity_df[1:]
    intensity_df.columns = header
    intensity_df_trans = intensity_df.T
    intensity_df_trans['sum_of_intensities'] = intensity_df_trans.abs().sum(axis=1)
    intensity_df_trans.sort_values(by=['sum_of_intensities'],ascending=False,inplace=True)
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    intensity_df_trans['sum_of_intensities'] = min_max_scaler.fit_transform(intensity_df_trans['sum_of_intensities'].values.reshape(-1,1)) * 100
    features_subset = []
    valid_inputs = []
    for x in range(1,101):
        valid_inputs.append(x)
    valid_inputs.append(-1)
    while(True):
            thresh_lime = input("Enter the threshold to select features between 1 to 100. Enter -1 to exit feature selection. - ")
            thresh_lime = int(thresh_lime)
            try:
                if(thresh_lime not in valid_inputs):
                    raise ValueError("Please enter a valid input.")
                else:
                    if(thresh_lime == -1):
                        features_subset = []
                        break
                    else:
                        features_subset = list(intensity_df_trans[intensity_df_trans['sum_of_intensities'] >= thresh_lime].index)
                        no_of_features = len(features_subset)
                        print('The {} features selected are:\n'.format(no_of_features))
                        print(features_subset)
                        while(True):
                            what_to_do = input('Do you want to train with selected features? - y/n Enter -1 to exit feature selection. - ')
                            what_to_do = what_to_do.strip(" ")
                            try:
                                if(what_to_do not in ['Y','y','N','n','-1']):
                                    raise ValueError("Please enter a valid input.")
                                else:
                                    break
                            except Exception as e:
                                print(e)
                                continue
                        if(what_to_do == '-1'):
                            features_subset = []
                            break
                        elif(what_to_do in ['n','N']):
                            continue
                        else:
                            return features_subset
            except Exception as e:
                print(e)
                continue
    return features_subset
    
def driver(csv_name,label_col,test_array,model_type):
    threshold = 150 * 1024 * 1024
    start_thread(threshold)
    global num 
    num = 1
    check_duplicacy(csv_name)
    data = data_fetch(csv_name)
    if(test_array != [None]):
        test_arg = test_array
        for test_csv in test_array:
            test_dummy = data_fetch_test_array(test_csv)
            check_authenticity(df=test_dummy,train_columns=train_columns,label_col=label_col,test_csv_name=test_csv)
    else:
        test_arg = None
    global best_model
    best_model, output_df, test_arg_processed = get_best_model(csv_name=csv_name,label_col=label_col,test_arg=test_arg)
    rmse, pct_error = display_results(model=best_model, output_df=output_df, label_col=label_col, test_arg_processed=test_arg_processed)
    pct_error_lime = -1
    while(True):
        feature_selection = input("Do you want to select features using LIME? y/n - ")
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
            best_model_lime, output_df_lime, test_arg_processed_lime = get_best_model(csv_name=csv_name,label_col=label_col,test_arg=test_arg,use_lime=1,features_subset=features_subset)
            rmse_lime, pct_error_lime = display_results(model=best_model_lime, output_df=output_df_lime,label_col=label_col,test_arg_processed=test_arg_processed_lime)
        else:
            pct_error_lime = -1
    elif(feature_selection in ['N','n']):
        pct_error_lime = -1
    while(True):
        save_model = input("Do you want to save the best model? y/n - ")
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
        if(pct_error_lime != -1):
            if(pct_error<pct_error_lime):
                save(csv_name=csv_name,label_col=label_col,model=best_model, output_df=output_df, selected_cols = list(training_dummy_columns), rmse=rmse, pct_error=pct_error,model_type=model_type)
                flush_memory()
            else:
                save(csv_name=csv_name,label_col=label_col,model=best_model_lime, output_df=output_df_lime, selected_cols = features_subset, rmse=rmse_lime, pct_error=pct_error_lime,model_type=model_type)
                flush_memory()
        else:
            save(csv_name=csv_name,label_col=label_col,model=best_model, output_df=output_df, selected_cols = list(training_dummy_columns), rmse=rmse, pct_error=pct_error,model_type=model_type)
            flush_memory()
    else:
        print("Exiting the script...")
        flush_memory()

def process(lst):
    try:
        if(len(lst) == 4):
            raise Exception("Test csv not provided. Using default value. \n Test csv : None")
    except Exception as e:
        print(e)
        test_array = None
    if(len(lst) == 5):
        test_array = lst[4]
    csv_name = lst[1]
    label_col = lst[2]
    model_type = lst[3]
    test_array = [test_array]
    try:
        driver(csv_name=csv_name,label_col=label_col,test_array=test_array,model_type=model_type)
    except MemoryError as e:
        print(e)
