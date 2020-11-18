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
    print("Reading the data...")
    try:
        df = pd.read_csv('datasets/{}.csv'.format(csv_name))
        global train_columns
        train_columns = df.columns
        return df
    except Exception as e:
        print("The csv is not present in the datasets folder. \nExiting the script...")
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
    print("Pre-Processing the data...")
    y = df[label_col]
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
    df = pd.get_dummies(df,columns=categorical_cols)
    global training_dummy_columns
    training_dummy_columns = df.columns
    df.columns = df.columns.str.replace(':string','')
    df[label_col] = y  
    return df

# Train Test Split
def split_train_test(df,label_col,test_size=0.2):    
    X = df.loc[:,df.columns != label_col]
    y = df[label_col]
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=test_size, random_state=42)
    return train_x, test_x, train_y, test_y

# Feature Selection
def rfe_feat_selection(train_x,train_y,label_col,num_features):    
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestRegressor
    print("Selecting the best features for training...")
    step = int(np.ceil(train_x.shape[1] / 100))
    estimator = RandomForestRegressor(warm_start=True, random_state=42)
    selector = RFE(estimator,step=step,n_features_to_select=num_features,verbose=0) 
    selector = selector.fit(train_x, train_y)
    print('No of selected features:',selector.n_features_)
    global selected_cols
    selected_cols = []
    for val,col in zip(selector.support_,train_x.columns):
        if(val == True):
            selected_cols.append(col)
    return selected_cols

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

def data(csv_name,label_col,num_features):    
    data = data_fetch(csv_name)
    pre_processed_data = pre_process(df=data,label_col=label_col)
    train_x, test_x, train_y, test_y = split_train_test(df=pre_processed_data,label_col=label_col)
    best_features = rfe_feat_selection(train_x,train_y,label_col=label_col,num_features=num_features)    
    best_features_copy = best_features.copy()
    best_features_copy.append(label_col)
    feature_selected_data = pre_processed_data[best_features_copy]
    x_train, x_test, y_train, y_test = split_train_test(df=feature_selected_data,label_col=label_col)
    return data, x_train, x_test, y_train, y_test

def create_model(params):    
    x_train_temp = x_train.copy()
    x_test_temp = x_test.copy()
    y_train_temp = y_train.copy()
    y_test_temp = y_test.copy()
    model = Sequential()
    model.add(Dense(params['units1'], input_shape=(x_train_temp.shape[1],)))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))
    # If we choose 'two_hidden', add an additional layer
    if(params['num_layers'] == 'two_hidden'):
        model.add(Dense(params['units2']))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['dropout2']))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mse', metrics=['mae'],
                  optimizer=params['optimizer'])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=params['early_stop_rounds'])
    history = History()
    model.fit(x_train_temp, y_train_temp,
              batch_size=params['batch_size'],
              epochs=500,
              callbacks=[early_stop, history],
              verbose=0,
              validation_split=0.2)
    [loss, mae] = model.evaluate(x_test_temp, y_test_temp, verbose=0)
    global num
    mem = psutil.virtual_memory()
    if(np.isnan(mae)):
        print("{}) Testing set Mean Abs Error: NaN".format(num),"\tAvailable Mem:",(mem.available/1024)/1024,"mb")
        num = num + 1
        return {'loss': np.inf, 'status': STATUS_OK, 'model': model}
    print("{}) Testing set Mean Abs Error: {:7.2f}".format(num,mae),"\tAvailable Mem:",(mem.available/1024)/1024,"mb")
    num = num + 1
    return {'loss': loss, 'status': STATUS_OK, 'model': model}

def get_best_model(csv_name,label_col,num_features):
    global x_train,x_test,y_train,y_test
    input_df, x_train, x_test, y_train, y_test = data(csv_name=csv_name,label_col=label_col,num_features=num_features)
    trials=Trials()
    space = get_search_space()
    print("Selecting the best network architecture specifically for your data...")
    best = fmin(create_model, space, algo=tpe.suggest, max_evals=25, trials=trials)
    best_model = trials.best_trial['result']['model']
    scaled_feature_df = pd.concat([x_train,x_test])
    label_df = pd.concat([y_train,y_test])
    pred_df = make_predictions(model=best_model,df=scaled_feature_df)
    output_df = pd.merge(input_df,pred_df['predictions'].to_frame(),left_index=True,right_index=True)
    return best_model, output_df

# Make Predictions
def make_predictions(model,df):    
    predictions = model.predict(df).flatten()
    df['predictions'] = predictions    
    return df

# Displaying Result
def display_results(best_model,output_df,label_col):    
    print("#####################################################")
    print("Results:")
    print("Training Size: {} rows".format(x_train.shape[0]))
    print("Testing Size: {} rows".format(x_test.shape[0]))
    # Evaluation on test data
    loss,mae = best_model.evaluate(x_test,y_test,verbose=0)
    rmse = loss**0.5
    pct_error = (rmse / output_df[label_col].mean())*100
    print("RMSE on the test data: ",rmse)
    print("Percent error on the test data: ", pct_error, "%")
    print("#####################################################")
    return rmse, pct_error

# Save the model and update the config db
def update_config_db(csv_name,label_col,rmse,pct_error,model_type):
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
    host="localhost",
    user="root",
    passwd="root",
    database="configdb")
    mydb.set_converter_class(NumpyMySQLConverter)
    mycursor = mydb.cursor()
    insert_query = "INSERT INTO config_table (model_name, model_type,list_of_features,target_column,RMSE,Percent_Error) VALUES (%s, %s, %s, %s, %s, %s)"
    values = (csv_name,model_type,str(selected_cols),label_col,rmse,pct_error)
    mycursor.execute(insert_query, values)
    mydb.commit()
    print(mycursor.rowcount, "record inserted for", csv_name, "in the config database.")

def check_duplicacy(csv_name):
    import mysql.connector
    mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="root",
    database="configdb")
    mycursor = mydb.cursor()
    duplicate_query = "SELECT model_name FROM config_table WHERE model_name = '{}';".format(csv_name)
    mycursor.execute(duplicate_query)
    myresult = mycursor.fetchall()
    if(mycursor.rowcount != 0):
        print("Model for that csv already exists. Please change the csv name.")
        print("Exiting the script...")
        flush_memory()
    
def save(csv_name,label_col,best_model,output_df,rmse, pct_error,model_type):
    print("Saving the model...")    
    best_model.save("models/{}.h5".format(csv_name))
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
    update_config_db(csv_name=csv_name,label_col=label_col,rmse=rmse,pct_error=pct_error,model_type=model_type)
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
                print("RAM is full. Please upgrade your EC2 instance.")
                print("Exiting the script...")
                flush_memory()
                break
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

# Driver function
def driver(csv_name,label_col,num_features,model_type):
    threshold = 150 * 1024 * 1024  # 150MB
    start_thread(threshold)
    global num 
    num = 1
#     check_duplicacy(csv_name)
    best_model, output_df = get_best_model(csv_name=csv_name,label_col=label_col,num_features=num_features)
    rmse, pct_error = display_results(best_model=best_model, output_df=output_df,label_col=label_col)
    while(True):
        save_model = input("Do you want to save the model? y/n - ")
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
        save(csv_name=csv_name,label_col=label_col,best_model=best_model, output_df=output_df,rmse=rmse, pct_error=pct_error,model_type=model_type)
    else:
        print("Exiting the script...")
        flush_memory()

# Main function
if __name__ == '__main__':
    
    try:
        if(len(sys.argv) > 5):
            raise Exception("Length of arguments do not match. Please enter in the follwing format: csv_name label_col num_of_features model_type. \nExiting the script...")
    except Exception as e:
        print(e)
        flush_memory()
    
    try:
        if(len(sys.argv) == 3):
            raise Exception("Num_of_features and Model_type not provided. Using default values. \n Num_of_features : 10 \n Model_type : predict")
    except Exception as e:
        print(e)
        num_features = 10
        model_type = 'predict'
        
    try:
        if(len(sys.argv) == 4):
            raise Exception("Model_type not provided. Using default value. \n Model_type : predict")
    except Exception as e:
        print(e)
        num_features = int(sys.argv[3])
        model_type = 'predict'
    
    if(len(sys.argv) == 4):
        try:
            if(int(sys.argv[3]) > 1000):
                raise Exception("Number of features cannot exceed 1000. Please enter lesser number of features. \nExiting the script...")
        except Exception as e:
            print(e)
            flush_memory()
        num_features=int(sys.argv[3])

    if(len(sys.argv) == 5):
        try:
            if(sys.argv[4] not in ['predict']):#,'classify','forecast']):
                raise Exception("Model type can be only be predict.\nExiting the script...")
        except Exception as e:
            print(e)
            flush_memory()
        num_features=int(sys.argv[3])
        model_type = sys.argv[4]
        
    csv_name = sys.argv[1]
    label_col = sys.argv[2]
       
    try:
        driver(csv_name=csv_name,label_col=label_col,num_features=num_features,model_type=model_type)
    except MemoryError as e:
        print(e)
