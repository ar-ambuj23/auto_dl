import numpy as np
import pandas as pd
import sys
from keras.models import load_model
from psqlConnector import doQuery

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def data_fetch(csv_name):
    try:
        import subprocess
        sub_output = subprocess.check_output('file -b --mime-encoding test_input/{}.csv'.format(csv_name), shell=True)
        encoding_csv = sub_output.decode('utf-8')
        df = pd.read_csv('test_input/{}.csv'.format(csv_name),index_col=0, encoding = encoding_csv)    
        if(len(df.index)!=df.shape[0]):
            raise Exception('There are duplicate values in the index column. \nExiting the script...')
        return df
    except UnicodeError as e: 
        raise Exception("The {} csv has wrong encoding. Please use a csv with utf-8 encoding. \nExiting the script...".format(csv_name))
    except:
        raise Exception('CSV Name does not exist in the test_input folder')

def get_label_col(model_name):
    try:
        get_label_query = "SELECT target_column FROM config_table WHERE model_name = '{}';".format(model_name)
        label_col = doQuery(get_label_query)
        return label_col[0][0]
    except:
        raise Exception('Model name does not exist in the config database')
        
def get_pickle_files(model_name):
    import pickle
    with open('models/{}_columns.pkl'.format(model_name), 'rb') as f:
        train_columns = pickle.load(f)
    with open('models/{}_dummy_columns.pkl'.format(model_name).format(model_name), 'rb') as f:
        training_dummy_columns = pickle.load(f)
    with open('models/{}_selected_columns.pkl'.format(model_name), 'rb') as f:
        selected_cols = pickle.load(f)
    return train_columns,training_dummy_columns,selected_cols

def check_authenticity(df,train_columns,label_col):
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
        raise Exception("The columns in the test csv do not match with the given model configuration")
    

def get_string_cols(df):    
    string_cols = list(df.select_dtypes(include=['object','category']).columns)
    return string_cols

def get_num_cols(df):    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num_cols = list(df.select_dtypes(include=numerics).columns)    
    return num_cols

def pre_process(df,label_col,training_dummy_columns):
    df_copy = df.copy()
    try:
        df_copy = df_copy.drop(label_col,axis=1)
    except:
        pass
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
    df_copy = pd.get_dummies(df_copy,columns=categorical_cols)
    missing_cols = set( training_dummy_columns ) - set( df_copy.columns )
    for col in missing_cols:
        df_copy.loc[:,col] = 0
    df_copy = df_copy[training_dummy_columns]
    df_copy.columns = df_copy.columns.str.replace(':string','')
    return df_copy

def select_features(processed_df,selected_cols):
    processed_df = processed_df[selected_cols]
    return processed_df

def get_predictions(original_df,feature_selected_df,model_name,label_col):
    from keras.models import load_model
    model = load_model('models/{}.h5'.format(model_name))
    predictions = model.predict(feature_selected_df).flatten()
    final_df = original_df.copy()
    final_df['{}_predictions'.format(label_col)] = predictions
    return final_df

def save_predictions(csv_name,final_df):
    path = 'test_output/{}_output.csv'.format(csv_name)
    final_df.to_csv(path,index=True)
    return path 

def driver(csv_name,model_name):
    try:
        df = data_fetch(csv_name=csv_name)
        label_col = get_label_col(model_name=model_name)
        train_columns,training_dummy_columns,selected_cols = get_pickle_files(model_name=model_name)
        check_authenticity(df=df,train_columns=train_columns,label_col=label_col)
        processed_df = pre_process(df=df,label_col=label_col,training_dummy_columns=training_dummy_columns)
        feature_selected_df = select_features(processed_df=processed_df,selected_cols=selected_cols)
        final_df = get_predictions(original_df=df,feature_selected_df=feature_selected_df,model_name=model_name,label_col=label_col)
        path = save_predictions(csv_name=csv_name,final_df=final_df)
        import keras.backend as K
        K.clear_session()
        print("Result:","The output csv has been saved at {}".format(path))
    except Exception as e:
        print("Error:", str(e))

driver('','')    