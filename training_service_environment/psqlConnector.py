#!/usr/bin/python
from configparser import ConfigParser
import os
import psycopg2
import logging
#from scripts.BaseError import NotFoundError, ValidationError, handle_error, default_error_handler,ServerError,ConnectionError

connection = None

def connect_database(hostname, username, password, database):
    global connection
    if connection == None:
        try:
            connection = psycopg2.connect( host=hostname, user=username, password=password, dbname=database )
        except:
            connection = None   
    return connection

def init_database(hostname, username, password, database):
    global connection
    conn = connect_database(hostname, username, password, database)
    if conn == None:
        print ("error connecting..")
    else:
        cur = conn.cursor()
        try:
            table = getTableName()
            cur.execute("CREATE TABLE IF NOT EXISTS {}(id serial PRIMARY KEY,model_name VARCHAR UNIQUE NOT NULL,model_type VARCHAR NOT NULL,list_of_features TEXT NOT NULL,target_column VARCHAR NOT NULL,rmse FLOAT,percent_error FLOAT,accuracy FLOAT,prec FLOAT,source VARCHAR NOT NULL,unique_id_column VARCHAR NOT NULL);".format(table))
        except:
            print("error creating the table")
        conn.commit()
        conn.close()
        cur.close()
    connection = None 

def setPsqlParams():
    pgsqlParser = ConfigParser()
    configFile = (os.path.join(os.getcwd(),'config','pgsql_config.ini'))
    pgsqlParser.read(configFile)
    hostname = pgsqlParser.get('psql', 'host')
    username = pgsqlParser.get('psql', 'user')
    password = pgsqlParser.get('psql', 'passwd')
    database = pgsqlParser.get('psql', 'database')
    table = pgsqlParser.get('psql', 'table')
    return hostname,username,password,database,table


def getPsqlParams():
    pgsqlParser = ConfigParser()
    configFile = (os.path.join(os.getcwd(),'config','pgsql_config.ini'))
    pgsqlParser.read(configFile)
    hostname = pgsqlParser.get('psql', 'host')
    username = pgsqlParser.get('psql', 'user')
    password = pgsqlParser.get('psql', 'passwd')
    database = pgsqlParser.get('psql', 'database')
    table = pgsqlParser.get('psql', 'table')
    # hostname, username, password, database, table = setPsqlParams()
    hostnameString = bytes(hostname, "utf-8").decode("unicode_escape")
    return hostnameString,username,password,database

def getTableName():
    hostname,username,password,database,table = setPsqlParams()
    return table

def doQuery(command) :
    global connection
    hostname,username,password,database,table = setPsqlParams()
    init_database(hostname, username, password, database)
    myconnection = connect_database(hostname, username, password, database)
    cur = myconnection.cursor()
    cur.execute(command)
    if("SELECT" in command):
        result = cur.fetchall()
        myconnection.close()
        cur.close()
        connection = None
        return result
    elif(("INSERT" in command) or ("DELETE" in command)):
        # cur.commit()
        myconnection.commit()
        myconnection.close()
        cur.close()
        connection = None

def getRowCount(value):
    table = getTableName()
    select_query = "SELECT model_name FROM {} WHERE model_name = '{}';".format(table,value) 
    result = doQuery(select_query)
    return len(result)

def insertQuery(values):
    model_type = values[1]
    table = getTableName()
    if(model_type == 'predict'):
        insert_query = "INSERT INTO {} (model_name, model_type,list_of_features,target_column,RMSE,Percent_Error,source,unique_id_column) VALUES ('{}','{}','{}','{}',{},{},'{}','{}')".format(table,*values)
    elif(model_type == 'classify'):
        insert_query = "INSERT INTO {} (model_name, model_type,list_of_features,target_column,accuracy,source,unique_id_column) VALUES ('{}','{}','{}','{}',{},'{}','{}')".format(table,*values)
    doQuery(insert_query)

def deleteQuery(value) :
    table = getTableName()
    delete_query = "DELETE FROM {} where model_name = '{}'".format(table,value)
    doQuery(delete_query)
