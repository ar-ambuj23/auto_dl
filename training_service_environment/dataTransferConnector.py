from configparser import ConfigParser
import os

def get_credentials():
    dataParser = ConfigParser()
    configFile = (os.path.join(os.getcwd(),'config','pgsql_config.ini'))
    dataParser.read(configFile)
    user_name = dataParser.get('data_transfer', 'user_name')
    ip = dataParser.get('data_transfer', 'ip')
    root_folder_path = dataParser.get('data_transfer', 'root_folder_path')
    return user_name,ip,root_folder_path
   