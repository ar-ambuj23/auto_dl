import sys
from regression_ml import flush_memory

try:
    if(len(sys.argv) > 5):
        raise Exception("Length of arguments do not match... \nPlease enter in the following format: csv_name label_col model_type test_csv \nExiting the script...")
except Exception as e:
    print(e)
    flush_memory()
    
try:
    if(len(sys.argv) < 4):
        raise Exception("Length of arguments do not match... \nPlease enter in the following format: csv_name label_col model_type test_csv \nExiting the script...")
except Exception as e:
    print(e)
    flush_memory()
        
if(sys.argv[3] == 'predict'):
    import regression_ml
    regression_ml.process(sys.argv)
elif(sys.argv[3] == 'classify'):
    import classification_ml
    classification_ml.process(sys.argv)
elif(sys.argv[3] == 'forecast'):
    print("Time Series Model is under construction for now.\nExiting the script...")
    flush_memory()
else:
    print("Model type can be only be predict, classify or forecast.\nExiting the script...")
    flush_memory()
