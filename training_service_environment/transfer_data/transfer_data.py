from flask import Flask,request,jsonify,render_template
from werkzeug.utils import secure_filename
import os

#ALLOWED_EXTENSIONS = set(['csv'])
UPLOAD_PATH = '/home/algoscale/Documents/Izenda/izenda_ml/training_service_environment/transfer_data'

app = Flask(__name__)
app.config['UPLOAD_PATH'] = UPLOAD_PATH

@app.route('/upload', methods=['POST'])
def upload_file():
    result = {}
    keys_list = list(request.files.to_dict(flat=False).keys())
    for key in keys_list:
        print(key)
        file = request.files[key]
        fname = file.filename
        fname = secure_filename(fname)

        print(os.path.join(app.config['UPLOAD_PATH'], 'a/'+fname))
        if not fname.endswith(('.csv')):
            result[fname] = {'error': 'bad file format for {}. Only .csv files accepted'.format(fname)}
        
        else:
            if("training" in str(key)):
                file.save(os.path.join(app.config['UPLOAD_PATH'], 'a/'+fname)) ##change folder name to datasets
                result[fname] = {'success':'File {} has been uploaded successfully.'.format(fname)}
            elif("testing" in str(key)):
                print(os.path.join(app.config['UPLOAD_PATH'], 'b/'+fname))
                file.save(os.path.join(app.config['UPLOAD_PATH'], 'b/'+fname)) ##change folder name to testing_input
                result[fname] = {'success':'File {} has been uploaded successfully.'.format(fname)}
            else:
                result[fname] = {'error':'key name for file {} is incorrect.'.format(fname)}
      
    return jsonify(result)

    
if __name__ == '__main__':
     app.run(host='localhost',port=8888)
