from flask import Flask,request,jsonify,render_template
from make_predictions import driver

app = Flask(__name__)
@app.route('/', methods=['POST'])
def run_app():
    payload = request.form
    csv_name = payload['csv_name']
    model_name = payload['model_name']
    result = driver(csv_name,model_name)
    return jsonify(result)

if __name__ == '__main__':
     app.run(host='0.0.0.0',port=8000)
