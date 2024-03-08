from flask import Flask ,jsonify,request
import pickle
import pandas as pd
import joblib

app =Flask(__name__)


# Load the Random Forest model
model = joblib.load('RandomForest.pkl')

print()


@app.route('/predict', methods=['POST'])
def home():
    data = request.get_json()

    N = data.get('n')
    P = data.get('p')
    k = data.get('k')
    temperature = data.get('t')
    humidity=data.get('h')
    pH=data.get('ph')
    rainfall=data.get("r")
    crop=model.predict([[N,P,k,temperature,humidity,pH,rainfall]])[0]
    return jsonify({'crop':crop})

if __name__=='__main__':
    app.run(port=3000,debug=True)
