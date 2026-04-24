from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "Iris ML API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    
    prediction = model.predict([data])
    
    return jsonify({
        'prediction': prediction[0]
    })

if __name__ == '__main__':
    app.run(debug=True)