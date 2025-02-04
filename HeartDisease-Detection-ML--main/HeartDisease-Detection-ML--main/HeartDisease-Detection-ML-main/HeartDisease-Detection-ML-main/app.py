from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load('heart_disease_model.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        cp = int(request.form['cp'])
        thalach = int(request.form['thalach'])
        user_data = pd.DataFrame([[age, cp, thalach]], columns=['age', 'cp', 'thalach'])
        prediction = model.predict(user_data)
        result = "Heart Disease Present" if prediction[0] == 1 else "No Heart Disease"
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
