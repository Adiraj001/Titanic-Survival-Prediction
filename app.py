import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template_string

app = Flask(__name__)

MODEL_FILE_NAME = 'titanic_Survival_Prediction.pkl'
try:
    model = pickle.load(open(MODEL_FILE_NAME, 'rb'))
    MODEL_FEATURES = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S'
    ]
except FileNotFoundError:
    print("="*50)
    print(f"ERROR: '{MODEL_FILE_NAME}' not found.")
    print("Please run the Streamlit app in 'Train New Model' mode to create it,")
    print(f"or rename your existing model to '{MODEL_FILE_NAME}'.")
    print("="*50)
    model = None
except Exception as e:
    print(f"An error occurred loading the model: {e}")
    model = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Titanic Survival Prediction</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f0f2f5;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }
        .container {
            background-color: #ffffff;
            padding: 2rem 3rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            width: 100%;
            max-width: 500px;
        }
        h1 {
            color: #1c294e;
            text-align: center;
            margin-bottom: 1.5rem;
            font-size: 1.8rem;
        }
        form {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.2rem;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-group.full-width {
            grid-column: 1 / -1;
        }
        label {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #333;
        }
        input, select {
            padding: 0.75rem;
            border: 1px solid #dcdcdc;
            border-radius: 8px;
            font-size: 1rem;
            width: 100%;
            box-sizing: border-box;
        }
        button {
            grid-column: 1 / -1;
            padding: 0.8rem;
            font-size: 1.1rem;
            font-weight: 700;
            color: #ffffff;
            background: linear-gradient(90deg, #365899, #1c294e);
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        button:hover {
            opacity: 0.9;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        .prediction-result {
            text-align: center;
            margin-top: 1.5rem;
            font-size: 1.5rem;
            font-weight: 700;
        }
        .result-survived {
            color: #28a745;
        }
        .result-not-survived {
            color: #dc3545;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚢 Titanic Survival Prediction</h1>
        
        {% if prediction_text %}
            <div class="prediction-result {{ 'result-survived' if 'Survived' in prediction_text else 'result-not-survived' }}">
                {{ prediction_text }}
            </div>
        {% endif %}

        <form action="/predict" method="post">
            <div class="form-group">
                <label for="pclass">Passenger Class</label>
                <select id="pclass" name="pclass">
                    <option value="1">1st Class</option>
                    <option value="2">2nd Class</option>
                    <option value="3">3rd Class</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="sex">Sex</label>
                <select id="sex" name="sex">
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                </select>
            </div>

            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" placeholder="e.g., 29" step="0.1" required>
            </div>
            
            <div class="form-group">
                <label for="fare">Fare</label>
                <input type="number" id="fare" name="fare" placeholder="e.g., 32.2" step="0.01" required>
            </div>

            <div class="form-group">
                <label for="sibsp">Siblings/Spouses Aboard</label>
                <input type="number" id="sibsp" name="sibsp" value="0" min="0">
            </div>

            <div class="form-group">
                <label for="parch">Parents/Children Aboard</label>
                <input type="number" id="parch" name="parch" value="0" min="0">
            </div>
            
            <div class="form-group full-width">
                <label for="embarked">Port of Embarkation</label>
                <select id="embarked" name="embarked">
                    <option value="S">Southampton (S)</option>
                    <option value="C">Cherbourg (C)</option>
                    <option value="Q">Queenstown (Q)</option>
                </select>
            </div>
            
            <button type="submit">Predict Survival</button>
        </form>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    if model is None:
        return f"<h1 style='color:red;'>Error: Model '{MODEL_FILE_NAME}' not loaded. Please check console.</h1>", 500
    return render_template_string(HTML_TEMPLATE, prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return f"<h1 style='color:red;'>Error: Model '{MODEL_FILE_NAME}' not loaded. Please check console.</h1>", 500
        
    try:
        pclass = int(request.form['pclass'])
        sex_str = request.form['sex']
        age = float(request.form.get('age', 29.7))
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form.get('fare', 14.45))
        embarked_str = request.form['embarked']

        data = {
            'Pclass': [pclass],
            'Sex': [1 if sex_str == 'female' else 0],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Embarked_Q': [1 if embarked_str == 'Q' else 0],
            'Embarked_S': [1 if embarked_str == 'S' else 0]
        }
        
        features_df = pd.DataFrame(data)
        features_df = features_df[MODEL_FEATURES]

        prediction = model.predict(features_df)
        prediction_proba = model.predict_proba(features_df)[0]

        if prediction[0] == 1:
            confidence = prediction_proba[1] * 100
            result_text = f"Prediction: Survived ({confidence:.1f}% confidence)"
        else:
            confidence = prediction_proba[0] * 100
            result_text = f"Prediction: Not Survived ({confidence:.1f}% confidence)"

    except Exception as e:
        result_text = f"Error during prediction: {e}"

    return render_template_string(HTML_TEMPLATE, prediction_text=result_text)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
