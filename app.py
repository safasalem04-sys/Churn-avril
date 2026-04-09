from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__) 

# charger le modèle
model = joblib.load('data/churn_model_clean.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Age = float(request.form['Age'])
    Account_Manager = float(request.form['Account_Manager'])
    Years = float(request.form['Years'])
    Num_Sites = float(request.form['Num_Sites'])

    data = pd.DataFrame([[Age, Account_Manager, Years, Num_Sites]],
                        columns=['Age','Account_Manager','Years','Num_Sites'])

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "Le client va partir ❌"
    else:
        result = "Le client reste ✅"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
