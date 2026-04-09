import os
import joblib
import pandas as pd

MODEL_PATH = "data/churn_model_clean.pkl"

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), "Le modèle n'existe pas"

def test_model_load():
    model = joblib.load(MODEL_PATH)
    assert model is not None, "Le modèle ne se charge pas"

def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    # données de test
    data = pd.DataFrame([[30, 1, 5, 10]],
                        columns=['Age','Account_Manager','Years','Num_Sites'])

    prediction = model.predict(data)

    assert prediction is not None
    assert len(prediction) == 1

def test_model_output_values():
    model = joblib.load(MODEL_PATH)

    data = pd.DataFrame([[40, 0, 3, 5]],
                        columns=['Age','Account_Manager','Years','Num_Sites'])

    prediction = model.predict(data)[0]

    # classification binaire attendue
    assert prediction in [0, 1]