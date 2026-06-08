import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from pandas import DataFrame

from src.models.customer_model import CustomerModel


# Carrega os binários apenas uma vez, durante a inicialização da API
@asynccontextmanager
async def lifespan(app: FastAPI):

    if not os.path.exists("models/best_model.pkl"):

        raise FileNotFoundError("Model file not found.")

    # Salva o modelo, scaler e colunas no estado global da API
    app.state.model = joblib.load("models/best_model.pkl")
    app.state.scaler = joblib.load("models/scaler.pkl")
    app.state.columns = joblib.load("models/columns.pkl")

    yield


app = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)


@app.post("/predict")
def predict_churn(payload: CustomerModel):

    try:

        # Converte o payload para um DataFrame e aplica one-hot encoding
        input_data = DataFrame([payload.model_dump()])
        customer_df = pd.get_dummies(input_data)

        # Garante que todas as colunas esperadas estejam presentes
        customer_df = customer_df.reindex(columns=app.state.columns, fill_value=0)

        # Aplica escalonamento nas colunas numéricas com o scaler carregado
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        customer_df[numerical_features] = app.state.scaler.transform(customer_df[numerical_features])

        # Realiza a previsão usando o modelo carregado
        prediction = app.state.model.predict(customer_df)[0]
        probability = app.state.model.predict_proba(customer_df)[0][1]

        churn_json = {
            "churn_prediction": int(prediction),
            "churn_probability": round(float(probability), 4),
            "status": "Churn detected" if prediction else "Churn not detected"
        }

        return churn_json

    except Exception as e:

        raise HTTPException(status_code=400, detail=f"Error processing the request: {str(e)}")
