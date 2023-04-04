from fastapi import FastAPI, Request, Depends
from CreditCard.entity import CreditData
from CreditCard.prediction_service import PredictionService
from fastapi.middleware.cors import CORSMiddleware
from CreditCard.utils import load_object
from pathlib import Path

import pandas
import numpy 


def predict_credit_card_fraud(credit_data: CreditData):
    try:
        prediction_service = PredictionService()
        prediction=prediction_service.get_model_prediction(data_to_predict=credit_data)
        return prediction
    except Exception as e:
        raise e



app = FastAPI(
    title="Credit card prediction",
    description="""The aim of the project was to analyze the dataset and create an ML model that would predict the 
    Credit Card Defaulter. We have used Python 
    Libraries for data analysis and model creation (backend) and HTML and CSS for creating Web UI for the project""",
    version="0.0.1", )

origins = ["http://localhost",
           "http://localhost:8000",
           "*"]

app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )


@app.get("/")
async def read_index():
    return 


@app.get('/notify/v1/health')
def get_health():
    """
    Usage on K8S
    readinessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    livenessProbe:
        httpGet:
            path: /notify/v1/health
            port: 80
    :return:
        dict(msg='OK')
    """
    return dict(msg='OK')


@app.post("/predict")
async def predict_credit_risk(credit_data: CreditData = Depends()):
    """
    CreditData :
    """
    prediction = predict_credit_card_fraud(credit_data)
    return { "prediction": str(prediction)}
