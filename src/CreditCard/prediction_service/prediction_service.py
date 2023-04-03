import os
import warnings
from pathlib import Path

import pandas
import numpy
from CreditCard.entity import  CreditData
from CreditCard.logging import logger
from CreditCard.utils import load_object

warnings.filterwarnings("ignore")

MODEL_DIR_NAME = "production_model"
PRODUCTION_MODEL_NAME = "production_model.pkl"
class PredictionService:
    def __int__(self,):
        try:
            self.trained_model = self.get_best_model()
        except Exception as e:
            logger.error(e)

    def __get_best_model(self, ) -> object:
        production_model_path = os.path.join(MODEL_DIR_NAME, PRODUCTION_MODEL_NAME)
        model = load_object(file_path = Path(production_model_path))
        return  model

    def __get_data(self, data_to_prepare: CreditData) -> pandas.DataFrame:
        data_dict = data_to_prepare.dict()
        data_df = pandas.DataFrame(data_dict , columns=list(data_dict.keys()) , index=[0])
        return  data_df

    def get_model_prediction(self, data_to_predict : CreditData ) -> int:
        data_to_predict = self.__get_data(data_to_prepare=data_to_predict)
        model = self.__get_best_model()
        logger.info(f"Model is {model}")
        prediction = model.predict(data_to_predict)
        logger.info(f"Prediction is {prediction}")
        return  prediction[0]

