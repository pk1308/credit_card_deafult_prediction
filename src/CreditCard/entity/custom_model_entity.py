import pandas
from CreditCard.utils import load_object
from CreditCard.logging import logger
import numpy as np
from pathlib import Path
from ensure import ensure_annotations


class BaseModel:
    """model estimator : Train the model and save the model to pickle """

    @ensure_annotations
    def __init__(self, preprocessing_object: object, trained_model_object: object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_dict:  {cluster : model saved path}
        """
        logger.info("Base model initiated")
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def preprocess_data(self, data_to_preprocess):
        transformed_feature_to_predict = self.preprocessing_object.transform(data_to_preprocess)
        return transformed_feature_to_predict

    def predict(self, x: pandas.DataFrame):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        data_to_predict = self.preprocess_data(data_to_preprocess=x)
        prediction = self.trained_model_object.predict(data_to_predict)
        return prediction

    def predict_proba(self, x: pandas.DataFrame):
        data_to_predict = self.preprocess_data(data_to_preprocess=x)
        prediction_proba = self.trained_model_object.predict_proba(data_to_predict)
        return prediction_proba

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class EstimatorModel:
    """model estimator : Train the model and save the model to pickle """

    @ensure_annotations
    def __init__(self, preprocessing_object: object, trained_model_dict: dict):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_dict:  {cluster : model saved path}
        """
        logger.info("custom model initiated")
        logger.info(trained_model_dict.items())
        self.preprocessing_object = preprocessing_object
        self.trained_model_dict = trained_model_dict
        self.trained_model_object = {cluster: load_object(file_path=Path(model_path)) for cluster, model_path in
                                     trained_model_dict.items()}

    def preprocess_data(self, data_to_preprocess):
        transformed_feature_to_predict = self.preprocessing_object.transform_data(data_to_preprocess)
        return transformed_feature_to_predict

    def predict(self, x):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        pre_data_to_predict = self.preprocess_data(data_to_preprocess=x)
        prediction = pandas.DataFrame(columns=["prediction"])

        for row in range(pre_data_to_predict.shape[0]):
            data_to_predict = pre_data_to_predict.loc[row]
            cluster = data_to_predict["cluster"]
            model = self.trained_model_object[cluster]
            prediction.loc[row] = model.predict(data_to_predict.drop(labels=["cluster"]))

        return prediction

    def predict_proba(self, x):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        pre_data_to_predict = self.preprocess_data(data_to_preprocess=x)
        length_of_data = pre_data_to_predict.shape[0]
        prediction = np.zeros((length_of_data, 2))
        for row in range(length_of_data):
            data_to_predict = pre_data_to_predict.loc[row]
            cluster = data_to_predict["cluster"]
            model = self.trained_model_object[cluster]
            cluster_data_to_predict = data_to_predict.drop(labels=["cluster"])
            prediction[row] = model.predict_proba(cluster_data_to_predict.values.reshape(1, -1))
        return prediction

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
