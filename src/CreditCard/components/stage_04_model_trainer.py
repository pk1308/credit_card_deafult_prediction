import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy
import pandas

from CreditCard.config import ConfigurationManager
from CreditCard.entity import (BaseModel, ModelTrainerArtifact, ModelTrainerConfig)
from CreditCard.exception import AppException
from CreditCard.logging import logger
from CreditCard.model_factory import ModelFactory
from CreditCard.utils import (load_object, read_yaml, save_object)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")


class ModelTrainer:

    def __init__(self, model_trainer_config_info: ModelTrainerConfig):
        """ model trainer class for training using  Model factory class - optuna hyperparameter tuning

        Args:
            model_trainer_config_info (ModelTrainerConfig): class ModelTrainerConfig(BaseModel):
                                                                model_config_file_path: FilePath
                                                                base_accuracy: float
                                                                trained_model_file_path: Path
                                                                model_report_dir: DirectoryPath
                                                                preprocessed_object_file_path: FilePath
                                                                to_train_data_path: FilePath
                                                                schema_file_path: FilePath
                                                                eval_difference: float
                                                                eval_param: str
                                                                experiment_id: st

        Raises:
            AppException: _description_
        """
        try:
            logger.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config_info = model_trainer_config_info
            preprocessing_obj_path = self.model_trainer_config_info.preprocessed_object_file_path
            self.preprocessing_obj = load_object(file_path=preprocessing_obj_path)
            self.schema = read_yaml(path_to_yaml=self.model_trainer_config_info.schema_file_path)
            self.data_to_train = self.get_test_and_train_df()
        except Exception as e:
            raise AppException(e, sys) from e

    def get_test_and_train_df(self) -> numpy.ndarray:
        """ get train data from pickle file
        and preprocess it using preprocessing object.
        target column is appended to the end of the data

        Returns:
           data_to_train: numpy.ndarray: data to train
        """
        data_path = self.model_trainer_config_info.to_train_data_path
        raw_data = pandas.read_pickle(data_path)
        x_train = raw_data.drop(self.schema.base_model_features_to_drop, axis=1)
        y_train = raw_data[self.schema.target_column]
        
        smote = SMOTE(random_state=42)
        x_over, y_over = smote.fit_resample(x_train, y_train)

        x_processed = self.preprocessing_obj.transform(x_over)
        data_to_train = numpy.c_[x_processed, y_over]
        return data_to_train

    def get_model_new_config_path(self, cluster_no: int, model_path):
        file_path = str(model_path)
        new_path = file_path.replace("base", str(cluster_no))
        return new_path

    def train_base_model(self, model_config_file_path: Path, base_accuracy: float, model_report_dir: Path,
                         eval_difference: float, eval_param: str) -> object:
        """ train base model using model factory class

        Args:
            model_config_file_path (Path): model config file path used to train the model
            base_accuracy (float):  base accuracy to be achieved
            model_report_dir (Path): model report directory
            eval_difference (float): min difference between eval param
            eval_param (str):  param for evaluate model

        Raises:
            AppException: _description_

        Returns:
            object:  best trained model object
        """
        try:

            logger.info(f"Expected accuracy: {base_accuracy}")
            logger.info("Extracting model config file path")
            trained_model_columns = self.schema.base_model_trained_columns
            model_factory = ModelFactory(model_factory_config_path=model_config_file_path,
                                         data_to_train=self.data_to_train)
            best_searched_model_config_path = model_factory.initiate_model_params_search()
            best_model_list = model_factory.get_best_model(model_report_path=best_searched_model_config_path)
            logger.info(f"{'>>' * 30}Base model started{'<<' * 30} ")
            best_model, report = model_factory.get_best_evaluated_model(train_model_list=best_model_list,
                                                                        base_accuracy=base_accuracy,
                                                                        base_report_dir=model_report_dir,
                                                                        eval_difference=eval_difference,
                                                                        eval_param=eval_param,
                                                                        columns_trained_on=trained_model_columns)
            base_model_path = os.path.join(os.path.dirname(self.model_trainer_config_info.trained_model_file_path), "base_model.pkl")
            save_object(file_path=Path(base_model_path), obj=best_model)

            base_model = BaseModel(trained_model_object=best_model, preprocessing_object=self.preprocessing_obj)
            save_object(file_path=Path(self.model_trainer_config_info.trained_model_file_path), obj=base_model)

            logger.info(f"{'>>' * 30}Base model done{'<<' * 30} ")
            return report

        except Exception as e:
            logger.error(e)
            raise AppException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """ initiate model trainer function

        Raises:
            AppException: _description_

        Returns:
            ModelTrainerArtifact: ModelTrainerArtifact: class ModelTrainerArtifact(BaseModel):
                                                             trained_model_file: FilePath
        """
        try:
            logger.info("Loading transformed training dataset")
            logger.info(f"{'>>' * 30}Base Model.{'<<' * 30} ")
            model_train_config_info = self.model_trainer_config_info
            model_report_dir = model_train_config_info.model_report_dir
            base_accuracy = model_train_config_info.base_accuracy
            model_config_file_path = model_train_config_info.model_config_file_path
            eval_difference = model_train_config_info.eval_difference
            eval_param = model_train_config_info.eval_param
            report = self.train_base_model(model_config_file_path=model_config_file_path,
                                           base_accuracy=base_accuracy,
                                           model_report_dir=model_report_dir, eval_difference=eval_difference,
                                           eval_param=eval_param)
            report.save_html(os.path.join(model_report_dir, "model_report.html"))
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file=self.model_trainer_config_info.trained_model_file_path, )

            logger.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            logger.error(e)
            raise AppException(e, sys) from e

    def __del__(self):
        logger.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args_parser.add_argument('--feature_store', dest='feature_store', required=True)
    args_parser.add_argument('--model_config', dest='model_config', required=True)
    args_parser.add_argument('--schema_file', dest='schema', required=True)
    args = args_parser.parse_args()
    config = ConfigurationManager(config_file_path=args.config)
    data_ingestion_config = config.get_data_ingestion_config()
    data_validation_config = config.get_data_validation_config(schema_file_path=args.schema , data_ingestion_config=data_ingestion_config)
    data_transformation_config = config.get_data_transformation_config(
        feature_generator_config_file_path=args.feature_store, schema_file_path=args.schema , data_validation_config_info=data_validation_config)

    model_trainer_config = config.get_model_trainer_config(model_config_file_path=args.model_config,
                                                           schema_file_path=args.schema , data_transformation_config_info=data_transformation_config, 
                                                           data_validation_config_info= data_validation_config)

    model_trainer = ModelTrainer(model_trainer_config_info=model_trainer_config)

    _ = model_trainer.initiate_model_trainer()
