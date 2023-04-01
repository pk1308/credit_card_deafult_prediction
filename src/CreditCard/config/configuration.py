import os
import sys
from pathlib import Path

from CreditCard.constants import (CURRENT_TIME_STAMP, ROOT_DIR)
from CreditCard.entity import (DataIngestionConfig, DataTransformationConfig,
                               DataValidationConfig, TrainingPipelineConfig, ModelTrainerConfig, ModelEvaluationConfig,
                               ModelPusherConfig)
from CreditCard.exception import AppException
from CreditCard.logging import logger
from CreditCard.utils import create_directories, read_yaml


class ConfigurationManager:

    def __init__(self, config_file_path: Path) -> None:
        """ Configuration manager class to read the configuration file and create the configuration objects.

        Args:
            config_file_path (Path, optional): _description_. Defaults to CONFIG_FILE_PATH.

        Raises:
            AppException: _description_ if the configuration file is not found or if the configuration file is not
            in the correct format.
        """

        try:
            self.config_info = read_yaml(path_to_yaml=Path(config_file_path))
            self.pipeline_config = self.get_training_pipeline_config()
            self.time_stamp = CURRENT_TIME_STAMP

        except Exception as e:
            raise AppException(e, sys) from e

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """ Get the data ingestion configuration object.

        Raises:
            AppException: _description_

        Returns:
            DataIngestionConfig:  Pydanctic base model data ingestion configuration object. 
            dataset_download_id: str
            raw_data_file_path: Path
            ingested_train_file_path: Path
            ingested_test_data_path: Path
            random_state: int     """

        try:
            logger.info("Getting data ingestion configuration.")
            data_ingestion_info = self.config_info.data_ingestion_config
            pipeline_config = self.pipeline_config
            artifact_dir = pipeline_config.artifact_dir
            random_state = pipeline_config.training_random_state
            dataset_download_id = data_ingestion_info.dataset_download_id
            data_ingestion_dir_name = data_ingestion_info.ingestion_dir
            raw_data_dir = data_ingestion_info.raw_data_dir
            raw_file_name = data_ingestion_info.dataset_download_file_name

            data_ingestion_dir = os.path.join(artifact_dir, data_ingestion_dir_name)
            raw_data_file_path = os.path.join(data_ingestion_dir, raw_data_dir, raw_file_name)
            ingested_dir_name = data_ingestion_info.ingested_dir
            ingested_dir_path = os.path.join(data_ingestion_dir, ingested_dir_name)

            ingested_train_file_path = os.path.join(ingested_dir_path, data_ingestion_info.ingested_train_file)
            ingested_test_file_path = os.path.join(ingested_dir_path, data_ingestion_info.ingested_test_file)
            create_directories([os.path.dirname(raw_data_file_path), os.path.dirname(ingested_train_file_path)])

            data_ingestion_config = DataIngestionConfig(dataset_download_id=dataset_download_id,
                                                        raw_data_file_path=raw_data_file_path,
                                                        ingested_train_file_path=ingested_train_file_path,
                                                        ingested_test_data_path=ingested_test_file_path,
                                                        random_state=random_state)
            logger.info(f"Data ingestion config: {data_ingestion_config.dict()}")
            logger.info("Data ingestion configuration completed.")

            return data_ingestion_config
        except Exception as e:
            raise AppException(e, sys) from e

    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        """ Get the training pipeline configuration object.
        class TrainingPipelineConfig(BaseModel):
                artifact_dir: DirectoryPath
                training_random_state: int
                pipeline_name: str
                experiment_code: str
        
        """
        try:
            training_config = self.config_info.training_pipeline_config
            training_pipeline_name = training_config.pipeline_name
            training_experiment_code = training_config.experiment_code
            training_random_state = training_config.random_state
            training_artifacts = os.path.join(ROOT_DIR, training_config.artifact_dir)
            create_directories(path_to_directories=[training_artifacts])
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=training_artifacts,
                                                              experiment_code=training_experiment_code,
                                                              pipeline_name=training_pipeline_name,
                                                              training_random_state=training_random_state)
            logger.info(f"Training pipeline config: {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise AppException(e, sys) from e

    def get_data_validation_config(self, schema_file_path: Path , data_ingestion_config  : DataIngestionConfig) -> DataValidationConfig:
        """ Get the data validation configuration object.

        Args:
            schema_file_path (Path):  Path( "configs/schema.yaml")

        Raises:
            AppException: _description_

        Returns:
            DataValidationConfig:  class DataValidationConfig(BaseModel):
                                    schema_file_path: FilePath
                                    report_file_dir: Path
                                    data_validated_test_file_path: Path
                                    data_validated_train_path: Path
                                    train_data_file: FilePath
                                    test_data_file: FilePath """
        try:
            logger.info("Getting data validation configuration.")
            pipeline_config = self.pipeline_config
            artifact_dir = pipeline_config.artifact_dir
            train_data_file = data_ingestion_config.ingested_train_file_path
            test_data_file = data_ingestion_config.ingested_test_data_path
            data_validation_config_info = self.config_info.data_validation_config
            validated_test_file_name = data_validation_config_info.validated_test_file
            validated_train_file_name = data_validation_config_info.validated_train_file

            data_validated_artifact_dir = Path(
                os.path.join(artifact_dir, data_validation_config_info.data_validation_dir))
            report_file_dir = os.path.join(data_validated_artifact_dir, data_validation_config_info.report_dir)
            validated_test_file = os.path.join(data_validated_artifact_dir, validated_test_file_name)
            validated_train_file = os.path.join(data_validated_artifact_dir, validated_train_file_name)

            create_directories([report_file_dir])

            data_validation_config = DataValidationConfig(schema_file_path=schema_file_path,
                                                          report_file_dir=report_file_dir,
                                                          data_validated_test_file_path=validated_test_file,
                                                          data_validated_train_file_path=validated_train_file,
                                                          train_data_file=train_data_file,
                                                          test_data_file=test_data_file)
            logger.info(f"Data validation config: {data_validation_config.dict()}")
            return data_validation_config

        except Exception as e:
            raise AppException(e, sys)

    def get_data_transformation_config(self, feature_generator_config_file_path: Path,
                                       schema_file_path : Path , data_validation_config_info : DataValidationConfig ) -> DataTransformationConfig:
        """ Get the data transformation configuration object.

        Args:
            feature_generator_config_file_path (Path): config file path to generate features
            schema_file_path (_type_):  schema file path to validate data

        Raises:
            AppException: _description_

        Returns:
            DataTransformationConfig: class DataTransformationConfig(BaseModel):
                                            data_validated_train_file_path: FilePath
                                            feature_generator_config_file_path: FilePath
                                            schema_file_path: FilePath
                                            preprocessed_object_file_path: Path
                                            random_state: int
        """
        try:
            pipeline_config = self.pipeline_config
            artifact_dir = pipeline_config.artifact_dir
            random_state = pipeline_config.training_random_state
            schema_file_path = data_validation_config_info.schema_file_path
            data_transformation_config_info = self.config_info.data_transformation_config

            data_transformation_dir_name = data_transformation_config_info.data_transformation_dir
            data_transformation_dir = os.path.join(artifact_dir, data_transformation_dir_name)
            preprocessed_object_dir = data_transformation_config_info.preprocessing_object_dir
            preprocessed_object_name = data_transformation_config_info.preprocessing_object_file_name
            preprocessed_object_file_path = os.path.join(data_transformation_dir, preprocessed_object_dir,
                                                         preprocessed_object_name)

            create_directories([os.path.dirname(preprocessed_object_file_path)])
            data_transformation_config = DataTransformationConfig(
                data_validated_train_file_path=data_validation_config_info.data_validated_train_file_path,
                feature_generator_config_file_path=feature_generator_config_file_path,
                schema_file_path=schema_file_path,
                preprocessed_object_file_path=preprocessed_object_file_path,
                random_state=random_state)
            return data_transformation_config
        except Exception as e:
            raise AppException(e, sys)

    def get_model_trainer_config(self, model_config_file_path: str, schema_file_path: str , data_validation_config_info : DataValidationConfig , 
                                 data_transformation_config_info : DataTransformationConfig) -> ModelTrainerConfig:
        """ Get the model trainer configuration object.

        Args:
            model_config_file_path (str):  model config file path to train optuna model
            schema_file_path (str): schema file path to validate data

        Raises:
            AppException: _description_

        Returns:
            ModelTrainerConfig:class ModelTrainerConfig(BaseModel):
                                model_config_file_path: FilePath
                                base_accuracy: float
                                trained_model_file_path: Path
                                model_report_dir: DirectoryPath
                                preprocessed_object_file_path: FilePath
                                to_train_data_path: FilePath
                                schema_file_path: FilePath
                                eval_difference: float
                                eval_param: str
                                experiment_id: str
        """
        try:
            pipeline_config = self.pipeline_config
            artifact_dir = pipeline_config.artifact_dir
            experiment_code = pipeline_config.experiment_code
            schema_file_path = data_validation_config_info.schema_file_path
            validated_train_file_path = data_validation_config_info.data_validated_train_file_path
            model_trainer_config_info = self.config_info.model_trainer_config
            model_trainer_artifact_dir_name = model_trainer_config_info.model_trainer_dir
            model_trainer_artifact_dir = os.path.join(artifact_dir, model_trainer_artifact_dir_name)
            model_report_dir_name = model_trainer_config_info.model_reports_dir

            trained_model_path = os.path.join(model_trainer_artifact_dir, "best_model", "best_model.pkl")
            model_report_dir = os.path.join(model_trainer_artifact_dir, model_report_dir_name)
            preprocessed_object_file_path = data_transformation_config_info.preprocessed_object_file_path

            create_directories([os.path.dirname(trained_model_path), model_report_dir])

            model_trainer_config = ModelTrainerConfig(model_config_file_path=model_config_file_path,
                                                      base_accuracy=model_trainer_config_info.base_accuracy,
                                                      trained_model_file_path=trained_model_path,
                                                      model_report_dir=model_report_dir,
                                                      preprocessed_object_file_path=preprocessed_object_file_path,
                                                      schema_file_path=schema_file_path,
                                                      eval_difference=model_trainer_config_info.eval_difference,
                                                      eval_param=model_trainer_config_info.eval_param,
                                                      experiment_id=experiment_code,
                                                      to_train_data_path=validated_train_file_path, )

            return model_trainer_config

        except Exception as e:
            raise AppException(e, sys)

    def get_model_evaluation_config(self , schema_file_path : Path, model_config_file_path: Path , pipeline_config_file_path : Path  ,
                                    data_validation_config_info : DataValidationConfig ,model_train_config : ModelTrainerConfig ) -> ModelEvaluationConfig:
        """ Get the model evaluation configuration object.

        Raises:
            AppException: _description_

        Returns:
            ModelEvaluationConfig:class ModelEvaluationConfig(BaseModel):
                                            trained_model_path: FilePath
                                            schema_file_path: FilePath
                                            train_data_path: FilePath
                                            test_data_path: FilePath
                                            pipeline_config_file_path: FilePath
                                            report_dir: DirectoryPath
                                            base_accuracy: float
                                            eval_difference: float
                                            eval_param: str
                                            eval_model_dir: DirectoryPath
                                            eval_model_path : Path
        """
        try:
            model_evaluation_config = self.config_info.model_evaluation_config
            pipeline_config = self.pipeline_config
            artifact_dir = pipeline_config.artifact_dir
            model_eval_dir_name = model_evaluation_config.model_evaluation_dir
            model_eval_dir = os.path.join(artifact_dir, model_eval_dir_name)
            eval_model_dir_name = model_evaluation_config.eval_model_dir_name
            trained_model_path = model_train_config.trained_model_file_path
            report_dir = os.path.join(model_eval_dir, "model_eval_report")
            eval_model_dir = os.path.join(model_eval_dir, eval_model_dir_name)
            eval_model_path = os.path.join(eval_model_dir, model_evaluation_config.evaluated_model_file_name)

            create_directories([report_dir, eval_model_dir])
            response = ModelEvaluationConfig(trained_model_path = trained_model_path,
                                            schema_file_path = schema_file_path,
                                            train_data_path =  data_validation_config_info.data_validated_train_file_path,
                                            test_data_path = data_validation_config_info.data_validated_test_file_path,
                                            pipeline_config_file_path = pipeline_config_file_path,
                                            report_dir = report_dir,
                                            base_accuracy =  model_evaluation_config.base_accuracy ,
                                            eval_difference = model_evaluation_config.eval_difference,
                                            eval_param = model_evaluation_config.eval_param,
                                            eval_model_dir = eval_model_dir,
                                            eval_model_path = eval_model_path)

            logger.info(f"Model Evaluation Config: {response}.")
            return response
        except Exception as e:
            raise AppException(e, sys) from e

    def get_model_pusher_config(self , schema_file_path : Path, model_config_file_path : Path, pipeline_config_file_path : Path , 
                                data_validation_config_info : DataValidationConfig ,model_eval_config  : ModelEvaluationConfig ) -> ModelPusherConfig:
        """  model pusher configuration object.

        Args:
            schema_file_path (Path): schema file path
            production_model_path 
            model_config_file_path (Path):  model config file path
            pipeline_config_file_path (Path):  pipeline config file path

        Raises:
            AppException: _description_

        Returns:
            ModelPusherConfig:  model pusher configuration object.
        """
        try:
            model_pusher_config_info = self.config_info.model_pusher_config
            model_pusher_artifact_dir_name = model_pusher_config_info.model_pusher_dir
            pipeline_config = self.pipeline_config
            artifact_dir = pipeline_config.artifact_dir
            model_pusher_artifact_dir = os.path.join(artifact_dir, model_pusher_artifact_dir_name)

            best_model_path = os.path.join(ROOT_DIR, model_pusher_config_info.model_export_dir,
                                           model_pusher_config_info.best_model_name)
            production_model_path = os.path.join(ROOT_DIR, model_pusher_config_info.model_export_dir, "production_model.pkl")
            report_dir = os.path.join(model_pusher_artifact_dir, "Model_eval_report")
            base_accuracy = model_pusher_config_info.base_accuracy
            eval_difference = model_pusher_config_info.eval_difference
            eval_param = model_pusher_config_info.eval_param
            create_directories([os.path.dirname(best_model_path), report_dir])
            model_pusher_config = ModelPusherConfig(best_model_path=best_model_path,
                                                    schema_file_path=schema_file_path,
                                                    production_model_path = production_model_path,
                                                    validated_train_path= data_validation_config_info.data_validated_train_file_path ,
                                                    validated_test_path= data_validation_config_info.data_validated_test_file_path ,
                                                    evaluated_model_path= model_eval_config.eval_model_path ,
                                                    report_dir=report_dir,
                                                    pipeline_config_file_path= pipeline_config_file_path ,
                                                    base_accuracy=base_accuracy,
                                                    eval_difference=eval_difference,
                                                    eval_param=eval_param)
            logger.info(f"Model pusher config {model_pusher_config}")
            return model_pusher_config
        except Exception as e:
            raise AppException(e, sys) from e
