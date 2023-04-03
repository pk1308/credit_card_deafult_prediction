import argparse
import os
import sys
import warnings
from pathlib import Path

import pandas
import numpy
from box import ConfigBox

from CreditCard.config import ConfigurationManager
from CreditCard.entity import (MetricEvalArtifact, ModelEvaluationArtifact,
                               ModelEvaluationConfig)
from CreditCard.exception import AppException
from CreditCard.logging import logger
from CreditCard.utils import (evaluate_classification_model, load_object,
                              read_yaml, save_object, read_yaml_as_dict)
from CreditCard.utils.common import write_json, write_yaml

warnings.filterwarnings("ignore")


class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig):
        try:
            logger.info(f"{'>>' * 30}Model Evaluation log started.{'<<' * 30} ")
            self.model_evaluation_config = model_evaluation_config
            self.train_df = pandas.read_pickle(self.model_evaluation_config.train_data_path)
            self.test_df = pandas.read_pickle(self.model_evaluation_config.test_data_path)
            self.schema = read_yaml(path_to_yaml=Path(self.model_evaluation_config.schema_file_path))
        except Exception as e:
            raise AppException(e, sys) from e

    def get_model_list(self):
        model_list = [load_object(file_path=Path(self.model_evaluation_config.trained_model_path))]
        return model_list

    def get_updated_model_status(self, report_obj: object, config_file_path: Path, eval_model_dir: Path):

        is_model_accepted = False
        config_info = read_yaml_as_dict(path_to_yaml=Path(config_file_path))
        report_data = ConfigBox(report_obj.as_dict())
        test_data_result = report_data.metrics[0].result.current
        train_data_result = report_data.metrics[0].result.reference
        test_json = {f"test_{key}": data for key, data in test_data_result.items()}
        train_json = {f"train_{key}": data for key, data in train_data_result.items()}
        data_to_dump = {**test_json, **train_json}
        logger.info(data_to_dump)
        logger.info(type(data_to_dump))
        metrics_path = os.path.join(eval_model_dir, "model_eval_report.json")
        logger.info(f"metrics_path : {metrics_path}")
        write_json(data_to_dump=data_to_dump, file_path=Path(metrics_path))
        base_accuracy = float(config_info["model_evaluation_config"]["base_accuracy"])
        test_accuracy = float(test_data_result["accuracy"])
        if test_accuracy > base_accuracy:
            is_model_accepted = True
            logger.info(f" model accepted_status {is_model_accepted}")
            config_info["model_evaluation_config"]["base_accuracy"] = test_accuracy
            logger.info(f"update config path {config_file_path} ")
            update_config_path = os.path.join(os.getcwd(), config_file_path)
            write_yaml(file_path=Path(update_config_path), data=config_info)
            logger.info("config updated")
        return is_model_accepted

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:

            model_list = self.get_model_list()
            target_column_name = self.schema.target_column
            base_model_features_to_drop = self.schema.base_model_features_to_drop
            x_train = self.train_df.drop(base_model_features_to_drop, axis=1).values
            y_train = self.train_df[target_column_name].values
            x_test = self.test_df.drop(base_model_features_to_drop, axis=1).values
            y_test = self.test_df[target_column_name].values
            logger.info(f"test_shape{x_test.shape} train_shape{x_train.shape}")
            report_dir = self.model_evaluation_config.report_dir
            base_accuracy = self.model_evaluation_config.base_accuracy
            eval_difference = self.model_evaluation_config.eval_difference
            eval_param = self.model_evaluation_config.eval_param
            columns = list(self.schema.columns_to_eval)

            metric_report_artifact: MetricEvalArtifact = evaluate_classification_model(x_train_eval=x_train,
                                                                                       y_train=y_train,
                                                                                       x_test_eval=x_test,
                                                                                       y_test=y_test,
                                                                                       base_accuracy=base_accuracy,
                                                                                       report_dir=str(report_dir),
                                                                                       eval_difference=eval_difference,
                                                                                       estimators=model_list,
                                                                                       eval_param=eval_param,
                                                                                       experiment_id="model_eval",
                                                                                       columns=columns, final_eval=True)

            logger.info(f"Model evaluation completed. model metric artifact: {metric_report_artifact}")

            if metric_report_artifact.best_model is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=None
                                                   )
                logger.info(response)
                return response

            else:
                logger.info("Trained model is found")
                eval_model_dir = self.model_evaluation_config.eval_model_dir
                logger.info(f"Evaluated model will be saved at {eval_model_dir}")
                
                eval_report_obj = metric_report_artifact.best_model_report
                eval_model_path = self.model_evaluation_config.eval_model_path
                eval_report_obj.save_html(os.path.join(eval_model_dir, "model_eval_report.html"))
                save_object(obj=metric_report_artifact.best_model, file_path=Path(eval_model_path))
                config_file_path = self.model_evaluation_config.pipeline_config_file_path
                is_model_accepted = self.get_updated_model_status(report_obj=eval_report_obj,
                                                                  config_file_path=config_file_path,
                                                                  eval_model_dir=Path(eval_model_dir))

                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=eval_model_path,
                                                                    is_model_accepted=is_model_accepted)
            return model_evaluation_artifact
        except Exception as e:
            logger.error(e)
            raise AppException(e, sys) from e

    def __del__(self):
        logger.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")


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
    model_eval_config_info = config.get_model_evaluation_config(schema_file_path=Path(args.schema), model_config_file_path=Path(args.model_config),
                                                                pipeline_config_file_path=Path(args.config) , data_validation_config_info=data_validation_config,
                                                                model_train_config=model_trainer_config)

    model_eval = ModelEvaluation(model_evaluation_config=model_eval_config_info)
    model_eval_response = model_eval.initiate_model_evaluation()
