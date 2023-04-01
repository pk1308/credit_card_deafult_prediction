import os
import sys
from datetime import datetime

import pandas
import numpy
from box import ConfigBox
from CreditCard.entity import MetricEvalArtifact, MetricReportArtifact
from CreditCard.exception import AppException
from CreditCard.logging import logger
from ensure import ensure_annotations
from evidently.metrics import (ClassificationClassBalance,
                               ClassificationClassSeparationPlot,
                               ClassificationConfusionMatrix,
                               ClassificationPRCurve,
                               ClassificationProbDistribution,
                               ClassificationPRTable,
                               ClassificationQualityByClass,
                               ClassificationQualityMetric,
                               ClassificationRocCurve,
                               ConflictPredictionMetric, ConflictTargetMetric)
from evidently.report import Report
from CreditCard.utils import create_directories
from CreditCard.logging import logger
import warnings

warnings.filterwarnings("ignore")


@ensure_annotations
def get_best_model(reports_artifacts: list, eval_param: str, eval_difference: float,
                   base_accuracy: float) -> MetricEvalArtifact:
    """ support function to get the best model based on the eval_param and eval_difference"""

    try:
        best_model = None
        best_train_eval_param = None
        best_test_eval_param = None
        best_eval_param_difference = eval_difference
        best_model_name = None
        best_model_report = eval_difference
        for report_artifact in reports_artifacts:
            classification_report = report_artifact.report
            data = ConfigBox(classification_report.as_dict())
            test_data_result = data.metrics[0].result.current
            train_data_result = data.metrics[0].result.reference
            test_eval_param = test_data_result[eval_param]
            train_eval_param = train_data_result[eval_param]
            model_eval_difference = train_eval_param - test_eval_param
            message = f"""Model :{report_artifact.model_name} , eval param : {eval_param} , 
                        Train score : {train_eval_param}, 
                        Test score {test_eval_param}  , EVAL difference {model_eval_difference}"""
            if model_eval_difference > 0.05:
                logger.warning(f"{'#' * 5} model eval difference greater than 0.05{'#' * 5}")
            logger.info(message)
            if model_eval_difference <= best_eval_param_difference and test_data_result["accuracy"] >= base_accuracy:
                best_model = report_artifact.model_obj
                best_train_eval_param = train_eval_param
                best_test_eval_param = test_eval_param
                best_eval_param_difference = model_eval_difference
                best_model_name = report_artifact.model_name
                best_model_report = report_artifact.report
                logger.info(f"eval_difference")
        if best_model is not None:
            model_eval_artifact = MetricEvalArtifact(best_model=best_model,
                                                     best_train_eval_param=best_train_eval_param,
                                                     best_test_eval_param=best_test_eval_param,
                                                     best_eval_param_difference=best_eval_param_difference,
                                                     best_model_name=best_model_name,
                                                     best_model_report=best_model_report)
            return model_eval_artifact
        else:
            return None
    except Exception as e:
        logger.error(e)
        raise AppException(e, sys)


@ensure_annotations
def evaluate_classification_model(x_train_eval: numpy.ndarray, y_train: numpy.ndarray,
                                  x_test_eval: numpy.ndarray, y_test: numpy.ndarray, base_accuracy: float,
                                  report_dir: str, eval_difference: float, estimators: list, columns: list,
                                  eval_param: str = "accuracy", experiment_id: str = None,
                                  final_eval: bool = False) -> MetricEvalArtifact:
    """_summary_

    Args:
        x_train_eval (numpy.ndarray): trained features 
        y_train (numpy.ndarray): trained target
        x_test_eval (numpy.ndarray): test features
        y_test (numpy.ndarray): test target
        base_accuracy (float): minimum accuracy to consider model
        report_dir (str): path to save the model report
        eval_difference (float): difference between train and test eval param
        estimators (list): list of trained models
        columns (list): list of columns used to train the model
        eval_param (str, optional): the model eval criteria. Defaults to "accuracy".
        experiment_id (str, optional):  experiment if. Defaults to None.
        final_eval (bool, optional): flag to indicate if the model is final. Defaults to False. to specify
        if the model is accepts pandas dataframe or numpy array

    Raises:
        AppException: _description_

    Returns:
        MetricEvalArtifact: class MetricEvalArtifact(BaseModel):
                                best_model:  best Model object
                                best_train_eval_param: eval param result for train data
                                best_test_eval_param: eval param result for test data
                                best_eval_param_difference: difference between train and test eval param
                                best_model_name: best model name
                                best_model_report: evidently report for best model
    """
    current_df = pandas.DataFrame(x_test_eval, columns=columns)
    current_df["target"] = y_test.astype(int)
    reference_df = pandas.DataFrame(x_train_eval, columns=columns)
    reference_df["target"] = y_train.astype(int)
    if final_eval:
        current_data_to_eval = current_df.iloc[:, :x_test_eval.shape[1]]
        reference_data_to_eval = reference_df.iloc[:, :x_test_eval.shape[1]]
    else:
        current_data_to_eval = current_df.iloc[:, :x_test_eval.shape[1]].values
        reference_data_to_eval = reference_df.iloc[:, :x_test_eval.shape[1]].values

    if experiment_id is None:
        experiment_id = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    model_dir = os.path.join(report_dir, "model_report")
    experiment_dir = os.path.join(model_dir, experiment_id)
    model_report_list = list()
    try:
        for estimator in estimators:
            model_name = estimator.__class__.__name__
            logger.info(f"{'*' * 10}evaluating model {model_name}{'*' * 10}")
            model = estimator
            current_df['prediction'] = model.predict_proba(current_data_to_eval)[:, 1]
            reference_df['prediction'] = model.predict_proba(reference_data_to_eval)[:, 1]
            classification_report = Report(metrics=[ClassificationQualityMetric(),
                                                    ClassificationClassBalance(),
                                                    ConflictTargetMetric(),
                                                    ConflictPredictionMetric(),
                                                    ClassificationConfusionMatrix(),
                                                    ClassificationQualityByClass(),
                                                    ClassificationClassSeparationPlot(),
                                                    ClassificationProbDistribution(),
                                                    ClassificationRocCurve(),
                                                    ClassificationPRCurve(),
                                                    ClassificationPRTable(), ])
            classification_report.run(reference_data=reference_df, current_data=current_df)
            report_file_name = f"{model_name}_evidently_classification_report.html"
            model_report_path = os.path.join(experiment_dir, report_file_name)
            create_directories([os.path.dirname(model_report_path)])
            classification_report.save_html(filename=model_report_path)
            model_report_artifact = MetricReportArtifact(experiment_id=experiment_id,
                                                         model_name=model_name,
                                                         model_obj=model,
                                                         report=classification_report)
            model_report_list.append(model_report_artifact)
        best_model = get_best_model(reports_artifacts=model_report_list, eval_param=eval_param,
                                    eval_difference=eval_difference,
                                    base_accuracy=base_accuracy)

        if best_model is None:
            logger.error("No acceptable model found")
            raise "No acceptable model found"
        else:
            logger.info(f"Acceptable model  name {best_model.best_model}. ")
            logger.info(f"Acceptable model test {eval_param} score {best_model.best_test_eval_param}. ")
            logger.info(f"best model artifact {best_model}")
    except Exception as e:
        logger.error(e)
        raise AppException(e, sys)
    return best_model
