from pathlib import Path

from pydantic import BaseModel, DirectoryPath, FilePath


class DataIngestionArtifact(BaseModel):
    train_file_path: FilePath
    test_file_path: FilePath


class DataValidationArtifact(BaseModel):
    schema_file_path: FilePath
    report_file_dir: DirectoryPath
    is_validated: bool


class DataTransformationArtifact(BaseModel):
    preprocessed_object_path: FilePath


class ModelTrainerArtifact(BaseModel):
    trained_model_file: FilePath


class MetricReportArtifact(BaseModel):
    experiment_id: str
    model_name: str
    model_obj: object
    report: object


class MetricEvalArtifact(BaseModel):
    best_model: object
    best_train_eval_param: float
    best_test_eval_param: float
    best_eval_param_difference: float
    best_model_name: str
    best_model_report: object


class ModelEvaluationArtifact(BaseModel):
    is_model_accepted: bool
    evaluated_model_path: Path

class ModelPusherArtifact(BaseModel):
     best_model_path : FilePath
     is_accepted : bool

class OptunaTrainingArtifact(BaseModel):
    Model_module : str
    Model_class : str
    Model_Best_params : dict
    Model_Best_params_score : float
    training_artifact_file_path: str