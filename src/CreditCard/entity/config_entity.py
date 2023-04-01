from pathlib import Path

from pydantic import BaseModel, DirectoryPath, FilePath


class DataIngestionConfig(BaseModel):
    raw_data_file_path: Path
    ingested_train_file_path: Path
    ingested_test_data_path: Path
    random_state: int
    dataset_download_id: str


class TrainingPipelineConfig(BaseModel):
    artifact_dir: DirectoryPath
    training_random_state: int
    pipeline_name: str
    experiment_code: str


class DataValidationConfig(BaseModel):
    report_file_dir: Path
    data_validated_test_file_path: Path
    data_validated_train_file_path: Path
    train_data_file: FilePath
    test_data_file: FilePath
    schema_file_path: FilePath


class DataTransformationConfig(BaseModel):
    data_validated_train_file_path: FilePath
    feature_generator_config_file_path: FilePath
    schema_file_path: FilePath
    preprocessed_object_file_path: Path
    random_state: int


class ModelTrainerConfig(BaseModel):
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


class ModelEvaluationConfig(BaseModel):
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
    eval_model_path: Path


class ModelPusherConfig(BaseModel):
    schema_file_path: FilePath
    validated_train_path: FilePath
    validated_test_path: FilePath
    best_model_path: Path
    evaluated_model_path: FilePath
    production_model_path : Path
    report_dir: DirectoryPath
    pipeline_config_file_path: FilePath
    base_accuracy: float
    eval_difference: float
    eval_param: str


class OptunaTrainingConfig(BaseModel):
    Model_index: int
    Model_module: str
    Model_class: str
    report_file_path: Path
