import argparse
import sys
from pathlib import Path

import numpy
import pandas
from box import ConfigBox
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from yellowbrick.cluster import KElbowVisualizer

from CreditCard.config import ConfigurationManager
from CreditCard.entity import (DataTransformationArtifact,
                               DataTransformationConfig)
from CreditCard.exception import AppException
from CreditCard.logging import logger
from CreditCard.utils import read_yaml, save_object


class FeaturePrepare(BaseEstimator, TransformerMixin):
    """custom feature generator class to generate cluster class for the data
    scaler : StandardScaler clustering using kmeans++ and kneed"""

    def __init__(self, feature_generator_config: ConfigBox):
        self.cluster = None
        try:
            self.feature_config_info = feature_generator_config
            self.encoder = OneHotEncoder(sparse_output=False)
        except Exception as e:
            raise AppException(e, sys) from e

    def get_cluster(self, data):
        try:
            kmeans = KMeans(n_init="auto", init='k-means++', random_state=42)
            visualizer = KElbowVisualizer(kmeans, k=(2, 15))
            visualizer.fit(data)
            total_clusters = visualizer.elbow_value_
            logger.info(f"total cluster :{total_clusters}")
            self.cluster = KMeans(n_clusters=total_clusters, init='k-means++', random_state=42)
            self.cluster.fit(data)
            return self.cluster
        except Exception as e:
            raise AppException(e, sys) from e

    def fit(self, data_to_fit):
        try:
            data_generated = self.__prepare_data(data=data_to_fit)
            self.encoder.fit(data_generated)
            data_encoded = self.encoder.transform(data_generated)
            self.cluster = self.get_cluster(data_encoded)
            return self
        except Exception as e:
            raise AppException(e, sys) from e

    def transform(self, data_to_transform):
        try:
            data_generated = self.__prepare_data(data=data_to_transform)
            data_encoded = self.encoder.transform(data_generated)
            prediction = self.cluster.predict(data_encoded)
            response = numpy.c_[data_encoded, prediction]
            return response
        except Exception as e:
            raise AppException(e, sys) from e

    def __prepare_data(self, data):
        try:
            feature_config_info = self.feature_config_info
            data_generated = data.copy()
            import pandas

            for master_data in feature_config_info.keys():
                feature = feature_config_info[master_data]
                for column in feature.columns:
                    data_generated[column] = data[column].clip(lower=feature.lower_bound, upper=feature.upper_bound)
                    data_generated[column] = pandas.cut(data_generated[column], bins=feature.bins)
            return data_generated
        except Exception as e:
            raise AppException(e, sys) from e


class DataTransformation:
    def __init__(self, data_transformation_config_info: DataTransformationConfig):
        """ Data Transformation class to perform data transformation on training data 

        Args:
            data_transformation_config_info (DataTransformationConfig):
            class DataTransformationConfig(BaseModel):
                                data_validated_train_file_path : FilePath
                                feature_generator_config_file_path : FilePath
                                schema_file_path: FilePath
                                preprocessed_object_file_path: Path
                                random_state: int

        Raises:
            AppException: _description_
        """
        try:
            self.data_transformation_config_info = data_transformation_config_info
            logger.info(f"{'>>' * 10}Data Transformation log started.{'<<' * 10} ")
        except Exception as e:
            raise AppException(e, sys) from e

    def get_data_transformer_object(self, feature_generator_file_path: Path) -> FeaturePrepare:
        try:
            preprocessing = FeaturePrepare(feature_generator_config=read_yaml(Path(feature_generator_file_path)))
            return preprocessing

        except Exception as e:
            raise AppException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Obtaining preprocessing object.")
            data_transformation_config_info = self.data_transformation_config_info
            feature_generator_file_path = data_transformation_config_info.feature_generator_config_file_path
            preprocessing_obj = self.get_data_transformer_object(
                feature_generator_file_path=feature_generator_file_path)
            validated_train_data_file_path = data_transformation_config_info.data_validated_train_file_path
            validated_train_data = pandas.read_pickle(validated_train_data_file_path)
            schema_file_path = data_transformation_config_info.schema_file_path
            schema_data = read_yaml(path_to_yaml=Path(schema_file_path))
            x_train = validated_train_data.drop(columns=schema_data.base_model_features_to_drop)
            preprocessing_obj.fit(data_to_fit=x_train)
            preprocessing_obj_path = data_transformation_config_info.preprocessed_object_file_path
            logger.info("Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_path, obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(preprocessed_object_path=preprocessing_obj_path)
            logger.info(f"Data transformations artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise AppException(e, sys) from e


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args_parser.add_argument('--feature_store', dest='feature_store', required=True)
    args_parser.add_argument('--schema_file', dest='schema', required=True)
    args = args_parser.parse_args()
    config = ConfigurationManager(config_file_path=args.config)
    data_ingestion_config_info = config.get_data_ingestion_config()
    data_validation_config_info = config.get_data_validation_config(schema_file_path=args.schema , data_ingestion_config=data_ingestion_config_info)
    data_transformation_config = config.get_data_transformation_config(
        feature_generator_config_file_path=args.feature_store, schema_file_path=args.schema , data_validation_config_info=data_validation_config_info)
    data_transformation = DataTransformation(data_transformation_config_info=data_transformation_config)
    data_transformation_response = data_transformation.initiate_data_transformation()
