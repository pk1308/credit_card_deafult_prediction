import argparse
import os
import sys

import pandas
from box import ConfigBox
from CreditCard.config import ConfigurationManager
from CreditCard.entity import DataValidationArtifact, DataValidationConfig
from CreditCard.exception import AppException
from CreditCard.logging import logger
from CreditCard.utils import read_yaml
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import *
from ensure import ensure_annotations


class DataValidation:
    def __init__(self, data_validation_config_info: DataValidationConfig):

        try:
            self.data_validation_config_info = data_validation_config_info
            self.train_df = pandas.read_csv(self.data_validation_config_info.train_data_file)
            self.test_df = pandas.read_csv(self.data_validation_config_info.test_data_file)
            logger.info(f"{'>>' * 30}Data Validation log started.{'<<' * 30} \n\n")
        except Exception as e:
            raise AppException(e, sys) from e

    def rename_target_train_and_test_df(self):
        try:
            self.train_df.rename(columns={"default.payment.next.month": "default"}, inplace=True)
            self.test_df.rename(columns={"default.payment.next.month": "default"}, inplace=True)
            return True
        except Exception as e:
            raise AppException(e, sys) from e

    def validate_dataset_schema(self) -> bool:
        """ Validate dataset schema using schema file

        Raises:
            AppException: _description_

        Returns:
            bool: True if schema validation is successful else False
        """
        try:
            validation_status = False
            logger.info("Validating dataset schema")
            schema_config = read_yaml(path_to_yaml=self.data_validation_config_info.schema_file_path)
            schema_dict = schema_config.columns
            target_rename_status = self.rename_target_train_and_test_df()
            logger.info(f" target rename status : {target_rename_status}")

            for column, data_type in schema_dict.items():
                self.train_df[column].astype(data_type)
                self.test_df[column].astype(data_type)
            logger.info("Dataset schema validation completed")
            validation_status = True
            logger.info(f"Validation_status {validation_status}")
            return validation_status
        except Exception as e:
            raise AppException(e, sys) from e

    @ensure_annotations
    def get_and_save_data_drift_report(self, reference_train: pandas.DataFrame, current_train: pandas.DataFrame):
        try:
            tests = TestSuite(
                tests=[TestNumberOfColumnsWithMissingValues(),
                       TestNumberOfRowsWithMissingValues(),
                       TestNumberOfConstantColumns(),
                       TestNumberOfDuplicatedRows(),
                       TestNumberOfDuplicatedColumns(),
                       TestColumnsType(),
                       TestNumberOfDriftedColumns(), ])
            drift_status = False
            current = self.train_df
            if not self.reference.empty:

                tests.run(reference_data=reference_train, current_data=current_train)
                test_file_name = f"data_drift_test_report.json"
                tests.save_html(os.path.join(self.data_validation_config_info.report_file_dir, test_file_name))
                report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
                report.run(reference_data=self.reference, current_data=current)
                profile_file_name = test_file_name.replace("json", "html")
                report.save_html(os.path.join(self.data_validation_config_info.report_file_dir, profile_file_name))
                report_data = ConfigBox(report.as_dict())
                if report_data.metrics[0].result.dataset_drift:
                    drift_status = True
            return drift_status

        except Exception as e:
            raise AppException(e, sys) from e

    def get_reference_data(self):
        try:
            reference_train = None
            reference_test = None
            logger.info("Getting reference data")
            validated_train_file_path = self.data_validation_config_info.data_validated_train_file_path
            validated_test_file_path = self.data_validation_config_info.data_validated_test_file_path
            logger.info(f"validated_train_file_path {validated_train_file_path}")
            logger.info(f"validated_test_file_path {validated_test_file_path}")

            if os.path.exists(validated_train_file_path):
                logger.info("Reference data found")
                reference_train = pandas.read_pickle(validated_train_file_path)
                reference_test = pandas.read_pickle(validated_test_file_path)
                logger.info("Reference data loaded train {reference_train.shape} test {reference_test.shape}")

            return reference_train, reference_test
        except Exception as e:
            raise AppException(e, sys) from e

    @staticmethod
    def check_data_to_insert(new_data: pandas.DataFrame, reference_data: pandas.DataFrame, key: str) -> pandas.DataFrame:
        """ check two data frame with the key and return the data to insert

        Args:
            new_data (pandas.DataFrame): _description_
            reference_data (pandas.DataFrame): _description_
            key (str):  unique key to check

        Returns:
            pandas.DataFrame:  the data to insert
        """
        data_to_insert = None
        key_list = reference_data[key].to_list()
        index_to_insert = new_data[key].isin(key_list)
        print(index_to_insert.sum())
        print(new_data.shape)
        if index_to_insert.sum() < new_data.shape[0]:
            data_to_insert = new_data[~index_to_insert]

        return data_to_insert

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            validation_status = self.validate_dataset_schema()
            reference_train, reference_test = self.get_reference_data()
            if reference_train is None and validation_status:
                self.train_df.to_pickle(self.data_validation_config_info.data_validated_train_file_path)
                self.test_df.to_pickle(self.data_validation_config_info.data_validated_test_file_path)
            else:
                data_drift = self.get_and_save_data_drift_report(reference_train=reference_train,
                                                                 current_train=self.train_df)
                if data_drift:
                    raise Exception("Data drift found")
                if validation_status and not data_drift:
                    logger.info("Data validation completed")
                    train_to_insert = self.check_data_to_insert(new_data=self.train_df, reference_data=reference_train,
                                                                key="ID")
                    test_to_insert = self.check_data_to_insert(new_data=self.test_df, reference_data=reference_test,
                                                               key="ID")
                    if train_to_insert is not None:
                        train_to_insert.concat(reference_train)
                        train_to_insert.to_pickle(self.data_validation_config_info.data_validated_train_file_path)
                    if test_to_insert is not None:
                        test_to_insert.concat(reference_test)
                        test_to_insert.to_pickle(self.data_validation_config_info.data_validated_test_file_path)
                    validation_status = True
                    logger.info("Data Validation successfully.")

                else:

                    validation_status = False
                    logger.info("Data Validation not successfully.")

            data_validation_artifact = DataValidationArtifact(
                schema_file_path=self.data_validation_config_info.schema_file_path,
                report_file_dir=self.data_validation_config_info.report_file_dir,
                is_validated=validation_status, )
            logger.info(f"{data_validation_artifact.dict()}")
            logger.info(f"{'>>' * 30}Data Validation log completed.{'<<' * 30} \n\n")

            return data_validation_artifact

        except Exception as e:
            raise AppException(e, sys) from e

    def __del__(self):
        logger.info(f"{'>>' * 30}Data Validation log completed.{'<<' * 30} \n\n")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args_parser.add_argument('--schema_file', dest='schema', required=True)
    args = args_parser.parse_args()
    config = ConfigurationManager(config_file_path=args.config)
    data_ingestion_config = config.get_data_ingestion_config()
    data_validation_config = config.get_data_validation_config(schema_file_path=args.schema , data_ingestion_config=data_ingestion_config)
    data_ingestion = DataValidation(data_validation_config_info=data_validation_config)
    data_ingestion_response = data_ingestion.initiate_data_validation()
