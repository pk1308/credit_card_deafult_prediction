import argparse
import sys
from pathlib import Path

import pandas
from CreditCard.config import ConfigurationManager
from CreditCard.entity import DataIngestionArtifact, DataIngestionConfig
from CreditCard.exception import AppException
from CreditCard.logging import logger
from sklearn.model_selection import StratifiedShuffleSplit
from ensure import ensure_annotations


class DataIngestion:

    def __init__(self, data_ingestion_config_info: DataIngestionConfig):
        """Data Ingestion class to download data from remote url and split data into train and test set.
        using stratified split

        Args:
            data_ingestion_config_info (DataIngestionConfig): class DataIngestionConfig(BaseModel):
                                                                        dataset_download_id: str
                                                                        raw_data_file_path: Path
                                                                        ingested_train_file_path: Path
                                                                        ingested_test_data_path: Path
                                                                        random_state: int


        Raises:
            AppException: _description_
        """
        try:
            logger.info(f"{'>>' * 10}Stage 01 data ingestion started  {'<<' * 10}")
            self.data_ingestion_config = data_ingestion_config_info
        except Exception as e:
            raise AppException(e, sys)

    @ensure_annotations
    def download_data(self, dataset_download_id: str, raw_data_file_path: Path) -> bool:
        """_summary_

        Args:
            dataset_download_id (str): gitHub url to download dataset
            raw_data_file_path (Path):  path to save the data 

        Raises:
            AppException: _description_

        Returns:
            bool:  True if data is downloaded successfully
        """

        try:
            # extraction remote url to download dataset
            logger.info(f"Downloading dataset from github")
            raw_data_frame = pandas.read_csv(dataset_download_id)
            raw_data_frame.to_csv(raw_data_file_path, index=False)
            logger.info("Dataset downloaded successfully")

            return True

        except Exception as e:
            raise AppException(e, sys) from e

    @ensure_annotations
    def split_data_as_train_test(self, data_file_path: Path) -> DataIngestionArtifact:
        """ read the data split the data using stratified split and save the data into train and test set

        Args:
            data_file_path (Path):  Raw data file path

        Raises:
            AppException: _description_

        Returns:
            DataIngestionArtifact: class DataIngestionArtifact(BaseModel):
                                        train_file_path: FilePath
                                        test_file_path: FilePath
        """
        try:
            logger.info(f"{'>>' * 20}Data splitting.{'<<' * 20}")
            train_file_path = self.data_ingestion_config.ingested_train_file_path
            test_file_path = self.data_ingestion_config.ingested_test_data_path

            logger.info(f"Reading csv file: [{data_file_path}]")
            raw_data_frame = pandas.read_csv(data_file_path)

            logger.info("Splitting data into train and test")
            strat_train_set = None
            strat_test_set = None
            random_state = self.data_ingestion_config.random_state
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)

            for train_index, test_index in split.split(raw_data_frame, raw_data_frame["default.payment.next.month"]):
                strat_train_set = raw_data_frame.loc[train_index]
                strat_test_set = raw_data_frame.loc[test_index]

            if strat_train_set is not None:
                logger.info(f"Exporting training dataset to file: [{train_file_path}]")
                strat_train_set.to_csv(train_file_path, index=False)

            if strat_test_set is not None:
                logger.info(f"Exporting test dataset to file: [{test_file_path}]")
                strat_test_set.to_csv(test_file_path, index=False)
                data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                                test_file_path=test_file_path)
                logger.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
                return data_ingestion_artifact

        except Exception as e:
            raise AppException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """ initiate data ingestion"""
        try:
            data_ingestion_config_info = self.data_ingestion_config
            dataset_download_id = data_ingestion_config_info.dataset_download_id
            raw_data_file_path = data_ingestion_config_info.raw_data_file_path
            self.download_data(dataset_download_id, Path(raw_data_file_path))

            data_ingestion_response_info = self.split_data_as_train_test(data_file_path=Path(raw_data_file_path))
            logger.info(f"{'>>' * 20}Data Ingestion artifact.{'<<' * 20}")
            logger.info(f" Data Ingestion Artifact{data_ingestion_response_info.dict()}")
            logger.info(f"{'>>' * 20}Data Ingestion completed.{'<<' * 20}")
            return data_ingestion_response_info
        except Exception as e:
            raise AppException(e, sys) from e

    def __del__(self):
        logger.info(f"{'>>' * 20}Data Ingestion log completed.{'<<' * 20} \n\n")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    config = ConfigurationManager(config_file_path=args.config)
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion_response = data_ingestion.initiate_data_ingestion()
