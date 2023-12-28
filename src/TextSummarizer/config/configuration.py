from pathlib import Path
from TextSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SECRETS_FILE_PATH
from TextSummarizer.utils import common
from TextSummarizer.entity import (DataIngestionConfig,
                                   DataTransformationConfig,
                                   DataPreprocessingConfig)


class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
                 secrets_filepath=SECRETS_FILE_PATH) -> None:
        self.config = common.read_yaml(config_filepath)
        self.params = common.read_yaml(params_filepath)
        self.credentials = common.read_yaml(secrets_filepath)

        common.create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        common.create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            destination_folder=config.destination_folder,
            filename=config.filename,
            miscellaneous_folder=config.miscellaneous_folder,
        )
    
        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        common.create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
        )

        return data_transformation_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing

        common.create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            destination_path=Path(config.destination_path),
            tokenizer_name=Path(config.tokenizer_name),
        )

        return data_preprocessing_config