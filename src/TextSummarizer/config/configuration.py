from pathlib import Path
from TextSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SECRETS_FILE_PATH
from TextSummarizer.utils import common
from TextSummarizer.entity import (DataIngestionConfig,
                                   DataTransformationConfig,
                                   ModelTrainerConfig,
                                   ModelTrainerParams)


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
            tokenizer_name=Path(config.tokenizer_name),
        )

        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer

        common.create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_checkpoint=Path(config.model_checkpoint),
        )

        return model_trainer_config
    
    def get_model_trainer_params(self) -> ModelTrainerParams:
        params = self.params.model_trainer

        model_trainer_params = ModelTrainerParams(
            num_train_epochs=int(params.num_train_epochs),
            warmup_steps=int(params.warmup_steps),
            per_device_train_batch_size=int(params.per_device_train_batch_size),
            per_device_eval_batch_size=int(params.per_device_eval_batch_size),
            weight_decay=float(params.weight_decay),
            logging_steps=int(params.logging_steps),
            evaluation_strategy=str(params.evaluation_strategy),
            eval_steps=int(params.eval_steps),
            save_steps=float(params.save_steps),
            gradient_accumulation_steps=int(params.gradient_accumulation_steps),
        )

        return model_trainer_params