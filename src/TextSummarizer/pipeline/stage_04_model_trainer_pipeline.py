from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.logging import logger
from TextSummarizer.components.model_trainer import ModelTrainer


class ModelTrainerPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_params = config.get_model_trainer_params()

        model_trainer = ModelTrainer(config=model_trainer_config,
                                     params=model_trainer_params)
        model_trainer.train()