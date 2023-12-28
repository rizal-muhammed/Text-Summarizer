from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.data_preprocessing import DataPreprocessing
from TextSummarizer.logging import logger


class DataPreprocessingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()

        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
        data_preprocessing.convert()