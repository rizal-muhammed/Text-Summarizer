from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.data_transformation import DataTransformation
from TextSummarizer.logging import logger


class DataTransformationPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()

        data_transformation = DataTransformation(config=data_transformation_config)
        
        split = "train"
        null_present_in_train_set = data_transformation.is_null_present(split)
        if null_present_in_train_set:
            train_set_without_null = data_transformation.remove_missing_values(split)
            data_transformation.save_samsum_data(split, train_set_without_null)
        else:
            data_transformation.save_samsum_data(split)

        
        split = "validation"
        null_present_in_validation_set = data_transformation.is_null_present(split)
        if null_present_in_validation_set:
            validation_set_without_null = data_transformation.remove_missing_values(split)
            data_transformation.save_samsum_data(split, validation_set_without_null)
        else:
            data_transformation.save_samsum_data(split)
        

        split = "test"
        null_present_in_test_set = data_transformation.is_null_present(split)
        if null_present_in_test_set:
            test_set_without_null = data_transformation.remove_missing_values(split)
            data_transformation.save_samsum_data(split, test_set_without_null)
        else:
            data_transformation.save_samsum_data(split)