import os

from datasets import load_dataset
from transformers import AutoTokenizer

from TextSummarizer.logging import logger
from TextSummarizer.utils import common
from TextSummarizer import CustomErrorInvalidSplit
from TextSummarizer.entity import DataPreprocessingConfig


class DataPreprocessing:
    def __init__(self,
                 config:DataPreprocessingConfig) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    
    def load_samsum_dataset(self, split="train"):
        """
        loads the data and returns.

        Parameters
        ----------
        split : {"train"}, default "train"
            The split to load.
            eg: if 'train', then 'train.csv' is loaded.

        Returns
        -------
        datasets or None
            Returns loaded in datasets.Datasets format.

        Raises
        ------
        Exception

        Notes
        ------
        None

        """
        try:
            logger.info(f"""Loading dataset...""")

            if split not in ["train", "validation", "test"]:
                raise CustomErrorInvalidSplit(f"split should be 'train', 'validation' or 'test'.")
            else:
                data_file = os.path.join(self.config.data_path, str(split) + ".csv")
                data_set = load_dataset('csv', data_files=[data_file], split="train")

                logger.info(f"""Data loaded successfully.""")
                return data_set

        except Exception as e:
            logger.exception(f"""Exception during loading {str(split)} set.""")
    
    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)

        target_encodings = self.tokenizer(example_batch['summary'], text_target=example_batch['summary'], max_length=128, truncation=True)

        # with self.tokenizer.as_target_tokenizer():
        #     target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True)
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    
    def convert(self):
        """
        loads the data and and convert examples to features.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        Raises
        ------
        Exception

        Notes
        ------
        Preprocessed data is saved at 'destination_path' specified in config(refer 'data_preprocessing' config).

        """
        try:
            train_dataset = self.load_samsum_dataset(split="train")
            validation_dataset = self.load_samsum_dataset(split="validation")
            test_dataset = self.load_samsum_dataset(split="test")

            train_dataset_pt = train_dataset.map(self.convert_examples_to_features, batched=True)
            validation_dataset_pt = validation_dataset.map(self.convert_examples_to_features, batched=True)
            test_dataset_pt = test_dataset.map(self.convert_examples_to_features, batched=True)

            common.create_directories([self.config.destination_path])
            train_dataset_pt.to_csv(os.path.join(self.config.destination_path, "train.csv"))
            validation_dataset_pt.to_csv(os.path.join(self.config.destination_path, "validation.csv"))
            test_dataset_pt.to_csv(os.path.join(self.config.destination_path, "test.csv"))
        except Exception as e:
            raise e