import os
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from TextSummarizer.logging import logger
from TextSummarizer import CustomErrorInvalidSplit
from TextSummarizer.entity import DataTransformationConfig



class DataTransformation:
    def __init__(self,
                 config:DataTransformationConfig) -> None:
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
        
    # def convert_examples_to_featrues(self, example_batch):
    #     input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)

    #     target_encodings = self.tokenizer(example_batch['summary'], text_target=example_batch['summary'], max_length=128, truncation=True)

    #     # with self.tokenizer.as_target_tokenizer():
    #     #     target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True)
        
    #     return {
    #         'input_ids': input_encodings['input_ids'],
    #         'attention_mask': input_encodings['attention_mask'],
    #         'labels': target_encodings['input_ids']
    #     }
    
    def is_null_present(self, split="train"):
        """
            This method checks whether there are null values present in the input data.

            Parameters
            ----------
            split : {"train"}, default "train"
                The split to load.
                eg: if 'train', then 'train.csv' is loaded.

            Returns
            -------
            null_present : bool type
                True if null values are present in df, False if null values are not present in df.

            Raises
            ------
            Exception

            Notes
            ------
        
        """
        try:
            dataset = self.load_samsum_dataset(split=split)
            null_present = False
            for column in dataset.column_names:
                if any(example[column] is None for example in dataset):
                    null_present = True
                    logger.info(f"Column {column} has None values.")
                    break
            
            if not null_present:
                logger.info(f"Data contains no null values.")

            return null_present

        except Exception as e:
            logger.exception(f"""Exception at 'is_null_present' method of 'DataTransformation' class. 
                             Exception message: {str(e)}""")
            raise e
    
    def remove_missing_values(self, split="train"):
        """
            This method removes the row that has None values if exists.

            Parameters
            ----------
            split : {"train"}, default "train"
                The split to load.
                eg: if 'train', then 'train.csv' is loaded.

            Returns
            -------
            dataset_cleaned : datasets.Dataset type or None
                Returns a datasets.Dataset by removing rows having None values.

            Raises
            ------
            Exception

            Notes
            ------
        
        """
        try:
            null_present = self.is_null_present(split=split)
            if null_present:
                dataset = self.load_samsum_dataset(split=split)
                dataset_cleaned = dataset.filter(lambda example: all(v is not None for v in example.values()))

                return dataset_cleaned

        except Exception as e:
            logger.exception(f"""Exception at 'remove_missing_values' method of 'DataTransformation' class.
                             Exception message: {str(e)}""")
            raise e
    
    def save_samsum_data(self, split="train", example_batch=None):
        try:
            if example_batch is not None:
                example_batch.to_csv(os.path.join(self.config.root_dir, split + ".csv"))
            else:
                dataset = self.load_samsum_dataset(split=split)
                dataset.to_csv(os.path.join(self.config.root_dir, split + ".csv"))

        except Exception as e:
            logger.exception(f"""Exception while saving {str(split)} data.
                             Exception message: {str(e)}""")
            raise e

    # def convert(self):
    #     try:
    #         train_dataset = self.load_samsum_dataset(split="train")
    #         validation_dataset = self.load_samsum_dataset(split="validation")
    #         test_dataset = self.load_samsum_dataset(split="test")

    #         # nan_values = train_dataset.filter(lambda example: all(v is not None for v in example.values()))
    #         train_dataset_cleaned = train_dataset.filter(lambda example: all(v is not None for v in example.values()))

    #         train_dataset_pt = train_dataset_cleaned.map(self.convert_examples_to_featrues, batched=True)
    #         validation_dataset_pt = validation_dataset.map(self.convert_examples_to_featrues, batched=True)
    #         test_dataset_pt = test_dataset.map(self.convert_examples_to_featrues, batched=True)

    #         train_dataset_pt.to_csv(os.path.join(self.config.root_dir, "train.csv"))
    #         validation_dataset_pt.to_csv(os.path.join(self.config.root_dir, "validation.csv"))
    #         test_dataset_pt.to_csv(os.path.join(self.config.root_dir, "test.csv"))
    #     except Exception as e:
    #         raise e
    

