import os
import json
from datasets import load_dataset, load_dataset_builder, get_dataset_split_names

from TextSummarizer.utils import common
from TextSummarizer.logging import logger
from TextSummarizer.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self,
                 config:DataIngestionConfig) -> None:
        self.config = config

        common.create_directories([config.root_dir])
    
    def data_ingestion(self):
        """
        Downloads data.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        Exception

        Notes
        ------
        Creates folder and downloads data at `destination_folder` specified in config.

        """

        try:
            # Load the samsum dataset from Hugging Face
            logger.info(f"""Downloading '{self.config.filename}' dataset...""")

            # create destination folder (if not exists)
            destination_folder = self.config.destination_folder
            if not os.path.exists(destination_folder or not os.path.isdir(destination_folder)):
                common.create_directories([self.config.destination_folder])

            # create miscellaneous folder (if not exists)
            miscellaneous_folder = self.config.miscellaneous_folder
            if not os.path.exists(miscellaneous_folder) or not os.path.isdir(miscellaneous_folder):
                common.create_directories([miscellaneous_folder])
            
            # save meta data into miscellaneous folder
            ds_builder = load_dataset_builder(self.config.filename)
            ds_builder.info.write_to_directory(dataset_info_dir=miscellaneous_folder,
                                               pretty_print=True)
            
            # load data set
            splits = get_dataset_split_names(self.config.filename)
            for split in splits:
                dataset = load_dataset(self.config.filename, split=split)
                dataset.to_csv(os.path.join(destination_folder, str(split)+".csv"))



            


        except Exception as e:
            logger.exception(f"""Exception during data ingestion.
                             Exception message: {str(e)}""")
        

            



