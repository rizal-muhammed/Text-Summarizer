import os

from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import torch

from TextSummarizer.logging import logger

from TextSummarizer.entity import ModelTrainerConfig, ModelTrainerParams


class ModelTrainer:
    def __init__(self,
                 config:ModelTrainerConfig,
                 params:ModelTrainerParams) -> None:
        self.config = config
        self.params = params
    
    def load_samsum_dataset(self):
        """
        loads the data and returns.

        Parameters
        ----------
        None

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

            data_set = load_from_disk(self.config.data_path)

            logger.info(f"""Data loaded successfully.""")
            return data_set

        except Exception as e:
            logger.exception(f"""Exception during loading dataset.""")
        
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_checkpoint).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

        dataset = self.load_samsum_dataset()

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.params.num_train_epochs,
            warmup_steps=self.params.warmup_steps,
            per_device_train_batch_size=self.params.per_device_train_batch_size,
            per_device_eval_batch_size=self.params.per_device_eval_batch_size,
            weight_decay=self.params.weight_decay,
            logging_steps=self.params.logging_steps,
            evaluation_strategy=self.params.evaluation_strategy,
            eval_steps=self.params.eval_steps,
            save_steps=self.params.save_steps,
            gradient_accumulation_steps=self.params.gradient_accumulation_steps
        )

        trainer = Trainer(model=model_pegasus,
                          args=trainer_args,
                          tokenizer=tokenizer,
                          data_collator=seq2seq_data_collator,
                          train_dataset=dataset['validation'],
                          eval_dataset=dataset['train'])
        trainer.train()

        # save model and tokenizer
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))

