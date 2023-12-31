from TextSummarizer.logging import logger
from TextSummarizer.pipeline.stage_01_data_ingestion_pipeline import DataIngestionPipeline
from TextSummarizer.pipeline.stage_02_data_transformation_pipeline import DataTransformationPipeline
from TextSummarizer.pipeline.stage_04_model_trainer_pipeline import ModelTrainerPipeline


# STAGE_NAME = f"""Data Ingestion"""
# try:
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
#     data_ingestion = DataIngestionPipeline()
#     data_ingestion.main()
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed... <<<<<<<\n\n""")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = f"""Data Transformation"""
# try:
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
#     data_transformation = DataTransformationPipeline()
#     data_transformation.main()
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed... <<<<<<<\n\n""")
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = f"""Model Training"""
try:
    logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
    model_trainer = ModelTrainerPipeline()
    model_trainer.main()
    logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed... <<<<<<<\n\n""")
except Exception as e:
    logger.exception(e)
    raise e