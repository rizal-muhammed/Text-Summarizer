from TextSummarizer.logging import logger
from TextSummarizer.pipeline.stage_01_data_ingestion_pipeline import DataIngestionPipeline
from TextSummarizer.pipeline.stage_02_data_transformation_pipeline import DataTransformationPipeline


# STAGE_NAME = f"""Data Ingestion"""
# try:
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
#     data_ingestion = DataIngestionPipeline()
#     data_ingestion.main()
#     logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed... <<<<<<<\n\n""")
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = f"""Data Transformation"""
try:
    logger.info(f""">>>>>>> Stage {STAGE_NAME} started... <<<<<<<""")
    data_transformation = DataTransformationPipeline()
    data_transformation.main()
    logger.info(f""">>>>>>> Stage {STAGE_NAME} Completed... <<<<<<<\n\n""")
except Exception as e:
    logger.exception(e)
    raise e