import os
import sys

from src.exceptions import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Initiating Data Ingestion")
        try:
            df = pd.read_csv(
                'notebook/housing.csv', sep = '\s+',
                names=[
                    'crime_rate', 'zoned_land', 'industry', 
                    'charles_dummy_var', 'nox_conc', 'rooms', 
                    'age', 'distance', 'highways', 'property_tax', 
                    'pt_ratio', 'black_prop', 'lower_status_popu', 'price_in_thousands'
                    ]
            )
            logging.info("Data Ingested Successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw Data Saved Successfully at: {}".format(self.ingestion_config.raw_data_path))

            logging.info("Splitting Data into Train and Test")
            train_data_set, test_data_set = train_test_split(df, test_size=0.2, random_state=42)
            train_data_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_data_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data Split Successfully into Train and Test at: {} and {}".format(self.ingestion_config.train_data_path, self.ingestion_config.test_data_path))

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        
        except Exception as e:
            raise CustomException(e, sys)
    

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

