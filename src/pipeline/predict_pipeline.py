import sys
import os

import numpy as np
import pandas as pd

from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_object

class PredictData():
    def __init__(self) -> None:
        self.model = load_object(os.path.join('artifacts', 'model.pkl'))
        self.preprocessor = load_object(os.path.join('artifacts', 'preprocessor.pkl'))
    
    def predict(self, data):
        try:
            logging.info("Initiating Prediction")
            logging.info("Data is {}".format(data))
            transformed_data = self.preprocessor.transform(data)
            prediction = self.model.predict(transformed_data)
            logging.info("Prediction Completed")
            logging.info("Prediction is {}".format(prediction))
            return prediction
        except Exception as e:
            raise CustomException(e, sys)

class CustomData():
    def __init__(
            self,
            crime_rate: float,
            zoned_land: float,
            industry: float,
            charles_dummy_var: float,
            nox_conc: float,
            rooms: float,
            age: float,
            distance: float,
            highways: float,
            property_tax: float,
            pt_ratio: float,
            black_prop: float,
            lower_status_popu: float
    ):
        self.crime_rate = crime_rate
        self.zoned_land = zoned_land
        self.industry = industry
        self.charles_dummy_var = charles_dummy_var
        self.nox_conc = nox_conc
        self.rooms = rooms
        self.age = age
        self.distance = distance
        self.highways = highways
        self.property_tax = property_tax
        self.pt_ratio = pt_ratio
        self.black_prop = black_prop
        self.lower_status_popu = lower_status_popu
    

    def get_data_as_data_frame(self):
        try:
            data = {
                'crime_rate': [self.crime_rate],
                'zoned_land': [self.zoned_land],
                'industry': [self.industry],
                'charles_dummy_var': [self.charles_dummy_var],
                'nox_conc': [self.nox_conc],
                'rooms': [self.rooms],
                'age': [self.age],
                'distance': [self.distance],
                'highways': [self.highways],
                'property_tax': [self.property_tax],
                'pt_ratio': [self.pt_ratio],
                'black_prop': [self.black_prop],
                'lower_status_popu': [self.lower_status_popu]
            }
            logging.info("Data Converted to Data Frame")
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)