import sys
import os

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        logging.info("Initiating Data Transformation")
        try:
            num_columns = [
                    'crime_rate', 'zoned_land', 'industry', 
                    'charles_dummy_var', 'nox_conc', 'rooms', 
                    'age', 'distance', 'highways', 'property_tax', 
                    'pt_ratio', 'black_prop', 'lower_status_popu'
                    ]
            
            logging.info("Only Numerical Columns are present in the dataset")

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('standard_scalar', StandardScaler()),
                #('poly_features', PolynomialFeatures(degree=2))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_transformer', num_pipeline, num_columns)
                ]
            )

            logging.info("Data Transformation Pipeline Created Successfully")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("Data Loaded Successfully")

            preprocessing_obj = self.get_data_transformer_object()

            target_column = 'price_in_thousands'
            numerical_columns = [
                    'crime_rate', 'zoned_land', 'industry', 
                    'charles_dummy_var', 'nox_conc', 'rooms', 
                    'age', 'distance', 'highways', 'property_tax', 
                    'pt_ratio', 'black_prop', 'lower_status_popu'
                    ]
            logging.info("Target Column: {}".format(target_column))
            logging.info("Numerical Columns: {}".format(numerical_columns))
            input_train_df = train_df.drop(target_column, axis=1)
            target_train_df = train_df[target_column]

            input_test_df = test_df.drop(target_column, axis=1)
            target_test_df = test_df[target_column]

            logging.info("Data Transformation Started")

            input_train_array = preprocessing_obj.fit_transform(input_train_df)
            input_test_array = preprocessing_obj.transform(input_test_df)

            logging.info("Data Transformed Successfully")

            train_array = np.c_[input_train_array, np.array(target_train_df)]
            test_array = np.c_[input_test_array, np.array(target_test_df)]

            logging.info("Data Merged Successfully with Target Column using numpy function np.c_")

            save_object(preprocessing_obj, self.data_transformation_config.preprocessor_obj_file_path)

            return train_array, test_array, self.data_transformation_config.preprocessor_obj_file_path
        
        except Exception as e:
            raise CustomException(e, sys)