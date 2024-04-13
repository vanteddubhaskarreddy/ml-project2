import sys
import os

import numpy as np
import pandas as pd
import dill

from src.exceptions import CustomException
from src.logger import logging

def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        logging.info("Object Saved Successfully at: {}".format(file_path))
    
    except Exception as e:
        raise CustomException(e, sys)