import sys
import os

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            obj = dill.load(file)
        logging.info("Object Loaded Successfully from: {}".format(file_path))
        return obj
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, X_test, y_train, y_test, model, params):
    try:
        logging.info("Evaluating models")
        report= {}
        for model_name, model_instance in model.items():
            param = params[model_name]
            gs = GridSearchCV(model_instance, param, cv=5, n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)
            model_instance.set_params(**gs.best_params_)
            model_instance.fit(X_train, y_train)
            
            y_train_pred = model_instance.predict(X_train)
            y_test_pred = model_instance.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            logging.info("Model: {}".format(model_name))
            logging.info("Train Score: {}".format(train_score))
            logging.info("Test Score: {}".format(test_score))

            report[model_name] = test_score

        logging.info("Model Evaluation Completed")
        logging.info("Model Report: {}".format(report))
        return report
    
    except Exception as e:
        raise CustomException(e, sys)