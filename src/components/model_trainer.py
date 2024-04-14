import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.metrics import r2_score

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    best_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_array, test_array):
        logging.info("Initiating model training")
        try:
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1]
            )
            logging.info("Finished splitting data into train and test sets")

            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "KNN": KNeighborsRegressor(),
                "SVM": SVR(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0),
            }

            # Hyperparameter Tuning
            params = {
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [5, 10, 15],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 5],
                },
                "Decision Tree": {
                    "max_depth": [5, 10, 15],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 5],
                },
                "AdaBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1],
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1],
                    "max_depth": [3, 5, 7],
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                },
                "XGBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1],
                    "max_depth": [3, 5, 7],
                },
                "CatBoost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1],
                    "max_depth": [3, 5, 7],
                },
                "Linear Regression": {},
                "SVM": {},
                
            }

            model_report = evaluate_model(X_train, X_test, y_train, y_test, models, params)
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            logging.info("Best Model: {} with R2 Score: {}".format(best_model_name, best_model_score))

            best_model = models[best_model_name]

            if best_model_score < 0.8:
                logging.info("Best Model Score is less than 0.8. Model Training Failed")
                raise CustomException("Best Model Score is less than 0.8. Model Training Failed", sys)
            logging.info("Model Training Completed Successfully")
            logging.info("Best Model is {} with r2 score {}".format(best_model_name, best_model_score))

            save_object(best_model, self.model_trainer_config.best_model_file_path)
            logging.info("Best Model Saved Successfully at: {}".format(self.model_trainer_config.best_model_file_path))

            best_model.fit(X_train, y_train)

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            logging.info("Model Evaluation Completed and model name is {}".format(best_model_name))
            logging.info("Train Score: {}".format(train_score))
            logging.info("Test Score: {}".format(test_score))

            return best_model_name, test_score
        
        except Exception as e:
            raise CustomException(e, sys)