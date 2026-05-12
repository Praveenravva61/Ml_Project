import sys
import os
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor
)

from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils.common import (
    save_object,
    evaluate_model
)


@dataclass
class ModelTrainingConfig:

    trained_model_file_path: str = os.path.join(
        'artifacts',
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Splitting training and test input data"
            )

            x_train, y_train, x_test, y_test = (

                train_array[:, :-1],
                train_array[:, -1],

                test_array[:, :-1],
                test_array[:, -1]
            )

            print(x_train.shape)
            print(y_train.shape)
            print(x_test.shape)
            print(y_test.shape)

            models = {

                "LinearRegression": LinearRegression(),

                "Lasso": Lasso(),

                "Ridge": Ridge(),

                "KNeighborsRegressor":
                    KNeighborsRegressor(),

                "DecisionTree":
                    DecisionTreeRegressor(),

                "RandomForest":
                    RandomForestRegressor(),

                "XGBoost":
                    XGBRegressor(),

                "CatBoost":
                    CatBoostRegressor(verbose=False),

                "AdaBoost":
                    AdaBoostRegressor()
            }

            model_report = evaluate_model(
                x_train,
                y_train,
                x_test,
                y_test,
                models
            )

            best_model_score = max(
                sorted(model_report.values())
            )

            best_model_name = list(
                model_report.keys()
            )[
                list(model_report.values()).index(
                    best_model_score
                )
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found",
                    sys
                )

            logging.info(
                "Best model found on both training and testing dataset"
            )

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model
            )

            predict = best_model.predict(x_test)

            r2_square = r2_score(
                y_test,
                predict
            )

            return r2_square

        except Exception as e:

            raise CustomException(e, sys)