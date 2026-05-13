import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_name= os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok= True )
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(x_train, y_train, x_test, y_test, models, params):

    try:

        report = {}

        for model_name, model in models.items():

            param = params[model_name]

            gs = GridSearchCV(model, param, cv=3)

            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)

            model.fit(x_train, y_train)

            y_test_preds = model.predict(x_test)

            r2 = r2_score(y_test, y_test_preds)

            report[model_name] = r2

        return report

    except Exception as e:

        raise CustomException(e, sys)

    except Exception as e:

        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
    