import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_name= os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok= True )
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(x_train, y_train, x_test, y_test, models):

    try:

        report = {}

        for model_name, model in models.items():

            # Train model
            model.fit(x_train, y_train)

            # Prediction on test data
            y_test_preds = model.predict(x_test)

            # Calculate R2 score
            r2 = r2_score(y_test, y_test_preds)

            # Store score
            report[model_name] = r2

        return report

    except Exception as e:

        raise CustomException(e, sys)
    
    