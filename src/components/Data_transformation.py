import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


from src.logger import logging 
from src.exception import CustomException
from src.utils.common import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_path= os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        try:
            num_cols= ['reading score', 'writing score']
            cat_cols= ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            
            
            num_pipeline= Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            cat_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy= "most_frequent")),
                    ('one_hot_encoder', OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean= False))
                ]
            )
            
            logging.info(f'Numerical_columns: {num_cols}')
            logging.info(f"Categorical_columns: {cat_cols}")
            
            
            preprocessor= ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_cols),
                    ("categorical_pipeline", cat_pipeline, cat_cols)
                ]
            )
            
            return preprocessor
            
            
            
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train = pd.read_csv(train_path)
            test= pd.read_csv(test_path)
            
            preprocess_obj= self.get_data_transformer_obj()
            target= "math score"
            
            x_train= train.drop(columns= [target])
            y_train= train[target]
            
            x_test= test.drop(columns= [target])
            y_test= test[target]
            
            logging.info(" Apply preprocess obj on train, test data............")
            
            input_train_arr= preprocess_obj.fit_transform(x_train)
            input_test_arr= preprocess_obj.transform(x_test)
            
            train_arr= np.c_[
                input_train_arr,
                np.array(y_train)

                
            ]
            test_arr= np.c_[
                input_test_arr,
                np.array(y_test)

                
            ]
            
            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_path,
                obj= preprocess_obj
            )
                    
            
            return(
                
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )
            
            logging.info('Proprocessing done...')
        except Exception as e:
            raise CustomException(e, sys)
        
