import os,sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class DataTransformationConfig():
    preprocessor_obj__file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed.")
            target_column_name = 'Class'

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Splitting of training and test data completed.")

            train_arr = np.c_[
                input_feature_train_df,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_df,np.array(target_feature_test_df)
            ]
            logging.info("Data Transformation Complete.")
            return(
                train_arr,
                test_arr
            )
        except Exception as e:
            raise CustomException(e,sys)