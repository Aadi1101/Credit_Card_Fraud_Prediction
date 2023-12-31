import os,sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.data_visualization import DataVisualization

@dataclass
class DataIngestionConfig():
    raw_data_path:str = os.path.join('artifacts','rawdata.csv')
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv('notebook\creditcard.csv')
            logging.info("Read the data")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train_test_split initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr = data_transformation.initiate_data_transformation(train_path,test_path)
    model_trainer = ModelTrainer()
    acc,best_model_name,model_report = model_trainer.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr)
    data_visualization = DataVisualization()
    print(data_visualization.initiate_data_visualization(model_report))