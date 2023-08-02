import os,sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from src.utils import json_file,fig2img

@dataclass
class DataVisulatizationConfig():
    data_visualization_config:str = os.path.join('artifacts','model_accuracies.png')

class DataVisualization():
    def __init__(self):
        self.data_visualization_config = DataVisulatizationConfig()
    
    def initiate_data_visualization(self,model_report:dict):
        try:
            # model_report_dict:dict = json_file(model_report) 
            yaxis = ["DT","Ab","Gb","Rf","Svm","KNN","Nb","Cb","LoR","XGB"]
            xaxis = [value for key,value in model_report.items()]
            plt.grid(False)
            #figure = plt.figure()
            plt.barh(yaxis,xaxis,color='aqua')
            plt.xlabel('variable')
            plt.ylabel('value')
            return plt.show()
        except Exception as e:
            raise CustomException(e,sys)