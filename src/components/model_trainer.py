import os,sys
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_models, save_object

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig():
    trained_model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting training and testing data.")
            X_train,y_train,X_test,y_test=(
                train_arr[:,1:-1],
                train_arr[:,-1],
                test_arr[:,1:-1],
                test_arr[:,-1]
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "AdaboostClassifier":AdaBoostClassifier(),
                "Gradient Boosting Classifier":GradientBoostingClassifier(),
                "Random Forest Classifier":RandomForestClassifier(),
                "Support Vector Machine":SVC(),
                "K Nearest Neighbours":KNeighborsClassifier(),
                "Naive Bayes":GaussianNB(),
                "Catboost Classifier":CatBoostClassifier(verbose=False),
                "XGBoost Classifer ":XGBClassifier()
            }

            model_report:dict=evaluate_models(x_train=X_train,y_train=y_train,x_test=X_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            logging.info(f"best model on both training and testing data")

            save_object(file_path=self.model_trainer_config.trained_model_path,obj=best_model)

            predicted = best_model.predict(X_test)
            acc = accuracy_score(y_test,predicted)
            return(acc,best_model_name)
            
        except Exception as e:
            raise CustomException(e,sys)
