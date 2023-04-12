import os,sys
import numpy as np
import pandas as pd
# from sklearn.model_selection
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
import dill


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as yt:
        raise CustomException(yt,sys)


def evaluate_model(X,y,X_test,y_test,models):
    report={}

    try:
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X, y)  # train model

            y_train_pred = model.predict(X)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as yt:
        raise CustomException(yt,sys)