import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.best_optimal_algo import BestOptimalAlgo
# from datetime import datetime

dataset = pd.read_csv("notebook/data.csv")

X = dataset.drop(["class"], axis=1)
Y = dataset[["class"]]

X_tr, X_t, y_tr, y_t = train_test_split(X, Y, test_size=0.2, random_state=42)

dataframe = pd.DataFrame()
best_model = []


algo = BestOptimalAlgo(X_train=X_tr,X_test=X_t, y_test=y_t, y_train=y_tr)

# startTime = datetime.now()

dataframe = algo.best_optimal_algo()

# endTime =  datetime.now()

print("Report : ", dataframe)
# print("Best model : ", best_model)
# print("Duration :", startTime-endTime)