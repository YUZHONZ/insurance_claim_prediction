import shap
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from data_processing import DataProcessing

dp = DataProcessing()

class TrainingAPI:

    def __init__(self):
        self.model = None

    
    def train(self):
        pass



    def prepare_data(self):


        df = pd.read_csv('data/processed_data.csv')

        pass

