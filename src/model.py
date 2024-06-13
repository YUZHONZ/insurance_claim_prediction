import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from xgboost import XGBRegressor

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KMeansClustering:
    def __init__(self, df=None, random_state=0):
        """
        Initialize the KMeansClustering class with a DataFrame and a random state.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame with features for clustering.
        random_state (int): The seed used by the random number generator.
        """
        self.df = df
        self.random_state = random_state
        self.scaled_data = None
        self.kmeans = None
        self.best_k = None
        self.scaler = None
        
        if df is not None:
            self._validate_inputs()
            self.scale_features()
        
    def _validate_inputs(self):
        if self.df.empty:
            raise ValueError("The input DataFrame is empty.")
        if not all(self.df.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
            raise ValueError("All columns in the DataFrame must be numeric.")
    
    def scale_features(self):
        logger.info("Scaling features...")
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.df)

        return self.scaled_data
        
    def _calculate_silhouette_score(self, k, **kmeans_kwargs):
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, **kmeans_kwargs)
        kmeans.fit(self.scaled_data)
        score = silhouette_score(self.scaled_data, kmeans.labels_)
        return score
    
    def _find_best_k(self, start_k, end_k, interval, **kmeans_kwargs):
        k_values = list(range(start_k, end_k + 1, interval))
        silhouette_scores = []

        logger.info("Starting to find the best k value...")
        for k in tqdm(k_values, desc="Finding best k"):
            try:
                score = self._calculate_silhouette_score(k, **kmeans_kwargs)
                silhouette_scores.append(score)
            except Exception as e:
                logger.error(f"Error with k={k}: {e}")
                silhouette_scores.append(-1)  # Assign a poor score for invalid k

        self.best_k = k_values[silhouette_scores.index(max(silhouette_scores))]
        logger.info(f"Best k value found: {self.best_k}")
    
    def _fit_kmeans_model(self, **kmeans_kwargs):
        self.kmeans = KMeans(n_clusters=self.best_k, random_state=self.random_state, **kmeans_kwargs)
        self.kmeans.fit(self.scaled_data)
        self.df['Cluster'] = self.kmeans.labels_
    
    def perform_clustering(self, start_k, end_k, interval=2, **kmeans_kwargs):
        """
        Perform clustering on the DataFrame and return the DataFrame with cluster labels, scaled data, KMeans model, and best k value.
        
        Parameters:
        start_k (int): The starting value for the number of clusters.
        end_k (int): The ending value for the number of clusters.
        interval (int): The step size for the range of k values.
        kmeans_kwargs (dict): Additional arguments for the KMeans model.
        
        Returns:
        pd.DataFrame: The DataFrame with an additional column for cluster labels.
        np.ndarray: The scaled data used for clustering.
        KMeans: The fitted KMeans model.
        int: The best k value based on the silhouette score.
        """
        start_k = max(2, start_k)
        end_k = min(len(self.df) - 1, end_k)

        self._find_best_k(start_k, end_k, interval, **kmeans_kwargs)
        self._fit_kmeans_model(**kmeans_kwargs)
        
        return self.df, self.scaled_data, self.kmeans, self.best_k
    
    def predict(self, new_data, model_path=None):
        if not all(new_data.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
            raise ValueError("All columns in the new data must be numeric.")
        
        if self.kmeans is None:
            if model_path is None:
                raise ValueError("KMeans model is not available. Provide a model path to load the model.")
            self.load_model(model_path)
        
        scaled_new_data = self.scaler.transform(new_data)
        return self.kmeans.predict(scaled_new_data)
    
    def save_model(self, file_path):
        model_data = {
            'kmeans_model': self.kmeans
        }
        joblib.dump(model_data, file_path)
        logger.info(f"Model saved to {file_path}")
    
    def load_model(self, file_path):
        model_data = joblib.load(file_path)
        self.scaler = model_data['scaler']
        self.kmeans = model_data['kmeans_model']
        logger.info(f"Model loaded from {file_path}")

class XGBRegression:
    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None, space=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        if space is None:
            self.space = self._define_search_space()
        self.trials = Trials()
        self.best_params = None
        self.model_best = None

    def _define_search_space(self):
        """Define the search space for hyperparameter optimization."""
        space = {
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
            'n_estimators': hp.choice('n_estimators', range(10, 300)),
            'max_depth': hp.choice('max_depth', range(1, 20)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'gamma': hp.uniform('gamma', 0, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'reg_alpha': hp.loguniform('reg_alpha', -3, 3),
            'reg_lambda': hp.loguniform('reg_lambda', -3, 3),
        }
        return space

    def _objective(self, params):
        """Objective function for hyperparameter optimization."""
        model = XGBRegressor(
            learning_rate=params['learning_rate'],
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_child_weight=int(params['min_child_weight']),
            subsample=params['subsample'],
            gamma=params['gamma'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            objective='reg:squarederror',
            random_state=42
        )

        # Calculate the cross-validation score
        mse = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error').mean()
        return {'loss': -mse, 'status': STATUS_OK}

    def run_hyperparameter_search(self, max_evals=50):
        """Run the hyperparameter optimization using the defined search space and objective function."""
        params = fmin(fn=self._objective, space=self.space, trials=self.trials, algo=tpe.suggest, max_evals=max_evals, return_argmin=False)
        self.best_params = space_eval(self.space, params)

        return self.best_params

    def train_best_model(self):
        """Train the model using the best hyperparameters."""
        if not self.best_params:
            raise ValueError("Run hyperparameter search before training the model.")

        self.model_best = XGBRegressor(
            learning_rate=self.best_params['learning_rate'],
            n_estimators=self.best_params['n_estimators'],
            max_depth=self.best_params['max_depth'],
            min_child_weight=int(self.best_params['min_child_weight']),
            subsample=self.best_params['subsample'],
            gamma=self.best_params['gamma'],
            colsample_bytree=self.best_params['colsample_bytree'],
            reg_alpha=self.best_params['reg_alpha'],
            reg_lambda=self.best_params['reg_lambda'],
            objective='reg:squarederror',
            random_state=42
        )
        self.model_best.fit(self.X_train, self.y_train)

        return self.model_best

    def evaluate_model(self, transform=True):
        """Evaluate the model on the test set and return the mean squared error."""
        if not self.model_best:
            raise ValueError("Train the model before evaluation.")

        y_pred = self.model_best.predict(self.X_test)
        if transform:
            y_pred_transform = np.expm1(y_pred)
            y_test_transform = np.expm1(self.y_test)
            mse = mean_squared_error(y_test_transform, y_pred_transform)
        else:
            mse = mean_squared_error(self.y_test, y_pred)

        return mse

    def save_model(self, file_path):
        """Save the trained model to a file."""
        if not self.model_best:
            raise ValueError("Train the model before saving.")

        joblib.dump(self.model_best, file_path)
        logger.info(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """Load a model from a file."""
        self.model_best = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")

    def predict(self, X, model_path=None):
        """Make predictions using the trained model."""
        if model_path:
            self.load_model(model_path)
        
        if not self.model_best:
            raise ValueError("Train the model or load a model before making predictions.")
        
        return self.model_best.predict(X)