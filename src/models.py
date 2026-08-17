"""
Module de modèles de prévision

Ce module fournit les modèles de prévision de séries temporelles :
- ARIMA (Statistical)
- LSTM (Deep Learning)
- Random Forest (Ensemble)
- Gradient Boosting (XGBoost, LightGBM)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class BaseTimeSeriesModel:
    """Classe de base pour les modèles de séries temporelles."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.model = None
    
    def fit(self, X_train, y_train):
        raise NotImplementedError
    
    def predict(self, X_test):
        raise NotImplementedError
    
    def evaluate(self, y_true, y_pred):
        """Évaluer le modèle avec les métriques standards."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
        }


class ARIMAModel(BaseTimeSeriesModel):
    """
    Modèle ARIMA (AutoRegressive Integrated Moving Average).
    
    ARIMA est un modèle statistique classique pour les séries temporelles.
    Nécessite la bibliothèque statsmodels.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Initialiser le modèle ARIMA.
        
        Parameters
        ----------
        order : Tuple[int, int, int]
            Tuple (p, d, q) pour ARIMA
            - p : nombre de termes autorégressifs
            - d : degré de différentiation
            - q : nombre de termes de moyenne mobile
        """
        super().__init__("ARIMA")
        self.order = order
    
    def fit(self, y_train: pd.Series):
        """
        Entraîner le modèle ARIMA.
        
        Parameters
        ----------
        y_train : pd.Series
            Données d'entraînement
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.model = ARIMA(y_train, order=self.order).fit()
            self.is_fitted = True
        except ImportError:
            raise ImportError("statsmodels est requis pour ARIMA")
    
    def predict(self, steps: int) -> np.ndarray:
        """
        Faire des prédictions.
        
        Parameters
        ----------
        steps : int
            Nombre de pas à prévoir
            
        Returns
        -------
        np.ndarray
            Prédictions
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit d'abord être entraîné")
        
        forecast = self.model.get_forecast(steps=steps)
        return forecast.predicted_mean.values


class LSTMModel(BaseTimeSeriesModel):
    """
    Modèle LSTM (Long Short-Term Memory).
    
    LSTM est un réseau de neurones récurrent bien adapté aux séries temporelles.
    Nécessite la bibliothèque tensorflow/keras.
    """
    
    def __init__(
        self,
        lookback: int = 60,
        lstm_units: int = 50,
        dense_units: int = 25,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Initialiser le modèle LSTM.
        
        Parameters
        ----------
        lookback : int
            Nombre de pas temporels à regarder en arrière
        lstm_units : int
            Nombre d'unités LSTM
        dense_units : int
            Nombre d'unités dans la couche dense
        epochs : int
            Nombre d'epochs d'entraînement
        batch_size : int
            Taille du batch
        """
        super().__init__("LSTM")
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = None
    
    def _create_sequences(
        self,
        data: np.ndarray,
        lookback: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Créer les séquences pour LSTM."""
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:(i + lookback)])
            y.append(data[i + lookback])
        return np.array(X), np.array(y)
    
    def fit(self, y_train: np.ndarray):
        """
        Entraîner le modèle LSTM.
        
        Parameters
        ----------
        y_train : np.ndarray
            Données d'entraînement (normalisées recommandées)
        """
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            raise ImportError("TensorFlow/Keras est requis pour LSTM")
        
        # Créer les séquences
        X_train, y_train_seq = self._create_sequences(y_train, self.lookback)
        
        # Construire le modèle
        self.model = Sequential([
            LSTM(self.lstm_units, return_sequences=True,
                 input_shape=(self.lookback, 1)),
            Dropout(0.2),
            LSTM(self.lstm_units),
            Dropout(0.2),
            Dense(self.dense_units, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Entraîner le modèle
        X_train = X_train.reshape(-1, self.lookback, 1)
        self.model.fit(
            X_train, y_train_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        
        self.is_fitted = True
    
    def predict(self, y_recent: np.ndarray, steps: int) -> np.ndarray:
        """
        Faire des prédictions.
        
        Parameters
        ----------
        y_recent : np.ndarray
            Les dernières valeurs observées (derniers lookback points)
        steps : int
            Nombre de pas à prévoir
            
        Returns
        -------
        np.ndarray
            Prédictions
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit d'abord être entraîné")
        
        predictions = []
        current_sequence = y_recent[-self.lookback:].copy()
        
        for _ in range(steps):
            X_input = current_sequence.reshape(1, self.lookback, 1)
            next_pred = self.model.predict(X_input, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Ajouter la prédiction et supprimer le premier élément
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        return np.array(predictions)


class RandomForestRegressor(BaseTimeSeriesModel):
    """Modèle Random Forest pour la régression."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 20):
        """
        Initialiser le modèle Random Forest.
        
        Parameters
        ----------
        n_estimators : int
            Nombre d'arbres
        max_depth : int
            Profondeur maximale des arbres
        """
        super().__init__("RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Entraîner le modèle."""
        try:
            from sklearn.ensemble import RandomForestRegressor as RF
            self.model = RF(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            self.is_fitted = True
        except ImportError:
            raise ImportError("scikit-learn est requis")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Faire des prédictions."""
        if not self.is_fitted:
            raise ValueError("Le modèle doit d'abord être entraîné")
        return self.model.predict(X_test)


class XGBoostModel(BaseTimeSeriesModel):
    """Modèle XGBoost pour la régression."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6
    ):
        """
        Initialiser le modèle XGBoost.
        
        Parameters
        ----------
        n_estimators : int
            Nombre d'itérations de boosting
        learning_rate : float
            Taux d'apprentissage
        max_depth : int
            Profondeur maximale
        """
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Entraîner le modèle."""
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=42,
                verbosity=0
            )
            self.model.fit(X_train, y_train)
            self.is_fitted = True
        except ImportError:
            raise ImportError("xgboost est requis")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Faire des prédictions."""
        if not self.is_fitted:
            raise ValueError("Le modèle doit d'abord être entraîné")
        return self.model.predict(X_test)


def create_lagged_features_for_ml(
    series: pd.Series,
    n_lags: int = 5,
    n_future: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Créer les features et targets pour ML.
    
    Parameters
    ----------
    series : pd.Series
        Série temporelle
    n_lags : int
        Nombre de lags pour les features
    n_future : int
        Nombre de pas à prévoir
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X features, y targets)
    """
    X, y = [], []
    
    for i in range(len(series) - n_lags - n_future):
        X.append(series.iloc[i:i + n_lags].values)
        y.append(series.iloc[i + n_lags + n_future - 1])
    
    return np.array(X), np.array(y)


def walk_forward_validation(
    series: pd.Series,
    train_size: int,
    test_size: int,
    model_class: type
) -> dict:
    """
    Effectuer une walk-forward validation.
    
    La walk-forward validation simule un vrai scénario d'investissement
    où on entraîne sur le passé et teste sur le futur, puis on glisse
    le fenêtre d'un pas.
    
    Parameters
    ----------
    series : pd.Series
        Série temporelle
    train_size : int
        Taille de la fenêtre d'entraînement
    test_size : int
        Taille de la fenêtre de test
    model_class : type
        Classe du modèle à utiliser
        
    Returns
    -------
    dict
        Résultats de la validation
    """
    results = {
        'predictions': [],
        'actuals': [],
        'metrics': []
    }
    
    for i in range(train_size, len(series) - test_size, test_size):
        train_data = series.iloc[i - train_size:i]
        test_data = series.iloc[i:i + test_size]
        
        # Entraîner le modèle
        model = model_class()
        model.fit(train_data.values)
        
        # Faire des prédictions
        preds = model.predict(len(test_data))
        
        results['predictions'].extend(preds)
        results['actuals'].extend(test_data.values)
    
    return results
