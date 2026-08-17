"""
Module de préparation et nettoyage des données

Ce module fournit les fonctions essentielles pour :
- Charger les données financières
- Nettoyer les données manquantes
- Normaliser et standardiser les données
- Créer des features temporelles
- Effectuer le split train/test avec respect de l'ordre temporel
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional


def load_financial_data(filepath: str) -> pd.DataFrame:
    """
    Charger les données financières depuis un fichier CSV.
    
    Parameters
    ----------
    filepath : str
        Chemin vers le fichier CSV
        
    Returns
    -------
    pd.DataFrame
        DataFrame avec les données financières
        
    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas
    """
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)
    return df


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'forward_fill'
) -> pd.DataFrame:
    """
    Gérer les valeurs manquantes.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec potentiellement des valeurs manquantes
    method : str, default='forward_fill'
        Méthode pour gérer les valeurs manquantes
        Options: 'forward_fill', 'backward_fill', 'interpolate', 'drop'
        
    Returns
    -------
    pd.DataFrame
        DataFrame sans valeurs manquantes
    """
    if method == 'forward_fill':
        return df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'backward_fill':
        return df.fillna(method='bfill').fillna(method='ffill')
    elif method == 'interpolate':
        return df.interpolate(method='linear')
    elif method == 'drop':
        return df.dropna()
    else:
        raise ValueError(f"Méthode inconnue: {method}")


def create_returns(
    df: pd.DataFrame,
    column: str = 'Close',
    return_type: str = 'log'
) -> pd.Series:
    """
    Créer les rendements à partir des prix.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec les prix
    column : str, default='Close'
        Colonne à utiliser pour les rendements
    return_type : str, default='log'
        Type de rendement: 'log' ou 'simple'
        
    Returns
    -------
    pd.Series
        Série des rendements
    """
    if return_type == 'log':
        returns = np.log(df[column] / df[column].shift(1))
    elif return_type == 'simple':
        returns = df[column].pct_change()
    else:
        raise ValueError(f"Type de rendement inconnu: {return_type}")
    
    return returns.dropna()


def normalize_data(
    data: pd.DataFrame,
    method: str = 'standard',
    fit_data: Optional[pd.DataFrame] = None
) -> Tuple[np.ndarray, StandardScaler or MinMaxScaler]:
    """
    Normaliser les données.
    
    Parameters
    ----------
    data : pd.DataFrame
        Données à normaliser
    method : str, default='standard'
        Méthode de normalisation: 'standard' (z-score) ou 'minmax'
    fit_data : pd.DataFrame, optional
        Données d'entraînement pour fit (utile pour train/test)
        
    Returns
    -------
    Tuple[np.ndarray, scaler]
        Données normalisées et objet scaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    if fit_data is not None:
        scaler.fit(fit_data)
    else:
        scaler.fit(data)
    
    normalized = scaler.transform(data)
    return normalized, scaler


def create_lagged_features(
    series: pd.Series,
    n_lags: int = 5
) -> pd.DataFrame:
    """
    Créer des features décalées (lags) pour les modèles ML.
    
    Parameters
    ----------
    series : pd.Series
        Série temporelle
    n_lags : int, default=5
        Nombre de lags à créer
        
    Returns
    -------
    pd.DataFrame
        DataFrame avec les features décalées
    """
    df = pd.DataFrame(series)
    df.columns = ['target']
    
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = series.shift(i)
    
    return df.dropna()


def train_test_split_timeseries(
    df: pd.DataFrame,
    test_size: float = 0.2,
    shuffle: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split train/test en respectant l'ordre temporel.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à diviser
    test_size : float, default=0.2
        Proportion de données de test (0.0 à 1.0)
    shuffle : bool, default=False
        Ne pas utiliser avec les séries temporelles!
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Données d'entraînement et de test
    """
    if shuffle:
        raise Warning("Ne pas shuffler les données temporelles!")
    
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    return train, test


def get_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculer les statistiques descriptives.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame
        
    Returns
    -------
    pd.DataFrame
        Statistiques descriptives (count, mean, std, min, max, etc.)
    """
    return df.describe()
