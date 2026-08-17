"""
Module d'indicateurs techniques

Ce module fournit les indicateurs techniques populaires :
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bandes de Bollinger
- Moyennes mobiles (SMA, EMA)
- ATR (Average True Range)
- Stochastique
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_sma(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculer la Simple Moving Average (SMA).
    
    Parameters
    ----------
    series : pd.Series
        Série de prix
    window : int, default=20
        Nombre de périodes pour la moyenne
        
    Returns
    -------
    pd.Series
        Simple Moving Average
    """
    return series.rolling(window=window).mean()


def calculate_ema(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculer l'Exponential Moving Average (EMA).
    
    Parameters
    ----------
    series : pd.Series
        Série de prix
    window : int, default=20
        Nombre de périodes pour la moyenne
        
    Returns
    -------
    pd.Series
        Exponential Moving Average
    """
    return series.ewm(span=window, adjust=False).mean()


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculer le Relative Strength Index (RSI).
    
    L'RSI mesure la magnitude des changements de prix récents pour évaluer
    les conditions de suracheté ou survente.
    - RSI > 70 : Potentiellement suracheté
    - RSI < 30 : Potentiellement survendu
    
    Parameters
    ----------
    series : pd.Series
        Série de prix (généralement Close)
    window : int, default=14
        Nombre de périodes pour le calcul (par défaut 14)
        
    Returns
    -------
    pd.Series
        RSI values (0-100)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculer le MACD (Moving Average Convergence Divergence).
    
    Le MACD est un indicateur de momentum qui montre la relation entre
    deux moyennes mobiles.
    
    Parameters
    ----------
    series : pd.Series
        Série de prix
    fast : int, default=12
        Période de la EMA rapide
    slow : int, default=26
        Période de la EMA lente
    signal : int, default=9
        Période de la ligne de signal
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        (MACD line, Signal line, Histogram)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculer les Bandes de Bollinger.
    
    Les Bandes de Bollinger mesurent la volatilité et les niveaux de support/résistance.
    
    Parameters
    ----------
    series : pd.Series
        Série de prix
    window : int, default=20
        Nombre de périodes pour la moyenne mobile
    num_std : float, default=2.0
        Nombre d'écarts-types
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        (Middle band, Upper band, Lower band)
    """
    middle_band = series.rolling(window=window).mean()
    std_dev = series.rolling(window=window).std()
    
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return middle_band, upper_band, lower_band


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14
) -> pd.Series:
    """
    Calculer l'Average True Range (ATR).
    
    L'ATR mesure la volatilité en utilisant l'amplitude de movement.
    
    Parameters
    ----------
    high : pd.Series
        Série des prix hauts
    low : pd.Series
        Série des prix bas
    close : pd.Series
        Série des prix de clôture
    window : int, default=14
        Nombre de périodes pour la moyenne
        
    Returns
    -------
    pd.Series
        Average True Range
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    
    return atr


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculer le Stochastique (%K et %D).
    
    L'oscillateur stochastique compare un prix de clôture particulier
    à une plage de prix sur une période donnée.
    
    Parameters
    ----------
    high : pd.Series
        Série des prix hauts
    low : pd.Series
        Série des prix bas
    close : pd.Series
        Série des prix de clôture
    window : int, default=14
        Nombre de périodes de lookback
    smooth_k : int, default=3
        Nombre de périodes pour lisser %K
    smooth_d : int, default=3
        Nombre de périodes pour lisser %D
        
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (%K, %D)
    """
    low_min = low.rolling(window=window).min()
    high_max = high.rolling(window=window).max()
    
    k_percent = 100 * ((close - low_min) / (high_max - low_min))
    k_percent_smooth = k_percent.rolling(window=smooth_k).mean()
    d_percent = k_percent_smooth.rolling(window=smooth_d).mean()
    
    return k_percent_smooth, d_percent


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculer l'On-Balance Volume (OBV).
    
    L'OBV utilise le volume pour prédire les changements de prix.
    
    Parameters
    ----------
    close : pd.Series
        Série des prix de clôture
    volume : pd.Series
        Série des volumes
        
    Returns
    -------
    pd.Series
        On-Balance Volume
    """
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def calculate_roc(series: pd.Series, window: int = 12) -> pd.Series:
    """
    Calculer le Rate of Change (ROC).
    
    Le ROC mesure la vitesse de changement de prix.
    
    Parameters
    ----------
    series : pd.Series
        Série de prix
    window : int, default=12
        Nombre de périodes pour le calcul
        
    Returns
    -------
    pd.Series
        Rate of Change
    """
    roc = ((series - series.shift(window)) / series.shift(window)) * 100
    return roc


def create_technical_indicators(
    df: pd.DataFrame,
    price_col: str = 'Close'
) -> pd.DataFrame:
    """
    Créer un ensemble complet d'indicateurs techniques.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec les colonnes OHLCV standard
    price_col : str, default='Close'
        Nom de la colonne des prix de clôture
        
    Returns
    -------
    pd.DataFrame
        DataFrame avec tous les indicateurs
    """
    result = df.copy()
    
    # Moyennes mobiles
    result['SMA_20'] = calculate_sma(df[price_col], 20)
    result['EMA_12'] = calculate_ema(df[price_col], 12)
    
    # RSI
    result['RSI_14'] = calculate_rsi(df[price_col], 14)
    
    # MACD
    macd, signal, histogram = calculate_macd(df[price_col])
    result['MACD'] = macd
    result['MACD_Signal'] = signal
    result['MACD_Histogram'] = histogram
    
    # Bandes de Bollinger
    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(df[price_col])
    result['BB_Upper'] = bb_upper
    result['BB_Middle'] = bb_middle
    result['BB_Lower'] = bb_lower
    
    # ATR
    if 'High' in df.columns and 'Low' in df.columns:
        result['ATR_14'] = calculate_atr(df['High'], df['Low'], df[price_col])
    
    # Stochastique
    if 'High' in df.columns and 'Low' in df.columns:
        k_percent, d_percent = calculate_stochastic(
            df['High'], df['Low'], df[price_col]
        )
        result['Stochastic_K'] = k_percent
        result['Stochastic_D'] = d_percent
    
    # ROC
    result['ROC_12'] = calculate_roc(df[price_col], 12)
    
    # OBV
    if 'Volume' in df.columns:
        result['OBV'] = calculate_obv(df[price_col], df['Volume'])
    
    return result.dropna()
