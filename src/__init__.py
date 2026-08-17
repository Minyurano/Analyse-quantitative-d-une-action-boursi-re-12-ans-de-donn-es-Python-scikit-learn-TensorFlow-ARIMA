"""
Analyse Quantitative et Prévision de Séries Financières

Ce package fournit une suite complète d'outils pour :
- L'analyse technique des séries temporelles financières
- La gestion du risque
- La prévision de séries temporelles
- La visualisation des résultats

Modules principaux:
    - data_processing: Préparation et nettoyage des données
    - technical_indicators: Indicateurs techniques (RSI, MACD, etc.)
    - risk_management: Calculs VaR, Sharpe ratio, etc.
    - models: Modèles de prévision (ARIMA, LSTM, Random Forest, etc.)
    - visualization: Visualisation des résultats

Author: Minyurano
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Minyurano"
__email__ = "contact@minyurano.com"

from . import data_processing
from . import technical_indicators
from . import risk_management
from . import models
from . import visualization

__all__ = [
    "data_processing",
    "technical_indicators",
    "risk_management",
    "models",
    "visualization",
]
