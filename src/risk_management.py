"""
Module de gestion du risque

Ce module fournit les outils essentiels de gestion du risque :
- Value at Risk (VaR)
- Ratio de Sharpe
- Volatilité
- Maximum Drawdown
- Ratio de Sortino
- Ratio d'Information
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Optional


def calculate_returns(
    prices: pd.Series,
    return_type: str = 'log'
) -> pd.Series:
    """
    Calculer les rendements des prix.
    
    Parameters
    ----------
    prices : pd.Series
        Série des prix
    return_type : str, default='log'
        Type de rendement: 'log' ou 'simple'
        
    Returns
    -------
    pd.Series
        Série des rendements
    """
    if return_type == 'log':
        returns = np.log(prices / prices.shift(1))
    else:  # simple
        returns = prices.pct_change()
    
    return returns.dropna()


def calculate_volatility(
    returns: pd.Series,
    periods: int = 252
) -> float:
    """
    Calculer la volatilité annualisée.
    
    Parameters
    ----------
    returns : pd.Series
        Série des rendements
    periods : int, default=252
        Nombre de périodes par an (252 pour quotidien)
        
    Returns
    -------
    float
        Volatilité annualisée
    """
    return returns.std() * np.sqrt(periods)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods: int = 252
) -> float:
    """
    Calculer le ratio de Sharpe.
    
    Le ratio de Sharpe mesure le rendement excédentaire par unité de risque.
    Plus le ratio est élevé, meilleur est le rendement ajusté au risque.
    
    Parameters
    ----------
    returns : pd.Series
        Série des rendements
    risk_free_rate : float, default=0.02
        Taux sans risque annualisé (par défaut 2%)
    periods : int, default=252
        Nombre de périodes par an
        
    Returns
    -------
    float
        Ratio de Sharpe
    """
    excess_returns = returns.mean() * periods - risk_free_rate
    volatility = returns.std() * np.sqrt(periods)
    
    if volatility == 0:
        return 0
    
    return excess_returns / volatility


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods: int = 252
) -> float:
    """
    Calculer le ratio de Sortino.
    
    Le ratio de Sortino est similaire au ratio de Sharpe mais ne considère
    que la volatilité des rendements négatifs (downside risk).
    
    Parameters
    ----------
    returns : pd.Series
        Série des rendements
    risk_free_rate : float, default=0.02
        Taux sans risque annualisé
    periods : int, default=252
        Nombre de périodes par an
        
    Returns
    -------
    float
        Ratio de Sortino
    """
    excess_returns = returns.mean() * periods - risk_free_rate
    
    # Volatilité des rendements négatifs uniquement
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods)
    
    if downside_std == 0:
        return 0
    
    return excess_returns / downside_std


def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Calculer la Value at Risk (VaR).
    
    La VaR estime la perte maximale probable avec un niveau de confiance donné.
    Exemple: VaR(95%) = -2% signifie qu'il y a 95% de chance que la perte
    n'excède pas 2%.
    
    Parameters
    ----------
    returns : pd.Series
        Série des rendements
    confidence_level : float, default=0.95
        Niveau de confiance (0.0 à 1.0)
    method : str, default='historical'
        Méthode de calcul: 'historical', 'parametric', ou 'monte_carlo'
        
    Returns
    -------
    float
        Value at Risk (en pourcentage)
    """
    alpha = 1 - confidence_level
    
    if method == 'historical':
        # VaR historique : percentile des rendements
        var = np.percentile(returns, alpha * 100)
    
    elif method == 'parametric':
        # VaR paramétrique : assume distribution normale
        mean = returns.mean()
        std = returns.std()
        z_score = stats.norm.ppf(alpha)
        var = mean + z_score * std
    
    elif method == 'monte_carlo':
        # VaR Monte Carlo : simulation
        mean = returns.mean()
        std = returns.std()
        simulations = np.random.normal(mean, std, 10000)
        var = np.percentile(simulations, alpha * 100)
    
    else:
        raise ValueError(f"Méthode inconnue: {method}")
    
    return var


def calculate_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculer la Conditional Value at Risk (CVaR) ou Expected Shortfall.
    
    La CVaR est la moyenne des pertes qui dépassent la VaR.
    
    Parameters
    ----------
    returns : pd.Series
        Série des rendements
    confidence_level : float, default=0.95
        Niveau de confiance
        
    Returns
    -------
    float
        Conditional Value at Risk (en pourcentage)
    """
    alpha = 1 - confidence_level
    var = np.percentile(returns, alpha * 100)
    
    # Moyenne des rendements pires que la VaR
    cvar = returns[returns <= var].mean()
    
    return cvar


def calculate_max_drawdown(prices: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculer le Maximum Drawdown.
    
    Le Maximum Drawdown est la plus grande baisse cumulée par rapport au
    maximum précédent.
    
    Parameters
    ----------
    prices : pd.Series
        Série des prix
        
    Returns
    -------
    Tuple[float, pd.Timestamp, pd.Timestamp]
        (Max drawdown en %, date du peak, date du trough)
    """
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max
    
    max_drawdown = drawdown.min()
    
    # Dates correspondantes
    trough_idx = drawdown.idxmin()
    peak_idx = prices[:trough_idx].idxmax()
    
    return max_drawdown, peak_idx, trough_idx


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods: int = 252
) -> float:
    """
    Calculer le Ratio d'Information.
    
    Le ratio d'information mesure le rendement excédentaire par unité de
    tracking error par rapport à un benchmark.
    
    Parameters
    ----------
    returns : pd.Series
        Rendements du portefeuille
    benchmark_returns : pd.Series
        Rendements du benchmark
    periods : int, default=252
        Nombre de périodes par an
        
    Returns
    -------
    float
        Information Ratio
    """
    excess_returns = returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(periods)
    
    if tracking_error == 0:
        return 0
    
    return (excess_returns.mean() * periods) / tracking_error


def calculate_calmar_ratio(
    returns: pd.Series,
    periods: int = 252
) -> float:
    """
    Calculer le Ratio de Calmar.
    
    Le ratio de Calmar mesure le retour annualisé par unité de Maximum Drawdown.
    
    Parameters
    ----------
    returns : pd.Series
        Série des rendements
    periods : int, default=252
        Nombre de périodes par an
        
    Returns
    -------
    float
        Calmar Ratio
    """
    prices = (1 + returns).cumprod()
    annual_return = returns.mean() * periods
    max_dd, _, _ = calculate_max_drawdown(prices)
    
    if abs(max_dd) < 1e-10:
        return 0
    
    return annual_return / abs(max_dd)


def calculate_beta(
    returns: pd.Series,
    market_returns: pd.Series
) -> float:
    """
    Calculer le Beta par rapport au marché.
    
    Le Beta mesure la sensibilité d'un actif aux mouvements du marché.
    Beta = 1 : Même risque que le marché
    Beta > 1 : Plus volatil que le marché
    Beta < 1 : Moins volatil que le marché
    
    Parameters
    ----------
    returns : pd.Series
        Rendements de l'actif
    market_returns : pd.Series
        Rendements du marché
        
    Returns
    -------
    float
        Beta
    """
    covariance = np.cov(returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    
    if market_variance == 0:
        return 0
    
    return covariance / market_variance


def calculate_alpha(
    returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods: int = 252
) -> float:
    """
    Calculer l'Alpha (Jensen's Alpha).
    
    L'Alpha mesure la performance excédentaire par rapport au modèle CAPM.
    
    Parameters
    ----------
    returns : pd.Series
        Rendements de l'actif
    market_returns : pd.Series
        Rendements du marché
    risk_free_rate : float, default=0.02
        Taux sans risque
    periods : int, default=252
        Nombre de périodes par an
        
    Returns
    -------
    float
        Alpha annualisé
    """
    beta = calculate_beta(returns, market_returns)
    
    asset_return = returns.mean() * periods
    market_return = market_returns.mean() * periods
    
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    alpha = asset_return - expected_return
    
    return alpha


class RiskAnalyzer:
    """
    Classe pour analyser le risque de manière complète.
    """
    
    def __init__(self, prices: pd.Series, risk_free_rate: float = 0.02):
        """
        Initialiser l'analyseur de risque.
        
        Parameters
        ----------
        prices : pd.Series
            Série des prix
        risk_free_rate : float
            Taux sans risque annualisé
        """
        self.prices = prices
        self.returns = calculate_returns(prices)
        self.risk_free_rate = risk_free_rate
    
    def get_risk_metrics(self) -> dict:
        """
        Calculer tous les métriques de risque.
        
        Returns
        -------
        dict
            Dictionnaire contenant tous les métriques
        """
        volatility = calculate_volatility(self.returns)
        sharpe = calculate_sharpe_ratio(self.returns, self.risk_free_rate)
        sortino = calculate_sortino_ratio(self.returns, self.risk_free_rate)
        var_95 = calculate_var(self.returns, 0.95)
        cvar_95 = calculate_cvar(self.returns, 0.95)
        max_dd, dd_start, dd_end = calculate_max_drawdown(self.prices)
        calmar = calculate_calmar_ratio(self.returns)
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_dd,
            'drawdown_start': dd_start,
            'drawdown_end': dd_end,
            'calmar_ratio': calmar,
        }
