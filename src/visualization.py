"""
Module de visualisation

Ce module fournit les fonctions de visualisation pour :
- Séries temporelles
- Indicateurs techniques
- Prévisions vs réalité
- Distribution des rendements
- Performance des modèles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List


# Configuration du style par défaut
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 6)
plt.rcParams['font.size'] = 10


def plot_price_series(
    df: pd.DataFrame,
    column: str = 'Close',
    title: str = 'Prix de Clôture',
    figsize: Tuple[int, int] = (15, 6)
) -> plt.Figure:
    """
    Tracer une série de prix.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les prix
    column : str
        Colonne à tracer
    title : str
        Titre du graphique
    figsize : Tuple[int, int]
        Dimensions du graphique
        
    Returns
    -------
    plt.Figure
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df.index, df[column], linewidth=2, color='steelblue')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Prix ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_price_with_ma(
    df: pd.DataFrame,
    price_col: str = 'Close',
    ma_short: int = 20,
    ma_long: int = 50,
    figsize: Tuple[int, int] = (15, 6)
) -> plt.Figure:
    """
    Tracer les prix avec les moyennes mobiles.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec les prix
    price_col : str
        Colonne des prix
    ma_short : int
        Période de la courte MA
    ma_long : int
        Période de la longue MA
    figsize : Tuple[int, int]
        Dimensions du graphique
        
    Returns
    -------
    plt.Figure
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculer les moyennes mobiles
    sma_short = df[price_col].rolling(window=ma_short).mean()
    sma_long = df[price_col].rolling(window=ma_long).mean()
    
    # Tracer
    ax.plot(df.index, df[price_col], label='Prix', linewidth=2, color='black')
    ax.plot(df.index, sma_short, label=f'SMA {ma_short}', 
            linewidth=2, color='blue', alpha=0.7)
    ax.plot(df.index, sma_long, label=f'SMA {ma_long}', 
            linewidth=2, color='red', alpha=0.7)
    
    ax.set_title('Prix avec Moyennes Mobiles', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Prix ($)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_rsi(
    df: pd.DataFrame,
    rsi_col: str = 'RSI',
    figsize: Tuple[int, int] = (15, 4)
) -> plt.Figure:
    """
    Tracer le RSI.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec le RSI
    rsi_col : str
        Colonne du RSI
    figsize : Tuple[int, int]
        Dimensions du graphique
        
    Returns
    -------
    plt.Figure
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(df.index, df[rsi_col], linewidth=2, color='darkblue')
    ax.axhline(y=70, color='r', linestyle='--', linewidth=1, label='Suracheté (70)')
    ax.axhline(y=30, color='g', linestyle='--', linewidth=1, label='Survendu (30)')
    ax.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
    
    ax.set_title('Relative Strength Index (RSI)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('RSI', fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_macd(
    df: pd.DataFrame,
    macd_col: str = 'MACD',
    signal_col: str = 'MACD_Signal',
    histogram_col: str = 'MACD_Histogram',
    figsize: Tuple[int, int] = (15, 4)
) -> plt.Figure:
    """
    Tracer le MACD.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec le MACD
    macd_col : str
        Colonne du MACD
    signal_col : str
        Colonne du signal
    histogram_col : str
        Colonne de l'histogramme
    figsize : Tuple[int, int]
        Dimensions du graphique
        
    Returns
    -------
    plt.Figure
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Tracer l'histogramme
    colors = ['green' if x > 0 else 'red' for x in df[histogram_col]]
    ax.bar(df.index, df[histogram_col], label='Histogramme', 
           color=colors, alpha=0.3)
    
    # Tracer les lignes
    ax.plot(df.index, df[macd_col], label='MACD', 
            linewidth=2, color='blue')
    ax.plot(df.index, df[signal_col], label='Signal', 
            linewidth=2, color='red')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title('MACD (Moving Average Convergence Divergence)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('MACD', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_bollinger_bands(
    df: pd.DataFrame,
    close_col: str = 'Close',
    upper_col: str = 'BB_Upper',
    lower_col: str = 'BB_Lower',
    middle_col: str = 'BB_Middle',
    figsize: Tuple[int, int] = (15, 6)
) -> plt.Figure:
    """
    Tracer les Bandes de Bollinger.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec les Bandes de Bollinger
    close_col : str
        Colonne de clôture
    upper_col : str
        Colonne de la bande supérieure
    lower_col : str
        Colonne de la bande inférieure
    middle_col : str
        Colonne de la bande du milieu
    figsize : Tuple[int, int]
        Dimensions du graphique
        
    Returns
    -------
    plt.Figure
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Remplir entre les bandes
    ax.fill_between(df.index, df[upper_col], df[lower_col],
                    alpha=0.1, color='blue', label='Bande')
    
    # Tracer les lignes
    ax.plot(df.index, df[close_col], label='Prix', 
            linewidth=2, color='black')
    ax.plot(df.index, df[upper_col], label='Bande Supérieure', 
            linewidth=1, color='red', linestyle='--')
    ax.plot(df.index, df[middle_col], label='SMA 20', 
            linewidth=1, color='blue')
    ax.plot(df.index, df[lower_col], label='Bande Inférieure', 
            linewidth=1, color='green', linestyle='--')
    
    ax.set_title('Bandes de Bollinger', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Prix ($)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Modèle',
    figsize: Tuple[int, int] = (15, 6)
) -> plt.Figure:
    """
    Tracer les prédictions vs valeurs réelles.
    
    Parameters
    ----------
    y_true : np.ndarray
        Valeurs réelles
    y_pred : np.ndarray
        Valeurs prédites
    model_name : str
        Nom du modèle
    figsize : Tuple[int, int]
        Dimensions du graphique
        
    Returns
    -------
    plt.Figure
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(y_true))
    ax.plot(x, y_true, label='Valeurs Réelles', linewidth=2, color='blue')
    ax.plot(x, y_pred, label='Prédictions', linewidth=2, color='red')
    
    ax.set_title(f'Prédictions vs Valeurs Réelles - {model_name}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Temps', fontsize=12)
    ax.set_ylabel('Valeur', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (15, 6)
) -> plt.Figure:
    """
    Tracer les résidus.
    
    Parameters
    ----------
    y_true : np.ndarray
        Valeurs réelles
    y_pred : np.ndarray
        Valeurs prédites
    figsize : Tuple[int, int]
        Dimensions du graphique
        
    Returns
    -------
    plt.Figure
        Figure matplotlib
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Graphique temporel des résidus
    axes[0].plot(residuals, linewidth=1, color='darkred')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_title('Résidus au Fil du Temps', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Temps', fontsize=10)
    axes[0].set_ylabel('Résidu', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Histogramme des résidus
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].set_title('Distribution des Résidus', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Résidu', fontsize=10)
    axes[1].set_ylabel('Fréquence', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_returns_distribution(
    returns: pd.Series,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Tracer la distribution des rendements.
    
    Parameters
    ----------
    returns : pd.Series
        Série des rendements
    figsize : Tuple[int, int]
        Dimensions du graphique
        
    Returns
    -------
    plt.Figure
        Figure matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogramme
    axes[0].hist(returns, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('Distribution des Rendements', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Rendement', fontsize=10)
    axes[0].set_ylabel('Fréquence', fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_cumulative_returns(
    returns: pd.Series,
    figsize: Tuple[int, int] = (15, 6)
) -> plt.Figure:
    """
    Tracer les rendements cumulatifs.
    
    Parameters
    ----------
    returns : pd.Series
        Série des rendements
    figsize : Tuple[int, int]
        Dimensions du graphique
        
    Returns
    -------
    plt.Figure
        Figure matplotlib
    """
    cumulative = (1 + returns).cumprod() - 1
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cumulative.index, cumulative.values, linewidth=2, color='green')
    ax.fill_between(cumulative.index, cumulative.values, alpha=0.3, color='green')
    
    ax.set_title('Rendements Cumulatifs', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rendement Cumulatif', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Tracer la matrice de corrélation.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame avec les colonnes numériques
    figsize : Tuple[int, int]
        Dimensions du graphique
        
    Returns
    -------
    plt.Figure
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax, cbar_kws={'label': 'Corrélation'})
    
    ax.set_title('Matrice de Corrélation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
