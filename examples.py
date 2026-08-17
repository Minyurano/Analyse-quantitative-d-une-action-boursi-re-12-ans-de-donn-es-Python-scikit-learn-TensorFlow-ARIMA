"""
Exemples d'utilisation du package d'analyse quantitative

Ce fichier fournit des exemples pratiques pour utiliser le package.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Imports depuis les modules
from src.data_processing import (
    load_financial_data,
    handle_missing_values,
    create_returns,
    normalize_data,
    train_test_split_timeseries
)

from src.technical_indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    create_technical_indicators
)

from src.risk_management import (
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_var,
    calculate_max_drawdown,
    RiskAnalyzer
)

from src.models import ARIMAModel, LSTMModel, RandomForestRegressor
from src.visualization import (
    plot_price_series,
    plot_rsi,
    plot_macd,
    plot_predictions,
    plot_cumulative_returns
)


def example_1_load_and_explore():
    """
    Exemple 1 : Charger et explorer les données
    """
    print("\n" + "="*50)
    print("EXEMPLE 1 : Charger et Explorer les Données")
    print("="*50 + "\n")
    
    # Charger les données
    data_path = Path("data/gold_stock.csv")
    df = load_financial_data(str(data_path))
    
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nPremières lignes:\n{df.head()}")
    print(f"\nStatistiques:\n{df.describe()}")
    
    return df


def example_2_data_preprocessing(df):
    """
    Exemple 2 : Préparation des données
    """
    print("\n" + "="*50)
    print("EXEMPLE 2 : Préparation des Données")
    print("="*50 + "\n")
    
    # Gérer les valeurs manquantes
    df = handle_missing_values(df, method='interpolate')
    print("✓ Valeurs manquantes gérées")
    
    # Calculer les rendements
    returns = create_returns(df, column='Close', return_type='log')
    print(f"✓ Rendements calculés")
    print(f"  - Moyenne quotidienne: {returns.mean():.6f}")
    print(f"  - Écart-type: {returns.std():.6f}")
    print(f"  - Min: {returns.min():.6f}")
    print(f"  - Max: {returns.max():.6f}")
    
    # Normaliser les données
    normalized, scaler = normalize_data(df[['Close']], method='minmax')
    print(f"✓ Données normalisées (MinMax)")
    print(f"  - Shape: {normalized.shape}")
    print(f"  - Min: {normalized.min():.4f}")
    print(f"  - Max: {normalized.max():.4f}")
    
    return df, returns, normalized, scaler


def example_3_technical_indicators(df):
    """
    Exemple 3 : Calculer les indicateurs techniques
    """
    print("\n" + "="*50)
    print("EXEMPLE 3 : Indicateurs Techniques")
    print("="*50 + "\n")
    
    # Calculer les indicateurs individuellement
    print("Calcul des indicateurs individuels...")
    rsi = calculate_rsi(df['Close'], window=14)
    print(f"✓ RSI calculé (dernière valeur: {rsi.iloc[-1]:.2f})")
    
    macd, signal, histogram = calculate_macd(df['Close'])
    print(f"✓ MACD calculé")
    print(f"  - Dernière valeur MACD: {macd.iloc[-1]:.6f}")
    print(f"  - Dernière valeur Signal: {signal.iloc[-1]:.6f}")
    
    upper, middle, lower = calculate_bollinger_bands(df['Close'], window=20, num_std=2)
    print(f"✓ Bandes de Bollinger calculées")
    
    # Ou créer tous les indicateurs à la fois
    print("\nCréation de tous les indicateurs...")
    df_indicators = create_technical_indicators(df, price_col='Close')
    print(f"✓ Tous les indicateurs créés")
    print(f"  - Nombre de colonnes: {df_indicators.shape[1]}")
    print(f"  - Colonnes: {df_indicators.columns.tolist()[:5]}...")
    
    return df_indicators


def example_4_risk_analysis(df):
    """
    Exemple 4 : Analyse du risque
    """
    print("\n" + "="*50)
    print("EXEMPLE 4 : Analyse du Risque")
    print("="*50 + "\n")
    
    # Calculer les rendements
    prices = df['Close']
    returns = create_returns(df, column='Close')
    
    # Indicateurs de risque individuels
    volatility = calculate_volatility(returns, periods=252)
    print(f"Volatilité annualisée: {volatility:.4f} ({volatility*100:.2f}%)")
    
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    print(f"Ratio de Sharpe: {sharpe:.4f}")
    
    var_95 = calculate_var(returns, confidence_level=0.95)
    print(f"Value at Risk (95%): {var_95:.6f}")
    
    max_dd, peak_date, trough_date = calculate_max_drawdown(prices)
    print(f"Maximum Drawdown: {max_dd:.4f} ({max_dd*100:.2f}%)")
    print(f"  - Du: {peak_date.date()}")
    print(f"  - Au: {trough_date.date()}")
    
    # Utiliser la classe RiskAnalyzer complète
    print("\nAnalyse complète avec RiskAnalyzer...")
    analyzer = RiskAnalyzer(prices, risk_free_rate=0.02)
    risk_metrics = analyzer.get_risk_metrics()
    
    print("\nMétriques de Risque:")
    for metric, value in risk_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  - {metric}: {value:.6f}")
        else:
            print(f"  - {metric}: {value}")
    
    return risk_metrics


def example_5_train_test_split(df):
    """
    Exemple 5 : Diviser train/test
    """
    print("\n" + "="*50)
    print("EXEMPLE 5 : Train/Test Split")
    print("="*50 + "\n")
    
    train, test = train_test_split_timeseries(df, test_size=0.2)
    
    print(f"Total: {len(df)} observations")
    print(f"Train: {len(train)} observations ({len(train)/len(df)*100:.1f}%)")
    print(f"Test: {len(test)} observations ({len(test)/len(df)*100:.1f}%)")
    
    print(f"\nPériode d'entraînement: {train.index[0].date()} à {train.index[-1].date()}")
    print(f"Période de test: {test.index[0].date()} à {test.index[-1].date()}")
    
    return train, test


def example_6_arima_model(train_data, test_data):
    """
    Exemple 6 : Modèle ARIMA
    """
    print("\n" + "="*50)
    print("EXEMPLE 6 : Modèle ARIMA")
    print("="*50 + "\n")
    
    try:
        # Entraîner le modèle
        arima = ARIMAModel(order=(1, 1, 1))
        print("Entraînement du modèle ARIMA(1,1,1)...")
        arima.fit(train_data['Close'])
        print("✓ Modèle ARIMA entraîné")
        
        # Faire des prédictions
        predictions = arima.predict(steps=len(test_data))
        print(f"✓ {len(test_data)} prédictions générées")
        
        # Évaluer
        metrics = arima.evaluate(test_data['Close'].values, predictions)
        print(f"\nPerformances:")
        for metric, value in metrics.items():
            print(f"  - {metric}: {value:.6f}")
        
        return predictions
        
    except ImportError as e:
        print(f"⚠ Erreur: {e}")
        print("Installation: pip install statsmodels")
        return None


def example_7_feature_engineering_and_ml(df):
    """
    Exemple 7 : Feature engineering et Random Forest
    """
    print("\n" + "="*50)
    print("EXEMPLE 7 : Feature Engineering et ML")
    print("="*50 + "\n")
    
    from src.models import create_lagged_features_for_ml
    
    # Créer les features (lags)
    X, y = create_lagged_features_for_ml(df['Close'], n_lags=5, n_future=1)
    print(f"Features créées:")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    
    # Split train/test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nTrain/Test split:")
    print(f"  - Train: {X_train.shape[0]} samples")
    print(f"  - Test: {X_test.shape[0]} samples")
    
    # Entraîner Random Forest
    rf = RandomForestRegressor(n_estimators=50, max_depth=15)
    print(f"\nEntraînement du Random Forest...")
    rf.fit(X_train, y_train)
    print(f"✓ Modèle entraîné")
    
    # Prédictions
    y_pred = rf.predict(X_test)
    metrics = rf.evaluate(y_test, y_pred)
    
    print(f"\nPerformances:")
    for metric, value in metrics.items():
        print(f"  - {metric}: {value:.6f}")
    
    return rf, X_test, y_pred


def example_8_visualizations(df, returns, y_true, y_pred):
    """
    Exemple 8 : Visualisations
    """
    print("\n" + "="*50)
    print("EXEMPLE 8 : Visualisations")
    print("="*50 + "\n")
    
    # Série de prix
    fig1 = plot_price_series(df, column='Close', title='Prix de l\'Or')
    print("✓ Graphique de prix créé")
    
    # Rendements cumulatifs
    fig2 = plot_cumulative_returns(returns, figsize=(15, 6))
    print("✓ Graphique de rendements cumulatifs créé")
    
    # Prédictions
    if y_true is not None and y_pred is not None:
        fig3 = plot_predictions(y_true, y_pred, model_name='Random Forest')
        print("✓ Graphique de prédictions créé")
    
    print("\nGraphiques générés (à afficher avec plt.show())")


def main():
    """
    Fonction principale pour exécuter tous les exemples
    """
    print("\n" + "="*60)
    print("DÉMONSTRATION COMPLÈTE : ANALYSE QUANTITATIVE")
    print("="*60)
    
    # Exemple 1 : Charger les données
    df = example_1_load_and_explore()
    
    # Exemple 2 : Préparation
    df, returns, normalized, scaler = example_2_data_preprocessing(df)
    
    # Exemple 3 : Indicateurs techniques
    df_indicators = example_3_technical_indicators(df)
    
    # Exemple 4 : Analyse du risque
    risk_metrics = example_4_risk_analysis(df)
    
    # Exemple 5 : Split train/test
    train, test = example_5_train_test_split(df)
    
    # Exemple 6 : ARIMA
    arima_preds = example_6_arima_model(train, test)
    
    # Exemple 7 : Random Forest
    rf_model, X_test, y_pred_rf = example_7_feature_engineering_and_ml(df)
    
    # Exemple 8 : Visualisations
    example_8_visualizations(df, returns, df[len(df)-len(y_pred_rf):]['Close'].values, y_pred_rf)
    
    print("\n" + "="*60)
    print("✅ DÉMONSTRATION COMPLÉTÉE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
