# Changelog

Tous les changements notables de ce projet sont documentés dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/),
et ce projet adhère à [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-08-17

### Added
- ✨ **Pipeline complet d'analyse quantitative**
  - Préparation et nettoyage des données
  - Calcul d'indicateurs techniques (RSI, MACD, Bollinger Bands, etc.)
  - Gestion du risque (VaR, Sharpe Ratio, Sortino Ratio, etc.)
  - Modèles de prévision (ARIMA, LSTM, Random Forest, XGBoost)
  - Visualisations avancées

- 📊 **Fonctionnalités de Data Processing**
  - `load_financial_data()` : Charger les données CSV
  - `handle_missing_values()` : Gérer les valeurs manquantes
  - `normalize_data()` : Normaliser avec StandardScaler/MinMaxScaler
  - `train_test_split_timeseries()` : Split temporel
  - `create_lagged_features()` : Features pour ML

- 📈 **Indicateurs Techniques**
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bandes de Bollinger
  - Moyennes mobiles (SMA, EMA)
  - ATR (Average True Range)
  - Stochastique
  - OBV (On-Balance Volume)
  - ROC (Rate of Change)
  - Fonction complète `create_technical_indicators()`

- 💰 **Gestion du Risque**
  - VaR (Value at Risk) - Historique, Paramétrique, Monte Carlo
  - CVaR (Conditional Value at Risk)
  - Ratio de Sharpe
  - Ratio de Sortino
  - Maximum Drawdown
  - Ratio d'Information
  - Ratio de Calmar
  - Beta et Alpha (CAPM)
  - Classe `RiskAnalyzer` complète

- 🤖 **Modèles de Prévision**
  - **ARIMA** : Modèle statistique classique
  - **LSTM** : Réseau de neurones récurrent
  - **Random Forest** : Ensemble learning
  - **XGBoost** : Gradient Boosting
  - Walk-forward validation
  - Fonctions d'évaluation (MAE, RMSE, MAPE)

- 📉 **Visualisations**
  - Graphiques de séries temporelles
  - Indicateurs techniques
  - Prévisions vs réalité
  - Distribution des rendements
  - Corrélations
  - Q-Q plots
  - Rendements cumulatifs
  - Analyse des résidus

- 📚 **Documentation**
  - README complet et professionnel
  - Guide détaillé des concepts
  - Guide de contribution
  - Configuration CI/CD GitHub Actions
  - pyproject.toml et requirements.txt

- 🔬 **Dataset**
  - 12 ans de données boursières (OHLCV)
  - Format CSV structuré
  - Données nettoyées et validées

### Changed
- Restructuration complète du projet avec organisation professionnelle
- Amélioration de la documentation existante

### Security
- Ajout de fichier .gitignore complet
- Configuration de licence MIT
- Standards de code avec black, flake8, pylint

## [0.1.0] - Date Antérieure

### Added
- Version initiale du notebook Jupyter
- Données brutes du marché boursier

---

## Guide de Versioning

Ce projet suit [Semantic Versioning](https://semver.org/) :

- **MAJOR** (1.x.x) : Changements incompatibles
- **MINOR** (x.1.x) : Nouvelles fonctionnalités compatibles
- **PATCH** (x.x.1) : Corrections de bugs

## Roadmap

### À Venir (v1.1.0)
- [ ] Support pour plusieurs actifs simultanément
- [ ] Portfolio optimization
- [ ] Stratégies de trading complètes
- [ ] Backtesting framework complet
- [ ] API REST pour les prédictions

### À Court Terme (v1.2.0)
- [ ] Modèles Transformer pour séries temporelles
- [ ] Ensemble de modèles (stacking, voting)
- [ ] Support de données temps réel (API)
- [ ] Dashboard interactif (Dash/Streamlit)

### À Moyen Terme (v2.0.0)
- [ ] Module de trading automatisé
- [ ] Intégration avec brokers
- [ ] Support de crypto-monnaies
- [ ] Machine learning avancé (AutoML)

---

## Notes de Migration

### De 0.x.x à 1.0.0
- Structure du projet entièrement réorganisée
- Code du notebook extrait en modules Python
- API publique stabilisée
- Documentation considérablement améliorée

---

## Remerciements

Merci à tous les contributeurs et utilisateurs qui aident à améliorer ce projet !

---

**Dernière mise à jour** : 17 Août 2026
