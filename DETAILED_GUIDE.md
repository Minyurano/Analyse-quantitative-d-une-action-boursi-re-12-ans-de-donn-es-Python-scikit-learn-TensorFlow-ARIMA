# Guide Détaillé - Analyse Quantitative de Séries Financières

## Table des Matières
1. [Introduction](#introduction)
2. [Principes Fondamentaux](#principes-fondamentaux)
3. [Préparation des Données](#préparation-des-données)
4. [Indicateurs Techniques](#indicateurs-techniques)
5. [Gestion du Risque](#gestion-du-risque)
6. [Modèles de Prévision](#modèles-de-prévision)
7. [Validation et Backtesting](#validation-et-backtesting)
8. [Ressources](#ressources)

## Introduction

Ce guide fournit une explication détaillée des concepts et techniques utilisés dans l'analyse quantitative de séries temporelles financières.

### Qu'est-ce que l'Analyse Quantitative ?

L'analyse quantitative utilise des modèles mathématiques et statistiques pour :
- **Analyser** les données historiques
- **Prédire** les mouvements futurs
- **Optimiser** les stratégies d'investissement
- **Gérer** le risque efficacement

## Principes Fondamentaux

### 1. Hypothèse de Marché Efficient
Le marché reflète toute l'information disponible, rendant la prédiction difficile mais pas impossible via :
- L'analyse technique (patterns historiques)
- L'analyse fondamentale (valeur intrinsèque)
- Les modèles statistiques

### 2. Rendements et Volatilité

**Rendement Simple** :
```
Rendement = (Prix_t - Prix_t-1) / Prix_t-1
```

**Rendement Logarithmique** (préféré) :
```
Log Return = ln(Prix_t / Prix_t-1)
```

**Volatilité Annualisée** :
```
σ_annuel = σ_quotidien × √252
```

Où 252 = nombre de jours de trading par an

### 3. Stationnarité
Une série est **stationnaire** si sa moyenne et variance sont constantes dans le temps.

**Test Augmented Dickey-Fuller (ADF)** :
```python
from statsmodels.tsa.stattools import adfuller

result = adfulter(series)
print(f"ADF Statistic: {result[0]}")
print(f"P-value: {result[1]}")
# p-value < 0.05 → Série stationnaire
```

## Préparation des Données

### 1. Nettoyage des Données

```python
from src.data_processing import *

# Charger les données
df = load_financial_data('data/gold_stock.csv')

# Gérer les valeurs manquantes
df = handle_missing_values(df, method='interpolate')

# Vérifier la qualité
print(df.describe())
```

### 2. Normalisation

```python
from sklearn.preprocessing import StandardScaler

# Normalisation Z-score
scaler = StandardScaler()
normalized = scaler.fit_transform(df[['Close']])

# Inverse transformation
original = scaler.inverse_transform(normalized)
```

### 3. Features Temporelles

```python
df['Day_of_Week'] = df.index.dayofweek
df['Month'] = df.index.month
df['Year'] = df.index.year
df['Quarter'] = df.index.quarter
```

## Indicateurs Techniques

### 1. Moyennes Mobiles

**Simple Moving Average (SMA)** :
```
SMA_n = (P_t + P_t-1 + ... + P_t-n+1) / n
```

**Exponential Moving Average (EMA)** :
- Donne plus de poids aux données récentes
- Formule : EMA_t = P_t × α + EMA_t-1 × (1-α), où α = 2/(n+1)

```python
from src.technical_indicators import calculate_sma, calculate_ema

sma = calculate_sma(df['Close'], window=20)
ema = calculate_ema(df['Close'], window=12)
```

### 2. Relative Strength Index (RSI)

Mesure l'intensité des changements de prix :
```
RSI = 100 - (100 / (1 + RS))
RS = Moyenne des Gains / Moyenne des Pertes
```

**Interprétation** :
- RSI > 70 : Suracheté (possible correction baissière)
- RSI < 30 : Survendu (possible rebond haussier)
- Période typique : 14 jours

```python
rsi = calculate_rsi(df['Close'], window=14)
```

### 3. MACD (Moving Average Convergence Divergence)

Indicateur de momentum et tendance :
```
MACD = EMA_12 - EMA_26
Signal = EMA_9(MACD)
Histogram = MACD - Signal
```

**Signaux** :
- MACD > Signal : Signal haussier
- MACD < Signal : Signal baissier
- Histogram > 0 : Momentum haussier croissant

### 4. Bandes de Bollinger

Mesure la volatilité et les niveaux extrêmes :
```
Middle Band = SMA_20
Upper Band = SMA_20 + 2×σ
Lower Band = SMA_20 - 2×σ
```

**Signaux** :
- Prix touche la bande supérieure : Peut être suracheté
- Prix touche la bande inférieure : Peut être survendu
- Bande étroite : Volatilité faible (explosion imminente possible)

## Gestion du Risque

### 1. Value at Risk (VaR)

Estime la perte maximale probable :

**VaR Historique** (95%) :
- Prendre le pire 5% des rendements passés
- Simple mais suppose que le passé se répète

**VaR Paramétrique** :
```
VaR = μ + z_α × σ
où z_α = z-score pour le niveau de confiance
```

Exemple : Si VaR(95%) = -2%, il y a 95% de chance que la perte n'excède pas 2%.

### 2. Ratio de Sharpe

Mesure le rendement par unité de risque (rendement ajusté au risque) :

```
Sharpe Ratio = (Retour - Taux Sans Risque) / Volatilité
```

**Interprétation** :
- \> 1.0 : Bon rendement ajusté au risque
- \> 2.0 : Très bon rendement ajusté au risque
- \< 0 : Rendement inférieur au taux sans risque

```python
from src.risk_management import calculate_sharpe_ratio

sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
```

### 3. Maximum Drawdown

La plus grande baisse cumulée depuis un pic :

```
Drawdown = (Valeur_Actuelle - Pic_Précédent) / Pic_Précédent
```

Exemple : Si le portefeuille passe de 100€ à 90€, le drawdown = -10%.

## Modèles de Prévision

### 1. ARIMA (AutoRegressive Integrated Moving Average)

**Modèle statistique classique** pour les séries temporelles.

**Paramètres (p, d, q)** :
- **p** : Nombre de termes autorégressifs (AR)
- **d** : Degré de différentiation (I)
- **q** : Nombre de termes de moyenne mobile (MA)

**Étapes** :
1. Tester la stationnarité (ADF test)
2. Différencier si nécessaire pour rendre stationnaire
3. Trouver les paramètres (p, q) optimaux (ACF, PACF)
4. Entraîner et valider

```python
from src.models import ARIMAModel

# Créer et entraîner le modèle
arima = ARIMAModel(order=(1, 1, 1))
arima.fit(train_data)

# Prédire
predictions = arima.predict(steps=30)
```

### 2. LSTM (Long Short-Term Memory)

**Réseau de neurones récurrent** pour capturer les dépendances long terme.

**Architecture** :
```
Données → LSTM Layer → Dropout → LSTM Layer → Dense → Sortie
```

**Avantages** :
- Capture les patterns longs terme
- Flexible et puissant
- Gère bien les dépendances temporelles

**Inconvénients** :
- Nécessite beaucoup de données
- Coûteux en computation
- Moins interprétable

```python
from src.models import LSTMModel

# Créer et entraîner
lstm = LSTMModel(lookback=60, lstm_units=50, epochs=50)
lstm.fit(normalized_data)

# Prédire
predictions = lstm.predict(recent_data, steps=30)
```

### 3. Random Forest

**Ensemble learning** : Utilise plusieurs arbres de décision.

**Avantages** :
- Robuste aux outliers
- Capture les non-linéarités
- Pas de normalisation requise

**Inconvénients** :
- Moins bon pour les séries temporelles pures
- Nécessite features engineered

```python
from src.models import RandomForestRegressor

# Créer des features (lags)
X, y = create_lagged_features_for_ml(prices, n_lags=5)

# Entraîner
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

# Prédire
predictions = rf.predict(X_test)
```

## Validation et Backtesting

### 1. Walk-Forward Validation

Simule un scénario réel d'investissement :

```
1. Entraîner sur période 1
2. Tester sur période 2
3. Avancer d'un pas
4. Entraîner sur période 2
5. Tester sur période 3
... répéter
```

Reproduit comment on investirait réellement :
- Pas de lookahead bias
- Plus réaliste que train/test simple

```python
from src.models import walk_forward_validation

results = walk_forward_validation(
    series=prices,
    train_size=500,
    test_size=50,
    model_class=ARIMAModel
)
```

### 2. Métriques de Performance

**MAE (Mean Absolute Error)** :
```
MAE = |Actual - Predicted| moyenne
```
Plus bas = meilleur

**RMSE (Root Mean Squared Error)** :
```
RMSE = √((Actual - Predicted)² moyenne)
```
Pénalise plus les grandes erreurs

**MAPE (Mean Absolute Percentage Error)** :
```
MAPE = |Actual - Predicted| / Actual moyenne en %
```
Plus facile à interpréter (en %)

### 3. Analyse des Résidus

Les résidus doivent être :
- **Centrés sur 0** : Pas de biais
- **Distribution normale** : Pas d'hétéroscédasticité
- **Non-autocorrélés** : Pas de pattern manqué
- **Homoscédastiques** : Variance constante

```python
from src.visualization import plot_residuals

plot_residuals(y_true, y_pred)
```

## Ressources

### Livres
- "Advances in Financial Machine Learning" - Marcos López de Prado
- "Machine Learning for Algorithmic Trading" - Stefan Jansen
- "Time Series Analysis" - Hamilton

### Cours en Ligne
- Coursera: Financial Engineering Specialization
- edX: Computational Investing
- Udacity: AI for Trading Nanodegree

### Outils
- TA-Lib : Indicateurs techniques optimisés
- Backtrader : Framework de backtesting
- Zipline : Backtesting engine d'Quantopian

### Documentation
- [Pandas Documentation](https://pandas.pydata.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Statsmodels Documentation](https://www.statsmodels.org/)

## Checklist de Bonnes Pratiques

- [ ] Données nettoyées et validées
- [ ] Stationnarité vérifiée
- [ ] Features normalisées/standardisées
- [ ] Pas de lookahead bias
- [ ] Walk-forward ou time-series cross-validation
- [ ] Résidus analyzed
- [ ] Risque géré et mesuré
- [ ] Résultats robustes (hors sample)
- [ ] Slippage et commissions considérées
- [ ] Stratégie documentée

---

Pour plus d'informations, consultez le README principal ou ouvrez une [issue](https://github.com/Minyurano/Analyse-quantitative-d-une-action-boursi-re/issues).
