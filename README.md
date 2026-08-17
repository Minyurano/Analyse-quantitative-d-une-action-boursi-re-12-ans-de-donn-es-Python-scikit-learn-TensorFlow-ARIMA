# Analyse Quantitative et Prévision de Séries Financières

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📊 Description

Pipeline complet d'analyse quantitative et de prévision de séries temporelles financières basé sur **12 ans de données boursières**. Ce projet combine :

- **Analyse Technique Avancée** : Indicateurs techniques (RSI, MACD)
- **Gestion du Risque** : Calcul de la Value at Risk (VaR), ratio de Sharpe
- **Modèles Statistiques** : ARIMA pour la prévision de séries temporelles
- **Machine Learning** : Random Forest, Gradient Boosting
- **Deep Learning** : LSTM pour la prévision de séries temporelles
- **Validation Stricte** : Cross-validation chronologique (walk-forward validation)

## 🎯 Objectifs

✅ Analyser les tendances long terme du marché boursier  
✅ Évaluer le risque des portefeuilles  
✅ Comparer les performances prédictives de différents modèles  
✅ Fournir des insights exploitables pour la décision d'investissement  

## 📁 Structure du Projet

```
.
├── README.md                          # Documentation principale
├── requirements.txt                   # Dépendances Python
├── LICENSE                            # Licence MIT
├── .gitignore                         # Fichiers à ignorer
│
├── notebooks/                         # Notebooks Jupyter
│   └── analyse-quantitative-et-prevision-de-series-financ.ipynb
│
├── data/                              # Données brutes
│   └── gold_stock.csv                 # 12 ans de données boursières (Prix de l'Or)
│
├── src/                               # Code Python réutilisable
│   ├── __init__.py
│   ├── data_processing.py             # Préparation et nettoyage des données
│   ├── technical_indicators.py        # Indicateurs techniques (RSI, MACD, etc.)
│   ├── risk_management.py             # Calculs VaR, Sharpe, etc.
│   ├── models.py                      # Modèles (ARIMA, RF, LSTM)
│   └── visualization.py               # Visualisation des résultats
│
└── results/                           # Résultats et visualisations
    ├── predictions/                   # Résultats de prévisions
    ├── analysis/                      # Analyses statistiques
    └── plots/                         # Graphiques générés
```

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.8+
- pip ou conda

### Installation

1. **Cloner le repository**
```bash
git clone https://github.com/Minyurano/Analyse-quantitative-d-une-action-boursi-re.git
cd Analyse-quantitative-d-une-action-boursi-re
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Lancer le notebook**
```bash
jupyter notebook notebooks/
```

## 📊 Dataset

**Fichier** : `data/gold_stock.csv`

- **Période** : 12 ans de données historiques
- **Fréquence** : Quotidienne
- **Colonnes** : Date, Open, High, Low, Close, Volume, Adj Close
- **Taille** : ~2,900 observations

```bash
head -5 data/gold_stock.csv
# Output:
# Date,Open,High,Low,Close,Volume,Adj Close
# 2011-01-01,1405.50,1410.25,1400.00,1405.50,150000,1405.50
# 2011-01-02,1408.00,1415.50,1408.00,1410.25,180000,1410.25
```

## 🔧 Méthodologie

### 1️⃣ Préparation des Données
- Nettoyage des valeurs manquantes
- Normalisation et standardisation
- Création de features temporelles
- Split train/test chronologique

### 2️⃣ Indicateurs Techniques
```python
# Exemples calculés
- RSI (Relative Strength Index) : Force du mouvement
- MACD (Moving Average Convergence Divergence) : Momentum
- Bandes de Bollinger : Volatilité
- SMA/EMA : Moyennes mobiles
```

### 3️⃣ Gestion du Risque
```python
# Métriques calculées
- Value at Risk (VaR) : Perte potentielle maximale
- Ratio de Sharpe : Rendement par unité de risque
- Volatilité : Écart-type des rendements
- Drawdown : Perte maximale cumulée
```

### 4️⃣ Modèles de Prévision

| Modèle | Description | Avantages |
|--------|-------------|-----------|
| **ARIMA** | Statistique classique | Interprétabilité, rapidité |
| **Random Forest** | Ensemble learning | Robustesse, non-linéarité |
| **LSTM** | Deep learning | Captation de patterns longs terme |
| **XGBoost** | Gradient Boosting | Performance excellente |

### 5️⃣ Validation
- **Walk-Forward Validation** : Reproduit un scénario réel de trading
- **Time Series Cross-Validation** : Respecte l'ordre temporel
- **Métriques** : MAE, RMSE, MAPE, R²

## 📈 Résultats

Les résultats de l'analyse sont disponibles dans le dossier `results/` :

### Prévisions
- Fichier CSV avec les prédictions du modèle
- Comparaison avec les valeurs réelles
- Calcul des erreurs (MAE, RMSE, MAPE)

### Visualisations
- Graphiques de séries temporelles
- Courbes de prévisions vs réalité
- Distribution des résidus
- Analyse des indicateurs techniques

### Statistiques
- Résumé descriptif des données
- Corrélations between features
- Performance des modèles

## 📦 Dépendances Principales

```
pandas>=1.3.0           # Manipulation de données
numpy>=1.20.0           # Calculs numériques
scikit-learn>=0.24.0    # Machine Learning
tensorflow>=2.6.0       # Deep Learning / LSTM
statsmodels>=0.12.0     # Modèles statistiques (ARIMA)
xgboost>=1.4.0          # Gradient Boosting
matplotlib>=3.3.0       # Visualisation
seaborn>=0.11.0         # Visualisation avancée
jupyter>=1.0.0          # Notebooks
```

Voir `requirements.txt` pour la liste complète.

## 💡 Utilisation

### En tant que Jupyter Notebook
```bash
jupyter notebook notebooks/analyse-quantitative-et-prevision-de-series-financ.ipynb
```

### Importer les modules
```python
from src.technical_indicators import calculate_rsi, calculate_macd
from src.risk_management import calculate_var, calculate_sharpe
from src.models import train_arima, train_lstm

# Charger et traiter les données
import pandas as pd
df = pd.read_csv('data/gold_stock.csv')

# Calculer les indicateurs
rsi = calculate_rsi(df['Close'])
macd = calculate_macd(df['Close'])

# Prédire avec LSTM
predictions = train_lstm(df['Close'], lookback=60)
```

## 🎓 Concepts Clés Expliqués

### ARIMA (AutoRegressive Integrated Moving Average)
Modèle statistique pour les séries temporelles non-stationnaires. Combine :
- **AR** : Régression autorégrégressive
- **I** : Intégration (différentiation)
- **MA** : Moyenne mobile

### LSTM (Long Short-Term Memory)
Réseau de neurones récurrent spécialisé dans les dépendances long terme.
- Dispose des portes d'oubli (forget gates)
- Excellent pour la prévision de séries temporelles

### Value at Risk (VaR)
Estimation de la perte maximale probable sur une période donnée avec un intervalle de confiance.
- **VaR(95%)** : Il y a 95% de chance que les pertes n'excèdent pas cette valeur

### Ratio de Sharpe
Mesure le rendement excédentaire par unité de risque.
```
Sharpe Ratio = (Retour Portefeuille - Taux sans risque) / Volatilité
```
- Ratio > 1.0 : Bon rendement ajusté au risque
- Ratio > 2.0 : Très bon rendement ajusté au risque

## 📊 Interprétation des Résultats

### Indicateurs Techniques
- **RSI > 70** : Signal de suracheté
- **RSI < 30** : Signal de survente
- **MACD positif** : Signal haussier
- **MACD négatif** : Signal baissier

### Performance du Modèle
- **RMSE bas** : Prédictions proches des valeurs réelles
- **MAPE < 5%** : Excellente performance
- **Ratio de Sharpe élevé** : Bonne stratégie ajustée au risque

## 🔗 Ressources Complémentaires

- [Documentation scikit-learn](https://scikit-learn.org/)
- [Documentation TensorFlow/Keras](https://www.tensorflow.org/)
- [Guide ARIMA](https://www.statsmodels.org/stable/tsa.html)
- [Indicateurs Techniques](https://en.wikipedia.org/wiki/Technical_indicator)
- [Investopedia - Financial Analysis](https://www.investopedia.com/)

## 📝 Méthodologie Recommandée pour Utiliser ce Projet

1. **Explorer les données** → Comprendre les patterns
2. **Calculer les indicateurs** → Identifier les signaux
3. **Évaluer le risque** → Comprendre l'exposition
4. **Entraîner les modèles** → Comparer les approches
5. **Valider les résultats** → Walk-forward validation
6. **Analyser les performances** → Métriques et visualisations
7. **Décision d'investissement** → Actions basées sur les insights

## ⚠️ Disclaimer

Ce projet est à **fins éducatives et de recherche uniquement**. Les prévisions ne constituent pas des recommandations d'investissement. 

**Avertissement de risque** :
- Les performances passées ne garantissent pas les résultats futurs
- Les investissements boursiers comportent des risques
- Consultez un professionnel financier avant de prendre des décisions d'investissement

## 🤝 Contribution

Les contributions sont bienvenues ! Pour contribuer :

1. Fork le repository
2. Créer une branche pour votre feature (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

### Guidelines
- Respect du style de code (PEP 8)
- Tests unitaires pour les nouvelles fonctionnalités
- Documentation claire et concise
- Commits messages descriptifs

## 📄 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👤 Auteur

**Minyurano**
- GitHub: [@Minyurano](https://github.com/Minyurano)

## 🙏 Remerciements

- Données : Yahoo Finance, Quandl
- Bibliothèques : scikit-learn, TensorFlow, statsmodels
- Communauté : Stack Overflow, Kaggle

## 📮 Contact & Support

Pour les questions ou les rapports de bugs :
- Ouvrir une [Issue](https://github.com/Minyurano/Analyse-quantitative-d-une-action-boursi-re/issues)
- Email : [votre-email@exemple.com]

---

**Dernière mise à jour** : Août 2026  
**Statut** : ✅ Production-ready
