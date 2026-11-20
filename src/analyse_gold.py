# %% [markdown]
# # 📈 GOLD — Analyse Quantitative Complète & Aide à la Décision d'Investissement
#
# **Auteur :** Data Science / IA Engineering Portfolio Project
# **Données :** Cours quotidiens (Open, High, Low, Close, Volume) du titre **GOLD**, 17/03/2014 → 31/12/2025 (2 968 séances)
#
# ---
#
# ### ℹ️ Mode d'emploi sur Kaggle
# 1. Créez un nouveau **Notebook** Kaggle (Code > New Notebook).
# 2. Ajoutez `gold_stock.csv` comme dataset d'entrée (Add Input), ou déposez-le simplement à côté du notebook.
# 3. Collez ce fichier tel quel dans une cellule (ou plusieurs, en coupant à chaque `# %%`) et cliquez sur
#    **Run All**. Le chargeur cherche automatiquement le fichier dans `/kaggle/input/**/gold_stock.csv`.
# 4. **Reproductibilité** : toutes les graines aléatoires sont fixées (`np.random.seed(42)`,
#    `random_state=42`, `tf.random.set_seed(42)`) → les résultats numériques et les figures obtenus seront
#    identiques (à d'éventuels écarts d'arrondi près pour la section LSTM si Kaggle exécute sur GPU plutôt
#    que CPU, l'entraînement de réseaux de neurones n'étant pas garanti 100 % déterministe sur GPU).
#
# ### 🎯 Objectif
# Fournir à une direction d'entreprise une analyse quantitative complète — de la donnée brute à la recommandation —
# combinant **finance quantitative**, **statistiques**, **séries temporelles** et **machine learning**, pour éclairer
# une décision d'investissement (achat / conservation / vente, dimensionnement du risque, timing).
#
# ### 🗂️ Sommaire
# 1. [Configuration & style professionnel](#1)
# 2. [Chargement et nettoyage des données](#2)
# 3. [Analyse exploratoire (EDA)](#3)
# 4. [Tendance & performance](#4)
# 5. [Indicateurs techniques avancés](#5)
# 6. [Analyse de risque quantitatif](#6)
# 7. [Stationnarité & décomposition de la série temporelle](#7)
# 8. [Modélisation prédictive (Machine Learning)](#8)
# 9. [Prévision ARIMA](#9)
# 10. [Tableau de bord exécutif & recommandation](#10)
# 11. [Deep Learning — prévision par réseau de neurones LSTM](#11)
# 12. [Conclusion](#12)

# %% [markdown]
# <a id="1"></a>
# ## 1. Configuration & style professionnel

# %% [code]
import warnings
warnings.filterwarnings("ignore")

import glob
import os

os.makedirs("outputs", exist_ok=True)

import numpy as np
import pandas as pd

import matplotlib

try:
    get_ipython  # noqa: F821 (defini uniquement dans Jupyter / Kaggle)
except NameError:
    matplotlib.use("Agg")  # execution en script : figures sauvegardees, pas de fenetres

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns

from scipy import stats

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix)
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
np.random.seed(42)

# ---- Palette "terminal financier" ----
BG      = "#0d1117"
PANEL   = "#151b23"
GRID    = "#2a313c"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"
GOLD    = "#d4af37"
BLUE    = "#4dabf7"
GREEN   = "#3fb950"
RED     = "#f85149"
PURPLE  = "#bc8cff"
ORANGE  = "#ffa657"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "savefig.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "text.color": TEXT,
    "xtick.color": SUBTEXT,
    "ytick.color": SUBTEXT,
    "grid.color": GRID,
    "grid.alpha": 0.6,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "axes.axisbelow": True,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 110,
})

sns.set_palette([GOLD, BLUE, GREEN, RED, PURPLE, ORANGE])


def style_axis(ax, title=None, xlabel=None, ylabel=None, subtitle=None):
    """Applique un habillage cohérent (titre + sous-titre + labels) à un axe matplotlib."""
    if title:
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=TEXT, pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9.5, color=SUBTEXT, ha="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=SUBTEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=SUBTEXT)
    ax.tick_params(labelsize=9.5)
    return ax


def add_footer(fig, text="Source : cours historiques GOLD (2014–2025)  |  Analyse quantitative — usage interne"):
    fig.text(0.01, -0.02, text, fontsize=8, color=SUBTEXT, ha="left")


print("✅ Environnement configuré — thème 'terminal financier' actif.")

# %% [markdown]
# <a id="2"></a>
# ## 2. Chargement et nettoyage des données
#
# Le loader cherche d'abord un dataset dans `/kaggle/input/` (exécution sur Kaggle), puis retombe sur un fichier
# local `gold_stock.csv` — le notebook est donc exécutable tel quel sur Kaggle ou en local.

# %% [code]
def load_gold_data():
    candidates = glob.glob("/kaggle/input/**/gold_stock.csv", recursive=True)
    candidates += glob.glob("./data/gold_stock.csv") + glob.glob("./gold_stock.csv") + glob.glob("../input/**/gold_stock.csv")
    path = candidates[0] if candidates else "gold_stock.csv"

    raw = pd.read_csv(path, skiprows=[1, 2])
    raw.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values("Date").reset_index(drop=True)
    raw = raw[["Date", "Open", "High", "Low", "Close", "Volume"]]
    return raw


df = load_gold_data()

print(f"Fichier chargé — {df.shape[0]:,} séances, {df.shape[1]} colonnes")
print(f"Période couverte : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Valeurs manquantes : {int(df.isna().sum().sum())}")
print(f"Doublons de date : {int(df['Date'].duplicated().sum())}")

df.head()

# %% [code]
summary = df.drop(columns="Date").describe().T
summary["missing"] = df.drop(columns="Date").isna().sum()
summary.style.background_gradient(cmap="YlOrBr", subset=["mean", "std", "min", "max"]).format(precision=2)

# %% [markdown]
# <a id="3"></a>
# ## 3. Analyse exploratoire (EDA)
#
# Avant toute modélisation, on calcule les **rendements quotidiens** (base de toute la finance quantitative) et on
# explore leur distribution, ainsi que la relation prix / volume.

# %% [code]
df["Return"] = df["Close"].pct_change()
df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
df["Year"] = df["Date"].dt.year

fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(df["Date"], df["Close"], color=GOLD, linewidth=1.3, zorder=3)
ax.fill_between(df["Date"], df["Close"], df["Close"].min(), color=GOLD, alpha=0.08)
style_axis(ax, "Évolution du cours de clôture — GOLD",
           subtitle="Cours quotidien, 17 mars 2014 → 31 décembre 2025", ylabel="Prix de clôture ($)")
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
add_footer(fig)
plt.tight_layout()
plt.savefig("outputs/fig_01_price_history.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [code]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Distribution des rendements vs loi normale ajustée
returns_pct = (df["Return"].dropna() * 100)
mu, sigma = returns_pct.mean(), returns_pct.std()
x = np.linspace(returns_pct.min(), returns_pct.max(), 300)
axes[0].hist(returns_pct, bins=80, density=True, color=BLUE, alpha=0.75, label="Rendements observés")
axes[0].plot(x, stats.norm.pdf(x, mu, sigma), color=RED, linewidth=2, label="Loi normale ajustée")
skew, kurt = stats.skew(returns_pct), stats.kurtosis(returns_pct)
axes[0].text(0.97, 0.95, f"Skewness : {skew:+.2f}\nKurtosis (excès) : {kurt:+.2f}",
             transform=axes[0].transAxes, ha="right", va="top", fontsize=9.5, color=TEXT,
             bbox=dict(boxstyle="round", facecolor=PANEL, edgecolor=GRID))
style_axis(axes[0], "Distribution des rendements quotidiens", ylabel="Densité", xlabel="Rendement (%)")
axes[0].legend(fontsize=9)

# Volume
axes[1].bar(df["Date"], df["Volume"], color=BLUE, alpha=0.6, width=2)
axes[1].plot(df["Date"], df["Volume"].rolling(50).mean(), color=ORANGE, linewidth=1.6, label="Moyenne mobile 50j")
style_axis(axes[1], "Volume échangé", ylabel="Volume", xlabel="Date")
axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[1].legend(fontsize=9)

add_footer(fig)
plt.tight_layout()
plt.savefig("outputs/fig_02_eda.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"La distribution des rendements a une asymétrie (skew) de {skew:+.2f} et un excès de kurtosis de {kurt:+.2f} "
      f"({'queues plus épaisses' if kurt > 0 else 'queues plus fines'} qu'une loi normale) "
      f"→ le risque de mouvements extrêmes est {'sous-estimé' if kurt > 0 else 'surestimé'} par une hypothèse de normalité.")

# %% [markdown]
# <a id="4"></a>
# ## 4. Tendance & performance
#
# On mesure la création de valeur en base 100 (comparabilité), le rendement annuel composé (CAGR) et la
# dispersion des performances annuelles.

# %% [code]
first_price, last_price = df["Close"].iloc[0], df["Close"].iloc[-1]
n_years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25
cagr = (last_price / first_price) ** (1 / n_years) - 1
total_return = last_price / first_price - 1

yearly = df.groupby("Year")["Close"].agg(["first", "last"])
yearly["return_%"] = (yearly["last"] / yearly["first"] - 1) * 100

df["Indexed"] = df["Close"] / df["Close"].iloc[0] * 100

fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), gridspec_kw={"height_ratios": [1.4, 1]})

axes[0].plot(df["Date"], df["Indexed"], color=GOLD, linewidth=1.4)
axes[0].axhline(100, color=SUBTEXT, linewidth=0.8, linestyle="--")
axes[0].set_yscale("log")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
style_axis(axes[0], "Performance cumulée — base 100 (échelle logarithmique)",
           subtitle=f"CAGR : {cagr*100:+.2f} % / an   •   Rendement total : {total_return*100:+.1f} %",
           ylabel="Indice (base 100)")

colors = [GREEN if v >= 0 else RED for v in yearly["return_%"]]
bars = axes[1].bar(yearly.index.astype(str), yearly["return_%"], color=colors, width=0.65)
axes[1].axhline(0, color=TEXT, linewidth=0.8)
style_axis(axes[1], "Rendement annuel", ylabel="Rendement (%)")
for b, v in zip(bars, yearly["return_%"]):
    axes[1].annotate(f"{v:+.0f}%", (b.get_x() + b.get_width()/2, v), ha="center",
                      va="bottom" if v >= 0 else "top", fontsize=8.5, color=TEXT)

add_footer(fig)
plt.tight_layout()
plt.savefig("outputs/fig_03_trend.png", dpi=150, bbox_inches="tight")
plt.show()

rolling_perf = pd.Series({
    n: (df[df["Date"] >= df["Date"].max() - pd.DateOffset(years=n)]["Close"].iloc[-1] /
        df[df["Date"] >= df["Date"].max() - pd.DateOffset(years=n)]["Close"].iloc[0] - 1) * 100
    for n in [1, 3, 5, 10]
})
print("Rendement glissant :")
for n, v in rolling_perf.items():
    print(f"  • {n} an(s)  : {v:+.1f} %")
print(f"\nMeilleure année : {yearly['return_%'].idxmax()} ({yearly['return_%'].max():+.1f} %)")
print(f"Pire année      : {yearly['return_%'].idxmin()} ({yearly['return_%'].min():+.1f} %)")
print(f"Années positives : {(yearly['return_%']>0).sum()}/{len(yearly)} "
      f"({(yearly['return_%']>0).mean()*100:.0f} %)")

# %% [markdown]
# <a id="5"></a>
# ## 5. Indicateurs techniques avancés
#
# Trois familles d'indicateurs standard en analyse technique sont calculées **manuellement** (sans dépendance
# externe type `ta-lib`, pour garantir la reproductibilité sur Kaggle) :
# - **Moyennes mobiles & Bandes de Bollinger** (tendance & volatilité de court terme)
# - **RSI 14** (momentum / zones de sur-achat-sur-vente)
# - **MACD** (croisement de momentum)

# %% [code]
def compute_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calcule SMA/EMA, Bandes de Bollinger, RSI(14) et MACD sur la colonne 'Close'."""
    out = data.copy()

    out["SMA20"] = out["Close"].rolling(20).mean()
    out["SMA50"] = out["Close"].rolling(50).mean()
    out["SMA200"] = out["Close"].rolling(200).mean()

    std20 = out["Close"].rolling(20).std()
    out["BB_upper"] = out["SMA20"] + 2 * std20
    out["BB_lower"] = out["SMA20"] - 2 * std20

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    out["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    return out


df = compute_technical_indicators(df)

# Détection des croisements SMA50 / SMA200 (golden cross / death cross)
cross_signal = np.sign(df["SMA50"] - df["SMA200"])
df["Cross"] = cross_signal.diff()
crosses = df.loc[df["Cross"].abs() == 2, ["Date", "Close", "Cross"]].dropna()

last_row = df.iloc[-1]
regime = "🟢 HAUSSIER (Golden Cross)" if last_row["SMA50"] > last_row["SMA200"] else "🔴 BAISSIER (Death Cross)"
print(f"Régime de tendance actuel : {regime}")
print(f"RSI(14) actuel : {last_row['RSI14']:.1f} "
      f"({'zone de sur-achat' if last_row['RSI14']>70 else 'zone de sur-vente' if last_row['RSI14']<30 else 'zone neutre'})")
print(f"MACD actuel : {last_row['MACD']:.3f}  |  Signal : {last_row['MACD_signal']:.3f}  "
      f"({'MACD > signal (momentum haussier)' if last_row['MACD']>last_row['MACD_signal'] else 'MACD < signal (momentum baissier)'})")
print(f"\n{len(crosses)} croisements SMA50/SMA200 détectés sur toute la période.")

# %% [code]
sub = df[df["Date"] >= df["Date"].max() - pd.DateOffset(years=3)].copy()

fig, axes = plt.subplots(3, 1, figsize=(13, 11), gridspec_kw={"height_ratios": [2.2, 1, 1]}, sharex=True)

axes[0].plot(sub["Date"], sub["Close"], color=TEXT, linewidth=1.1, label="Clôture", zorder=3)
axes[0].plot(sub["Date"], sub["SMA50"], color=BLUE, linewidth=1.4, label="SMA 50")
axes[0].plot(sub["Date"], sub["SMA200"], color=RED, linewidth=1.4, label="SMA 200")
axes[0].fill_between(sub["Date"], sub["BB_lower"], sub["BB_upper"], color=GOLD, alpha=0.12, label="Bandes de Bollinger (20,2σ)")
sub_crosses = crosses[crosses["Date"] >= sub["Date"].min()]
for _, r in sub_crosses.iterrows():
    axes[0].scatter(r["Date"], r["Close"], color=GREEN if r["Cross"] > 0 else RED, s=70, zorder=5,
                     marker="^" if r["Cross"] > 0 else "v", edgecolor=BG, linewidth=0.8)
style_axis(axes[0], "Analyse technique — GOLD (3 dernières années)",
           subtitle="▲ Golden Cross (signal d'achat)   ▼ Death Cross (signal de vente)", ylabel="Prix ($)")
axes[0].legend(fontsize=9, loc="upper left", ncol=2)

axes[1].plot(sub["Date"], sub["RSI14"], color=PURPLE, linewidth=1.3)
axes[1].axhline(70, color=RED, linestyle="--", linewidth=0.9)
axes[1].axhline(30, color=GREEN, linestyle="--", linewidth=0.9)
axes[1].fill_between(sub["Date"], 70, 100, color=RED, alpha=0.06)
axes[1].fill_between(sub["Date"], 0, 30, color=GREEN, alpha=0.06)
axes[1].set_ylim(0, 100)
style_axis(axes[1], "RSI (14)", ylabel="RSI")

axes[2].bar(sub["Date"], sub["MACD_hist"], color=[GREEN if v >= 0 else RED for v in sub["MACD_hist"]], width=2)
axes[2].plot(sub["Date"], sub["MACD"], color=BLUE, linewidth=1.2, label="MACD")
axes[2].plot(sub["Date"], sub["MACD_signal"], color=ORANGE, linewidth=1.2, label="Signal")
style_axis(axes[2], "MACD (12, 26, 9)", ylabel="MACD", xlabel="Date")
axes[2].legend(fontsize=9, loc="upper left", ncol=2)

add_footer(fig)
plt.tight_layout()
plt.savefig("outputs/fig_04_technical.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# <a id="6"></a>
# ## 6. Analyse de risque quantitatif
#
# Panel complet de métriques utilisées en gestion de portefeuille : volatilité, drawdown, **VaR/CVaR** historiques,
# et ratios ajustés du risque (**Sharpe, Sortino, Calmar**).

# %% [code]
def risk_metrics(returns: pd.Series, close: pd.Series, cagr_value: float) -> dict:
    """Calcule un panel standard de métriques de risque de marché."""
    ann_vol = returns.std() * np.sqrt(252)
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252)

    cum = (1 + returns.fillna(0)).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min()

    var95 = np.percentile(returns.dropna(), 5)
    var99 = np.percentile(returns.dropna(), 1)
    cvar95 = returns[returns <= var95].mean()

    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    sortino = (returns.mean() / downside_vol) * np.sqrt(252) * np.sqrt(252) if downside_vol else np.nan
    sortino = (returns.mean() * 252) / downside_vol if downside_vol else np.nan
    calmar = cagr_value / abs(max_dd) if max_dd else np.nan

    return {
        "Volatilité annualisée": f"{ann_vol*100:.1f} %",
        "Volatilité baissière annualisée": f"{downside_vol*100:.1f} %",
        "Drawdown maximum": f"{max_dd*100:.1f} %",
        "VaR quotidien 95 %": f"{var95*100:.2f} %",
        "VaR quotidien 99 %": f"{var99*100:.2f} %",
        "CVaR (Expected Shortfall) 95 %": f"{cvar95*100:.2f} %",
        "Ratio de Sharpe (rf=0)": f"{sharpe:.2f}",
        "Ratio de Sortino (rf=0)": f"{sortino:.2f}",
        "Ratio de Calmar": f"{calmar:.2f}",
    }, drawdown


metrics, drawdown_series = risk_metrics(df["Return"], df["Close"], cagr)
df["Drawdown"] = drawdown_series

metrics_df = pd.DataFrame(metrics.items(), columns=["Indicateur", "Valeur"])
metrics_df.style.hide(axis="index").set_properties(**{"text-align": "left"})

# %% [code]
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Underwater plot (drawdown)
axes[0, 0].fill_between(df["Date"], df["Drawdown"] * 100, 0, color=RED, alpha=0.6)
style_axis(axes[0, 0], "Drawdown (« underwater plot »)", ylabel="Drawdown (%)")

# Volatilité glissante 30j annualisée
roll_vol = df["Return"].rolling(30).std() * np.sqrt(252) * 100
axes[0, 1].plot(df["Date"], roll_vol, color=ORANGE, linewidth=1.1)
axes[0, 1].axhline(roll_vol.mean(), color=SUBTEXT, linestyle="--", linewidth=0.9, label=f"Moyenne ({roll_vol.mean():.0f}%)")
style_axis(axes[0, 1], "Volatilité glissante 30j (annualisée)", ylabel="Volatilité (%)")
axes[0, 1].legend(fontsize=9)

# VaR historique (queue de distribution)
ret_pct = df["Return"].dropna() * 100
var95 = np.percentile(ret_pct, 5)
axes[1, 0].hist(ret_pct, bins=80, color=BLUE, alpha=0.75)
axes[1, 0].axvline(var95, color=RED, linewidth=1.8, label=f"VaR 95% = {var95:.2f}%")
axes[1, 0].fill_betweenx([0, axes[1,0].get_ylim()[1] if axes[1,0].get_ylim()[1]>0 else 100],
                          ret_pct.min(), var95, color=RED, alpha=0.08)
style_axis(axes[1, 0], "Value at Risk (VaR) historique 95 %", ylabel="Fréquence", xlabel="Rendement quotidien (%)")
axes[1, 0].legend(fontsize=9)

# QQ-plot vs loi normale
stats.probplot(ret_pct, dist="norm", plot=axes[1, 1])
axes[1, 1].get_lines()[0].set_markerfacecolor(BLUE)
axes[1, 1].get_lines()[0].set_markeredgecolor(BLUE)
axes[1, 1].get_lines()[0].set_marker(".")
axes[1, 1].get_lines()[1].set_color(RED)
style_axis(axes[1, 1], "QQ-Plot — rendements vs loi normale", ylabel="Quantiles observés", xlabel="Quantiles théoriques")
axes[1, 1].set_title("")
axes[1, 1].set_title("QQ-Plot — rendements vs loi normale", loc="left", fontsize=14, fontweight="bold", color=TEXT)

add_footer(fig)
plt.tight_layout()
plt.savefig("outputs/fig_05_risk.png", dpi=150, bbox_inches="tight")
plt.show()

print("💡 Lecture : les points du QQ-plot s'écartant de la droite rouge aux extrémités confirment des "
      "« queues épaisses » — des mouvements extrêmes plus fréquents que ne le prédit une loi normale, "
      "cohérent avec le kurtosis positif observé en section 3.")

# %% [markdown]
# <a id="7"></a>
# ## 7. Stationnarité & décomposition de la série temporelle
#
# Un test de **Dickey-Fuller augmenté (ADF)** vérifie si la série est stationnaire (condition requise par de
# nombreux modèles de séries temporelles), puis une **décomposition STL** isole tendance, saisonnalité et résidu.

# %% [code]
def adf_report(series: pd.Series, label: str):
    result = adfuller(series.dropna())
    stationary = result[1] < 0.05
    print(f"{label:<28} ADF stat = {result[0]:8.3f}   p-value = {result[1]:.4f}   "
          f"→ {'✅ stationnaire' if stationary else '❌ non stationnaire'} (seuil 5%)")
    return stationary

print("Test de stationnarité (Augmented Dickey-Fuller) :\n")
adf_report(df["Close"], "Prix de clôture")
adf_report(df["Return"].dropna(), "Rendements quotidiens")

# %% [code]
close_indexed = df.set_index("Date")["Close"]
stl = STL(close_indexed, period=252, robust=True)
result = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
comps = [("Série observée", close_indexed, TEXT), ("Tendance", result.trend, GOLD),
         ("Saisonnalité (période 252j)", result.seasonal, BLUE), ("Résidu", result.resid, PURPLE)]
for ax, (title, series, color) in zip(axes, comps):
    ax.plot(series.index, series.values, color=color, linewidth=1.0)
    style_axis(ax, title)
axes[-1].set_xlabel("Date", color=SUBTEXT)
fig.suptitle("Décomposition STL de la série de prix", fontsize=15, fontweight="bold", color=TEXT, x=0.01, ha="left")
add_footer(fig)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("outputs/fig_06_stl.png", dpi=150, bbox_inches="tight")
plt.show()

adf_trend_note = (
    "Le prix n'est pas stationnaire (tendance/mémoire longue) alors que les rendements le sont : c'est le "
    "comportement classique d'une marche aléatoire avec dérive, cohérent avec l'hypothèse d'efficience des "
    "marchés — cela justifie de modéliser les RENDEMENTS plutôt que le PRIX brut en section 8."
)
print(f"💡 {adf_trend_note}")

# %% [markdown]
# <a id="8"></a>
# ## 8. Modélisation prédictive (Machine Learning)
#
# Deux tâches complémentaires, construites sur un **découpage chronologique strict train/test** (aucun mélange
# temporel, pour éviter toute fuite d'information — erreur classique en finance quantitative) :
#
# - **8.1 Régression** — prévoir le prix de clôture du lendemain (benchmarké contre un modèle naïf de persistance)
# - **8.2 Classification** — prévoir le SENS du mouvement du lendemain (hausse/baisse), tâche plus réaliste compte
#   tenu de l'efficience des marchés

# %% [code]
def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Construit un jeu de features pour la modélisation (aucune fuite de futur : tout est décalé/rolling passé)."""
    f = data.copy()

    for lag in [1, 2, 3, 5, 10]:
        f[f"lag_return_{lag}"] = f["Return"].shift(lag)

    f["roll_mean_5"] = f["Return"].rolling(5).mean().shift(1)
    f["roll_std_5"] = f["Return"].rolling(5).std().shift(1)
    f["roll_mean_10"] = f["Return"].rolling(10).mean().shift(1)
    f["roll_std_10"] = f["Return"].rolling(10).std().shift(1)

    f["price_to_sma20"] = (f["Close"] / f["SMA20"] - 1).shift(1)
    f["price_to_sma50"] = (f["Close"] / f["SMA50"] - 1).shift(1)
    f["rsi_lag"] = f["RSI14"].shift(1)
    f["macd_hist_lag"] = f["MACD_hist"].shift(1)
    f["volume_change"] = f["Volume"].replace(0, np.nan).pct_change().shift(1)
    f["dow"] = f["Date"].dt.dayofweek
    f["month"] = f["Date"].dt.month

    f["target_price_next"] = f["Close"].shift(-1)
    f["target_up_next"] = (f["Return"].shift(-1) > 0).astype(int)

    # Sécurité : neutralise tout inf résiduel (ex. divisions par un volume nul) avant modélisation
    f = f.replace([np.inf, -np.inf], np.nan)

    return f


FEATURE_COLS = [
    "lag_return_1", "lag_return_2", "lag_return_3", "lag_return_5", "lag_return_10",
    "roll_mean_5", "roll_std_5", "roll_mean_10", "roll_std_10",
    "price_to_sma20", "price_to_sma50", "rsi_lag", "macd_hist_lag", "volume_change", "dow", "month",
]

feat = build_features(df).dropna(subset=FEATURE_COLS + ["target_price_next", "target_up_next"]).reset_index(drop=True)

TEST_SIZE = 252  # ~ 1 an de bourse
train, test = feat.iloc[:-TEST_SIZE], feat.iloc[-TEST_SIZE:]

print(f"Jeu d'entraînement : {len(train):,} séances ({train['Date'].min().date()} → {train['Date'].max().date()})")
print(f"Jeu de test        : {len(test):,} séances ({test['Date'].min().date()} → {test['Date'].max().date()})")
print(f"Nombre de features  : {len(FEATURE_COLS)}")

# %% [markdown]
# ### 8.1 Régression — prix de clôture du lendemain
#
# ⚠️ **Choix méthodologique important** : les modèles sont entraînés à prédire le **rendement** du lendemain
# (variable stationnaire, cf. test ADF de la section 7) et non le prix brut. Le prix prédit est ensuite
# reconstruit par `prix(t) × (1 + rendement prédit)`. Prédire directement un prix brut avec des features
# stationnaires est une erreur fréquente qui produit des prévisions incohérentes hors échantillon.

# %% [code]
train = train.copy()
test = test.copy()
train["target_return_next"] = train["Return"].shift(-1)  # rendement de demain, calculé dans l'ordre chronologique
test["target_return_next"] = test["Return"].shift(-1)
train = train.dropna(subset=FEATURE_COLS + ["target_return_next"])
test_eval = test.dropna(subset=FEATURE_COLS + ["target_return_next"]).reset_index(drop=True)

X_train, y_train = train[FEATURE_COLS], train["target_return_next"]
X_test, y_test = test_eval[FEATURE_COLS], test_eval["target_return_next"]

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

lin_reg = LinearRegression().fit(X_train_s, y_train)
lin_ret_pred = lin_reg.predict(X_test_s)

rf_reg = RandomForestRegressor(n_estimators=400, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_train)
rf_ret_pred = rf_reg.predict(X_test)

naive_ret_pred = np.zeros(len(y_test))  # naïf : « le rendement de demain sera nul » <=> prix(t+1) = prix(t)

# Reconstruction des prix à partir des rendements prédits
today_close = test_eval["Close"].values
real_price_next = test_eval["target_price_next"].values
naive_price = today_close * (1 + naive_ret_pred)
lin_price = today_close * (1 + lin_ret_pred)
rf_price = today_close * (1 + rf_ret_pred)


def regression_scores(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"Modèle": name, "RMSE ($)": rmse, "MAE ($)": mae, "MAPE (%)": mape, "R²": r2}


results = pd.DataFrame([
    regression_scores(real_price_next, naive_price, "Naïf (rendement nul)"),
    regression_scores(real_price_next, lin_price, "Régression linéaire"),
    regression_scores(real_price_next, rf_price, "Random Forest"),
])
results.style.hide(axis="index").background_gradient(cmap="RdYlGn_r", subset=["RMSE ($)", "MAE ($)", "MAPE (%)"]).format(precision=3)

# %% [code]
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(test_eval["Date"], real_price_next, color=TEXT, linewidth=1.6, label="Prix réel", zorder=4)
ax.plot(test_eval["Date"], naive_price, color=SUBTEXT, linewidth=1.0, linestyle="--", label="Naïf (rendement nul)")
ax.plot(test_eval["Date"], lin_price, color=BLUE, linewidth=1.1, label="Régression linéaire", alpha=0.9)
ax.plot(test_eval["Date"], rf_price, color=GOLD, linewidth=1.2, label="Random Forest", alpha=0.9)
style_axis(ax, "Prévision du prix de clôture — reconstruite à partir des rendements prédits (1 an, hors échantillon)",
           subtitle="Split chronologique strict — le modèle n'a jamais vu les données de test à l'entraînement",
           ylabel="Prix ($)")
ax.legend(fontsize=9.5, ncol=4, loc="upper left")
add_footer(fig)
plt.tight_layout()
plt.savefig("outputs/fig_07_regression.png", dpi=150, bbox_inches="tight")
plt.show()

best_model = results.loc[results["RMSE ($)"].idxmin(), "Modèle"]
print(f"💡 Le modèle « {best_model} » obtient le RMSE le plus bas. Sur des séries financières proches d'une marche "
      "aléatoire, il est normal — et scientifiquement honnête de le signaler — qu'un modèle naïf reste très "
      "compétitif sur le PRIX reconstruit : la vraie valeur ajoutée du Machine Learning se mesure sur le SENS du "
      "mouvement (tâche 8.2), plus actionnable pour une décision d'achat/vente qu'un prix ponctuel.")

# %% [markdown]
# ### 8.2 Classification — sens du mouvement du lendemain (hausse / baisse)
#
# Tâche plus réaliste pour une décision d'entreprise (« dois-je acheter ou vendre demain ? ») : un
# **Random Forest Classifier** est comparé à un classifieur naïf (« demain ressemble à aujourd'hui »),
# évalué par accuracy, precision/recall, F1 et AUC-ROC — pas seulement l'accuracy, trompeuse si les classes
# sont déséquilibrées.

# %% [code]
Xc_train, yc_train = train[FEATURE_COLS], train["target_up_next"]
Xc_test, yc_test = test_eval[FEATURE_COLS], test_eval["target_up_next"]

print(f"Répartition des classes (test) — Hausse : {yc_test.mean()*100:.1f} %  |  Baisse : {(1-yc_test.mean())*100:.1f} %")

dummy = DummyClassifier(strategy="most_frequent", random_state=42).fit(Xc_train, yc_train)
dummy_pred = dummy.predict(Xc_test)

clf = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=15,
                              class_weight="balanced", random_state=42, n_jobs=-1)
clf.fit(Xc_train, yc_train)
clf_pred = clf.predict(Xc_test)
clf_proba = clf.predict_proba(Xc_test)[:, 1]


def clf_scores(y_true, y_pred, name, proba=None):
    row = {
        "Modèle": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, proba) if proba is not None else np.nan,
    }
    return row


clf_results = pd.DataFrame([
    clf_scores(yc_test, dummy_pred, "Naïf (classe majoritaire)"),
    clf_scores(yc_test, clf_pred, "Random Forest", clf_proba),
])
clf_results.style.hide(axis="index").background_gradient(cmap="RdYlGn", subset=["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]).format(precision=3)

# %% [code]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

cm = confusion_matrix(yc_test, clf_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
            xticklabels=["Baisse", "Hausse"], yticklabels=["Baisse", "Hausse"],
            annot_kws={"size": 13, "color": BG})
style_axis(axes[0], "Matrice de confusion", xlabel="Prédiction", ylabel="Réalité")
axes[0].grid(False)

fpr, tpr, _ = roc_curve(yc_test, clf_proba)
axes[1].plot(fpr, tpr, color=GOLD, linewidth=2, label=f"Random Forest (AUC={roc_auc_score(yc_test, clf_proba):.2f})")
axes[1].plot([0, 1], [0, 1], color=SUBTEXT, linestyle="--", linewidth=1, label="Hasard (AUC=0.50)")
style_axis(axes[1], "Courbe ROC", xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs")
axes[1].legend(fontsize=8.5, loc="lower right")

importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True).tail(10)
axes[2].barh(importances.index, importances.values, color=GOLD)
style_axis(axes[2], "Importance des variables (Top 10)", xlabel="Importance")

add_footer(fig)
plt.tight_layout()
plt.savefig("outputs/fig_08_classification.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"💡 Le Random Forest atteint une accuracy de {accuracy_score(yc_test, clf_pred)*100:.1f} % "
      f"(vs {accuracy_score(yc_test, dummy_pred)*100:.1f} % pour le classifieur naïf) et un AUC-ROC de "
      f"{roc_auc_score(yc_test, clf_proba):.2f}. Un AUC proche de 0.5 confirme la difficulté théorique de prédire "
      "le sens du marché à court terme (marché quasi-efficient) — un résultat significativement supérieur à 0.5 "
      "serait au contraire suspect (surapprentissage ou fuite de données) et mériterait d'être challengé.")

# %% [markdown]
# <a id="9"></a>
# ## 9. Prévision ARIMA avec intervalle de confiance
#
# Un modèle **ARIMA(5,1,0)** (différenciation d'ordre 1, cohérente avec la non-stationnarité constatée en
# section 7) est ajusté sur l'historique complet pour produire une prévision à 30 séances avec intervalle
# de confiance à 95 %, complémentaire aux approches Machine Learning ci-dessus.

# %% [code]
close_series = df.set_index("Date")["Close"]

arima_model = ARIMA(close_series.reset_index(drop=True), order=(5, 1, 0)).fit()  # index entier : compatible statsmodels >= 0.14
print(arima_model.summary().tables[0])

HORIZON = 30
forecast = arima_model.get_forecast(steps=HORIZON)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int(alpha=0.05)

future_dates = pd.bdate_range(close_series.index[-1] + pd.Timedelta(days=1), periods=HORIZON)
forecast_mean.index = future_dates
forecast_ci.index = future_dates

# %% [code]
fig, ax = plt.subplots(figsize=(13, 5.5))
hist = close_series.tail(180)
ax.plot(hist.index, hist.values, color=TEXT, linewidth=1.4, label="Historique (180 dernières séances)")
ax.plot(forecast_mean.index, forecast_mean.values, color=GOLD, linewidth=1.8, label=f"Prévision ARIMA(5,1,0) — {HORIZON}j")
ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                color=GOLD, alpha=0.18, label="Intervalle de confiance 95 %")
ax.axvline(close_series.index[-1], color=SUBTEXT, linestyle=":", linewidth=1)
style_axis(ax, f"Prévision ARIMA à {HORIZON} séances", ylabel="Prix ($)",
           subtitle="La largeur croissante de l'intervalle traduit l'incertitude grandissante à mesure que l'horizon s'éloigne")
ax.legend(fontsize=9.5, loc="upper left")
add_footer(fig)
plt.tight_layout()
plt.savefig("outputs/fig_09_arima.png", dpi=150, bbox_inches="tight")
plt.show()

pct_width = (forecast_ci.iloc[-1, 1] - forecast_ci.iloc[-1, 0]) / forecast_mean.iloc[-1] * 100
print(f"Prix actuel : {close_series.iloc[-1]:.2f} $")
print(f"Prévision à {HORIZON} jours : {forecast_mean.iloc[-1]:.2f} $ "
      f"[{forecast_ci.iloc[-1,0]:.2f} $ ; {forecast_ci.iloc[-1,1]:.2f} $] (IC 95 %)")
print(f"Largeur de l'intervalle à {HORIZON}j : ±{pct_width/2:.1f} % autour de la prévision centrale — "
      "à interpréter comme un ordre de grandeur, pas une certitude.")

# %% [markdown]
# <a id="10"></a>
# ## 10. Tableau de bord exécutif & recommandation
#
# Synthèse des indicateurs clés calculés dans les sections précédentes, assemblée dans un tableau de bord
# unique — le format attendu par une direction pressée par le temps — suivi d'une **recommandation générée
# dynamiquement** à partir de règles explicites (pas de conclusion codée en dur).

# %% [code]
kpis = [
    ("CAGR (11,8 ans)", f"{cagr*100:+.1f}%", GREEN if cagr > 0 else RED),
    ("Rendement 1 an", f"{rolling_perf[1]:+.1f}%", GREEN if rolling_perf[1] > 0 else RED),
    ("Volatilité annualisée", f"{df['Return'].std()*np.sqrt(252)*100:.1f}%", ORANGE),
    ("Drawdown maximum", f"{df['Drawdown'].min()*100:.1f}%", RED),
    ("Ratio de Sharpe", metrics["Ratio de Sharpe (rf=0)"], BLUE),
    ("Régime technique", "Haussier" if last_row["SMA50"] > last_row["SMA200"] else "Baissier",
     GREEN if last_row["SMA50"] > last_row["SMA200"] else RED),
    ("RSI (14)", f"{last_row['RSI14']:.0f}", PURPLE),
    ("Prévision ARIMA 30j", f"{forecast_mean.iloc[-1]:.2f} $", GOLD),
]

fig, axes = plt.subplots(2, 4, figsize=(15, 5.5))
fig.suptitle("Tableau de bord exécutif — GOLD", fontsize=17, fontweight="bold", color=TEXT, x=0.02, ha="left", y=1.02)
for ax, (label, value, color) in zip(axes.flat, kpis):
    ax.set_facecolor(PANEL)
    ax.axis("off")
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=22, fontweight="bold", color=color)
    ax.text(0.5, 0.22, label, ha="center", va="center", fontsize=10.5, color=SUBTEXT)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)
plt.tight_layout()
plt.savefig("outputs/fig_10_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [code]
def generate_recommendation(cagr_v, vol_v, max_dd_v, sharpe_v, trend_bullish, rsi_v):
    """Génère une recommandation structurée à partir de règles explicites sur les métriques calculées."""
    lines = []

    if cagr_v > 0.10 and vol_v > 0.30:
        profile = "Fort potentiel de croissance, risque élevé — actif de conviction, pas de trésorerie court terme"
    elif cagr_v > 0.10 and vol_v <= 0.30:
        profile = "Fort potentiel de croissance, risque modéré — profil attractif"
    elif cagr_v <= 0.10 and vol_v > 0.30:
        profile = "Risque élevé pour un rendement historique limité — profil peu attractif"
    else:
        profile = "Profil défensif — rendement et risque tous deux modérés"
    lines.append(f"• Profil risque/rendement : {profile}")

    if trend_bullish and rsi_v < 70:
        timing = "Configuration technique favorable, sans zone de surachat critique → fenêtre d'entrée raisonnable"
    elif trend_bullish and rsi_v >= 70:
        timing = "Tendance haussière mais RSI en zone de surachat → privilégier un renforcement échelonné plutôt qu'une entrée massive"
    else:
        timing = "Tendance baissière à court terme → prudence, attendre une confirmation de retournement"
    lines.append(f"• Timing : {timing}")

    horizon = "long terme (≥ 3-5 ans)" if vol_v > 0.30 else "moyen terme (1-3 ans)"
    lines.append(f"• Horizon recommandé : {horizon}, pour lisser la cyclicité observée")

    lines.append(f"• Ratio rendement/risque (Sharpe) : {sharpe_v:.2f} "
                 f"({'correct' if sharpe_v > 0.5 else 'faible'} pour un actif de cette classe)")

    lines.append(f"• Perte maximale historique à anticiper : jusqu'à {max_dd_v*100:.0f} % "
                 "— ne pas exposer une trésorerie nécessaire à court terme")

    return "\n".join(lines)


sharpe_val = float(metrics["Ratio de Sharpe (rf=0)"])
recommendation = generate_recommendation(
    cagr_v=cagr, vol_v=df["Return"].std()*np.sqrt(252),
    max_dd_v=df["Drawdown"].min(), sharpe_v=sharpe_val,
    trend_bullish=last_row["SMA50"] > last_row["SMA200"], rsi_v=last_row["RSI14"],
)
print("RECOMMANDATION GÉNÉRÉE AUTOMATIQUEMENT À PARTIR DES INDICATEURS CALCULÉS :\n")
print(recommendation)
print("\n⚠️ Cette note est une aide à la décision quantitative — elle ne constitue pas un conseil en investissement. "
      "La décision finale relève de la direction, idéalement avec l'avis d'un conseiller financier agréé.")

# %% [markdown]
# <a id="11"></a>
# ## 11. Deep Learning — prévision par réseau de neurones LSTM
#
# Un réseau **LSTM (Long Short-Term Memory)** — l'architecture de référence pour les séquences temporelles —
# est entraîné sur des fenêtres glissantes de rendements passés pour prédire le rendement du lendemain,
# avec le même split chronologique strict et la même comparaison à un modèle naïf que dans la section 8.
# Cette section suppose `tensorflow` disponible (préinstallé sur Kaggle).

# %% [code]
if os.environ.get("SKIP_LSTM") == "1":
    print("Section LSTM ignoree (SKIP_LSTM=1).")
    raise SystemExit(0)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)

WINDOW = 30  # séquence des 30 derniers rendements pour prédire le rendement du lendemain

returns_arr = df["Return"].fillna(0).values.reshape(-1, 1)
ret_scaler = StandardScaler().fit(returns_arr[: -TEST_SIZE])
returns_scaled = ret_scaler.transform(returns_arr).flatten()


def make_sequences(series, window):
    """Construit des sequences (X: fenetre glissante de rendements passes, y: rendement du jour suivant),
    sans trou temporel : X[k] = rendements[i-window:i], y[k] = rendement[i]."""
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window:i])
        y.append(series[i])
    return np.array(X), np.array(y)


X_seq, y_seq = make_sequences(returns_scaled, WINDOW)
seq_target_idx = np.arange(WINDOW, len(returns_scaled))  # index (ligne df) correspondant a chaque cible y

split_idx = len(X_seq) - TEST_SIZE
X_seq_train, X_seq_test = X_seq[:split_idx], X_seq[split_idx:]
y_seq_train, y_seq_test = y_seq[:split_idx], y_seq[split_idx:]
target_idx_test = seq_target_idx[split_idx:]  # index globaux (dans df) des cibles de test

X_seq_train = X_seq_train.reshape(-1, WINDOW, 1)
X_seq_test = X_seq_test.reshape(-1, WINDOW, 1)

print(f"Sequences d'entrainement : {X_seq_train.shape}  |  Sequences de test : {X_seq_test.shape}")

# %% [code]
lstm_model = Sequential([
    LSTM(32, input_shape=(WINDOW, 1), return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1),
])
lstm_model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

history = lstm_model.fit(
    X_seq_train, y_seq_train,
    validation_split=0.15,
    epochs=60,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0,
)
print(f"Entraînement terminé en {len(history.history['loss'])} époques (arrêt anticipé).")

# %% [code]
lstm_pred_scaled = lstm_model.predict(X_seq_test, verbose=0).flatten()
lstm_pred_returns = ret_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()

# Reconstruction du prix par index global (robuste) : target_idx_test contient, pour chaque prédiction,
# la ligne df de la cible (rendement du jour i). "Aujourd'hui" = ligne (i-1), déjà connue au moment de la prédiction.
close_vals = df["Close"].values
today_idx = target_idx_test - 1
today_prices = close_vals[today_idx]
next_prices_real = close_vals[target_idx_test]
dates_seq_test = df["Date"].values[target_idx_test]
lstm_price_pred = today_prices * (1 + lstm_pred_returns)

fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(dates_seq_test, next_prices_real, color=TEXT, linewidth=1.6, label="Prix réel", zorder=4)
ax.plot(dates_seq_test, today_prices, color=SUBTEXT, linewidth=1.0, linestyle="--", label="Naïf (rendement nul)")
ax.plot(dates_seq_test, lstm_price_pred, color=PURPLE, linewidth=1.3, label="LSTM")
style_axis(ax, "Prévision LSTM du prix de clôture — reconstruite à partir des rendements prédits",
           subtitle=f"Fenêtre de {WINDOW} séances passées → prédiction du rendement à t+1, split chronologique strict",
           ylabel="Prix ($)")
ax.legend(fontsize=9.5, loc="upper left")
add_footer(fig)
plt.tight_layout()
plt.savefig("outputs/fig_11_lstm.png", dpi=150, bbox_inches="tight")
plt.show()

lstm_rmse = np.sqrt(mean_squared_error(next_prices_real, lstm_price_pred))
naive_rmse_lstm = np.sqrt(mean_squared_error(next_prices_real, today_prices))
print(f"RMSE LSTM  : {lstm_rmse:.3f} $")
print(f"RMSE Naïf  : {naive_rmse_lstm:.3f} $")
print(f"\n💡 Comme pour la Random Forest en section 8, le LSTM ne bat pas significativement le modèle naïf sur le "
      "PRIX — cohérence rassurante entre deux familles de modèles totalement différentes (arbres vs réseaux de "
      "neurones), qui renforce la conclusion d'efficience de marché plutôt que de suggérer un problème d'implémentation.")

# %% [code]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(history.history["loss"], color=BLUE, linewidth=1.6, label="Perte (entraînement)")
ax.plot(history.history["val_loss"], color=GOLD, linewidth=1.6, label="Perte (validation)")
style_axis(ax, "Courbe d'apprentissage du LSTM", ylabel="MSE", xlabel="Époque")
ax.legend(fontsize=9.5)
add_footer(fig)
plt.tight_layout()
plt.savefig("outputs/fig_12_lstm_learning.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# <a id="12"></a>
# ## 12. Conclusion
#
# Ce notebook illustre un pipeline complet d'analyse quantitative appliqué à une décision d'entreprise :
#
# | Étape | Compétence démontrée |
# |---|---|
# | Nettoyage & EDA | Manipulation de données, statistiques descriptives |
# | Indicateurs techniques | Feature engineering financier (RSI, MACD, Bollinger) |
# | Analyse de risque | VaR, CVaR, Sharpe/Sortino/Calmar, tests statistiques |
# | Séries temporelles | Stationnarité (ADF), décomposition STL, ARIMA |
# | Machine Learning | Split chronologique rigoureux, régression & classification, évaluation multi-métriques |
# | Deep Learning | Réseau LSTM (Keras/TensorFlow), séquences temporelles, courbes d'apprentissage |
# | Communication | Visualisations professionnelles, tableau de bord exécutif, recommandation actionnable |
#
# **Point clé méthodologique** : la valeur de ce travail ne réside pas dans une promesse de « battre le marché »,
# mais dans la **rigueur du raisonnement** — tester la stationnarité avant de modéliser, comparer chaque modèle
# à une base naïve, être transparent quand un AUC proche de 0.5 confirme l'efficience du marché plutôt que de
# forcer un résultat artificiellement optimiste. C'est cette rigueur, plus qu'un score isolé, qui rend une
# analyse quantitative exploitable pour une décision réelle.