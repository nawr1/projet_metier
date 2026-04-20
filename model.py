import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, SelectFromModel
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from config import FEATURE_COLS, TARGET_COL


# ==============================================================================
# FEATURE SELECTION
# ==============================================================================

def apply_feature_selection(X_train, y_train, X_test):
    """
    3 passes successives de feature selection.
    Ne supprime que ce qui est réellement inutile.
    Retourne X_train filtré, X_test filtré, liste des features retenues.
    """
    selected = list(X_train.columns)

    # 1. VarianceThreshold — supprime les features constantes
    vt   = VarianceThreshold(threshold=0.01)
    vt.fit(X_train)
    mask = vt.get_support()
    selected = [f for f, m in zip(selected, mask) if m]
    removed  = [f for f, m in zip(list(X_train.columns), mask) if not m]
    print(f"  VarianceThreshold  : supprimé {removed if removed else 'aucune'}")
    X_train, X_test = X_train[selected], X_test[selected]

    # 2. SelectKBest — garde les K plus corrélées à la cible
    k   = min(7, len(selected))
    skb = SelectKBest(score_func=f_regression, k=k)
    skb.fit(X_train, y_train)
    mask = skb.get_support()
    removed  = [f for f, m in zip(selected, mask) if not m]
    selected = [f for f, m in zip(selected, mask) if m]
    print(f"  SelectKBest(k={k})  : supprimé {removed if removed else 'aucune'}")
    X_train, X_test = X_train[selected], X_test[selected]

    # 3. SelectFromModel — importance via Random Forest
    sfm = SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42), threshold="median")
    sfm.fit(X_train, y_train)
    mask    = sfm.get_support()
    removed = [f for f, m in zip(selected, mask) if not m]
    kept    = [f for f, m in zip(selected, mask) if m]
    if not kept:
        print(f"  SelectFromModel    : aucune retenue, fallback sur SelectKBest")
    else:
        selected = kept
        print(f"  SelectFromModel    : supprimé {removed if removed else 'aucune'}")
        X_train, X_test = X_train[selected], X_test[selected]

    print(f"  Features finales   : {selected}")
    return X_train, X_test, selected


# ==============================================================================
# ENTRAÎNEMENT ET ÉVALUATION
# ==============================================================================

def train_and_evaluate(X_train, y_train, X_test, y_test, label=""):
    results = {}
    for name, model in [
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("XGBoost",       XGBRegressor(n_estimators=100, random_state=42, verbosity=0)),
    ]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred)
        cv  = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()
        results[name] = {"MAE": round(mae, 3), "R2": round(r2, 3), "CV_R2": round(cv, 3)}
        print(f"  [{label}] {name:15s} → MAE={mae:.2f} | R²={r2:.3f} | CV R²={cv:.3f}")
    return results


# ==============================================================================
# ANALYSE DE LA DÉRIVE DES MÉTRIQUES
# ==============================================================================

def compute_drift(df_pre, df_post):
    drift = {}
    for col in FEATURE_COLS:
        mean_pre  = df_pre[col].mean()
        mean_post = df_post[col].mean()
        pct       = ((mean_post - mean_pre) / max(abs(mean_pre), 0.001)) * 100
        drift[col] = {
            "Moyenne pre-IA":  round(mean_pre, 3),
            "Moyenne post-IA": round(mean_post, 3),
            "Dérive (%)":      round(pct, 1),
            "Statut":          "CONTAMINEE" if abs(pct) > 50 else "STABLE"
        }
        print(f"  {col:25s} | pre={mean_pre:.2f} | post={mean_post:.2f} | dérive={pct:+.1f}% | {drift[col]['Statut']}")
    return drift


# ==============================================================================
# VISUALISATIONS
# ==============================================================================

def generate_visualizations(df_pre, df_post, drift, results_classic, results_adaptive, selected):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Effort logiciel : Impact de l'IA sur les métriques GitHub", fontsize=14, fontweight="bold")

    # 1. Dérive des métriques
    ax = axes[0, 0]
    feats   = list(drift.keys())
    drifts  = [drift[f]["Dérive (%)"] for f in feats]
    colors  = ["#E24B4A" if abs(d) > 50 else "#1D9E75" for d in drifts]
    ax.barh(feats, drifts, color=colors)
    ax.axvline(0,   color="black", linewidth=0.8)
    ax.axvline(50,  color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="Seuil ±50%")
    ax.axvline(-50, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Dérive des métriques (pre vs post IA)")
    ax.set_xlabel("Dérive (%)")
    ax.legend(fontsize=8)

    # 2. Distribution du churn
    ax = axes[0, 1]
    ax.hist(df_pre["churn"].clip(0, 2000),  bins=30, alpha=0.6, label="Pre-IA  (2019-2021)", color="#378ADD")
    ax.hist(df_post["churn"].clip(0, 2000), bins=30, alpha=0.6, label="Post-IA (2023-2025)", color="#D85A30")
    ax.set_title("Distribution du Code Churn")
    ax.set_xlabel("Churn (lignes)")
    ax.set_ylabel("Nombre de PRs")
    ax.legend()

    # 3. Comparaison R² des modèles
    ax = axes[1, 0]
    labels = ["RF Classique", "XGB Classique", "RF Adaptatif", "XGB Adaptatif"]
    r2s    = [
        results_classic["Random Forest"]["R2"],
        results_classic["XGBoost"]["R2"],
        results_adaptive["Random Forest"]["R2"],
        results_adaptive["XGBoost"]["R2"],
    ]
    bar_colors = ["#B5D4F4", "#B5D4F4", "#1D9E75", "#1D9E75"]
    bars = ax.bar(labels, r2s, color=bar_colors)
    ax.set_title("Performance des modèles (R²)")
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Baseline 0.5")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=10)

    # 4. Features retenues vs rejetées
    ax = axes[1, 1]
    colors_feat = ["#1D9E75" if f in selected else "#E24B4A" for f in FEATURE_COLS]
    ax.barh(FEATURE_COLS, [1] * len(FEATURE_COLS), color=colors_feat)
    ax.set_title("Features retenues (vert) vs rejetées (rouge)")
    ax.set_xticks([])
    legend_elems = [
        mpatches.Patch(facecolor="#1D9E75", label="Retenue"),
        mpatches.Patch(facecolor="#E24B4A", label="Rejetée")
    ]
    ax.legend(handles=legend_elems, fontsize=8)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches="tight")
    print("\nVisualisations sauvegardées : results.png")
    plt.show()


# ==============================================================================
# PIPELINE COMPLET
# ==============================================================================

def run_analysis():
    print("\n=== Chargement des données ===")
    df_pre  = pd.read_csv("data_pre_ai.csv").dropna(subset=FEATURE_COLS + [TARGET_COL])
    df_post = pd.read_csv("data_post_ai.csv").dropna(subset=FEATURE_COLS + [TARGET_COL])
    print(f"Pre-IA  : {len(df_pre)} PRs | Post-IA : {len(df_post)} PRs")

    X_pre, y_pre   = df_pre[FEATURE_COLS],  df_pre[TARGET_COL]
    X_post, y_post = df_post[FEATURE_COLS], df_post[TARGET_COL]

    # Modèle classique : entraîné pre-IA, testé post-IA
    print("\n=== Modèle classique (pre-IA → post-IA) ===")
    results_classic = train_and_evaluate(X_pre, y_pre, X_post, y_post, "Classique")

    # Modèle adaptatif : feature selection + entraîné post-IA
    print("\n=== Modèle adaptatif avec feature selection (post-IA) ===")
    split = int(len(df_post) * 0.8)
    X_tr, X_te = X_post.iloc[:split], X_post.iloc[split:]
    y_tr, y_te = y_post.iloc[:split], y_post.iloc[split:]

    print("  Application de la feature selection :")
    X_tr_sel, X_te_sel, selected = apply_feature_selection(X_tr, y_tr, X_te)
    results_adaptive = train_and_evaluate(X_tr_sel, y_tr, X_te_sel, y_te, "Adaptatif")

    # Analyse de la dérive
    print("\n=== Dérive des métriques ===")
    drift = compute_drift(df_pre, df_post)

    # Visualisations
    generate_visualizations(df_pre, df_post, drift, results_classic, results_adaptive, selected)

    return results_classic, results_adaptive, drift, selected