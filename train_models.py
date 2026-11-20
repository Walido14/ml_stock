#!/usr/bin/env python3
"""
Entraîne les 3 modèles Mediko à partir de data/dataset_*.csv
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import CATEGORICAL, LOT_FEATURES, STOCK_FEATURES

DATA, MODELS = Path("data"), Path("models")
MODELS.mkdir(exist_ok=True)
SEED = 42

# ... (Gardez les fonctions pipe, clf, best_threshold, cls_metrics, split_time, split_lot identiques) ...
def pipe(features, model):
    num = [f for f in features if f not in CATEGORICAL]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
                            remainder="passthrough")
    return Pipeline([("pre", pre), ("model", model)])

def clf():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=24,
                                          l2_regularization=1.0, early_stopping=False, random_state=SEED)

def best_threshold(y, p):
    ts = np.linspace(0.05, 0.9, 86)
    f1s = [f1_score(y, p >= t, zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f1s))])

def cls_metrics(y, p, thr, baseline_score):
    return {
        "auc": round(roc_auc_score(y, p), 4),
        "auc_baseline_regle_simple": round(roc_auc_score(y, baseline_score), 4),
        "average_precision": round(average_precision_score(y, p), 4),
        "taux_positif": round(float(np.mean(y)), 4),
        "seuil_alerte": round(thr, 3),
        "precision": round(precision_score(y, p >= thr, zero_division=0), 4),
        "rappel": round(recall_score(y, p >= thr, zero_division=0), 4),
        "f1": round(f1_score(y, p >= thr, zero_division=0), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "n_test": int(len(y)),
    }

def split_time(df):
    d = pd.to_datetime(df["date_snapshot"])
    tr = d <= "2025-12-31"
    va = (d >= "2026-02-01") & (d <= "2026-04-30")
    te = d >= "2026-06-01"
    return df[tr], df[va], df[te]

def train_classifier(name, df, features, target, baseline_fn, split):
    print(f"  -> Préparation des données pour {name}...")
    tr, va, te = split(df)
    
    print(f"  -> Entraînement du modèle {name} (Train: {len(tr)} lignes)...")
    m = pipe(features, clf()).fit(tr[features], tr[target])
    
    print(f"  -> Calcul du seuil optimal et évaluation...")
    thr = best_threshold(va[target], m.predict_proba(va[features])[:, 1])
    p_te = m.predict_proba(te[features])[:, 1]
    metrics = cls_metrics(te[target].values, p_te, thr, baseline_fn(te))
    
    print(f"  -> Ré-entraînement sur toutes les données et sauvegarde...")
    final = pipe(features, clf()).fit(df[features], df[target])
    joblib.dump({"model": final, "features": features, "seuil_moyen": thr, "seuil_eleve": 0.6,
                 "metrics": metrics}, MODELS / f"{name}.joblib")
    
    print(f"[{name}] ✅ Terminé :", json.dumps(metrics, ensure_ascii=False))
    return metrics


def main():
    print("🚀 Démarrage de l'entraînement des 3 modèles Mediko...\n")
    out = {}
    start_time = time.time()

    # 1) RUPTURE
    print("📦 [1/3] Chargement du dataset RUPTURE...")
    rup = pd.read_csv(DATA / "dataset_rupture.csv")
    out["rupture"] = train_classifier("rupture", rup, STOCK_FEATURES, "rupture_14j",
                                      lambda d: -d["jours_couverture"].values, split_time)

    # 2) EXPIRATION
    print("\n📦 [2/3] Chargement du dataset EXPIRATION...")
    exp = pd.read_csv(DATA / "dataset_expiration.csv", dtype={"numero_lot": str})
    exp["grp"] = exp["stock_id"].astype(str) + "|" + exp["numero_lot"].astype(str)

    def split_lot(df):
        g = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tv, te = next(g.split(df, groups=df["grp"]))
        g2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr, va = next(g2.split(df.iloc[tv], groups=df.iloc[tv]["grp"]))
        return df.iloc[tv].iloc[tr], df.iloc[tv].iloc[va], df.iloc[te]

    out["expiration"] = train_classifier("expiration", exp, LOT_FEATURES, "expire_avec_reste",
                                         lambda d: d["ratio_couverture"].values, split_lot)

    # 3) RÉAPPROVISIONNEMENT
    print("\n📦 [3/3] Chargement du dataset RÉAPPROVISIONNEMENT...")
    rea = pd.read_csv(DATA / "dataset_reapprovisionnement.csv")
    print("  -> Découpage temporel...")
    tr, va, te = split_time(rea)
    
    reg = lambda: pipe(STOCK_FEATURES, HistGradientBoostingRegressor(
        loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=24, l2_regularization=1.0,
        early_stopping=False, random_state=SEED))
    
    print("  -> Entraînement du modèle de régression...")
    m = reg().fit(tr[STOCK_FEATURES], tr["demande_30j"])
    
    print("  -> Évaluation et calcul de l'incertitude...")
    pv = np.clip(m.predict(va[STOCK_FEATURES]), 0, None)
    big = pv > 1
    sigma_ratio = float(np.std((va["demande_30j"].values[big] - pv[big]) / pv[big]))
    pt = np.clip(m.predict(te[STOCK_FEATURES]), 0, None)
    base = te["conso_jour_30j"].values * 30
    y = te["demande_30j"].values
    metrics = {
        "mae_modele": round(mean_absolute_error(y, pt), 3),
        "mae_baseline_conso30j_x30": round(mean_absolute_error(y, base), 3),
        "r2_modele": round(r2_score(y, pt), 4),
        "r2_baseline": round(r2_score(y, base), 4),
        "sigma_relatif": round(sigma_ratio, 4),
        "n_test": int(len(y)),
    }
    
    print("  -> Ré-entraînement final et sauvegarde...")
    final = reg().fit(rea[STOCK_FEATURES], rea["demande_30j"])
    joblib.dump({"model": final, "features": STOCK_FEATURES, "sigma_relatif": sigma_ratio,
                 "metrics": metrics}, MODELS / "reapprovisionnement.joblib")
    print("[reapprovisionnement] ✅ Terminé :", json.dumps(metrics, ensure_ascii=False))
    out["reapprovisionnement"] = metrics

    (MODELS / "metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n🎉 Tous les modèles ont été entraînés en {time.time() - start_time:.2f} secondes.")
    print(f"📁 Les fichiers .joblib sont dans le dossier '{MODELS}/'.")

if __name__ == "__main__":
    main()