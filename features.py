"""
Mediko ML - définition UNIQUE des features.
Utilisé à l'identique par generate_dataset.py / train_models.py (entraînement)
et predict.py (inférence sur la vraie base SQLite) => pas de décalage train/serving.
"""
from __future__ import annotations
import numpy as np

CATEGORICAL = ["forme", "categorie"]

STOCK_NUMERIC = [
    "quantite_actuelle", "quantite_minimale", "ratio_qte_min",
    "ventes_7j", "ventes_30j", "ventes_90j",
    "conso_jour_7j", "conso_jour_30j", "tendance_7_30", "std_30j",
    "jours_couverture", "jours_depuis_derniere_entree", "mois", "jour_semaine",
]
STOCK_FEATURES = STOCK_NUMERIC + CATEGORICAL

LOT_NUMERIC = [
    "quantite_lot", "jours_restants", "age_lot_jours", "duree_vie_totale",
    "pct_vie_restante", "quantite_avant_fefo", "quantite_stock",
    "conso_jour_30j", "couverture_lot_jours", "ratio_couverture", "mois",
]
LOT_FEATURES = LOT_NUMERIC + CATEGORICAL

# Mots-clés qui, dans `motif`, indiquent une SORTIE qui n'est PAS une consommation
LOSS_KEYWORDS = ("perte", "expir", "casse", "retour", "périm", "perim")


def is_consumption(motif: str | None) -> bool:
    m = (motif or "").lower()
    return not any(k in m for k in LOSS_KEYWORDS)


def _pad(ventes, n: int = 90) -> np.ndarray:
    v = np.asarray(ventes, dtype=float)[-n:]
    if len(v) < n:
        v = np.concatenate([np.zeros(n - len(v)), v])
    return v


def stock_features(qty, qty_min, ventes, jours_depuis_entree, asof, forme, categorie) -> dict:
    """ventes = consommation journalière (oldest -> newest), jusqu'à `asof` inclus."""
    v = _pad(ventes)
    v7, v30, v90 = v[-7:].sum(), v[-30:].sum(), v.sum()
    c7, c30 = v7 / 7.0, v30 / 30.0
    return {
        "quantite_actuelle": int(qty),
        "quantite_minimale": int(qty_min),
        "ratio_qte_min": float(qty) / max(int(qty_min), 1),
        "ventes_7j": float(v7), "ventes_30j": float(v30), "ventes_90j": float(v90),
        "conso_jour_7j": float(c7), "conso_jour_30j": float(c30),
        "tendance_7_30": (c7 + 0.05) / (c30 + 0.05),
        "std_30j": float(v[-30:].std()),
        "jours_couverture": min(float(qty) / max(c30, 0.05), 365.0),
        "jours_depuis_derniere_entree": float(min(jours_depuis_entree, 365)),
        "mois": int(asof.month),
        "jour_semaine": int(asof.weekday()),
        "forme": forme,
        "categorie": categorie,
    }


def lot_features(qty_lot, jours_restants, age_lot, duree_vie, qty_avant, conso30,
                 qty_stock, forme, categorie, asof) -> dict:
    """qty_avant = quantité des lots du même stock qui expirent AVANT celui-ci (FEFO)."""
    couverture = min((qty_avant + qty_lot) / max(conso30, 0.05), 3650.0)
    return {
        "quantite_lot": int(qty_lot),
        "jours_restants": int(jours_restants),
        "age_lot_jours": int(age_lot),
        "duree_vie_totale": int(duree_vie),
        "pct_vie_restante": float(jours_restants) / max(float(duree_vie), 1.0),
        "quantite_avant_fefo": int(qty_avant),
        "quantite_stock": int(qty_stock),
        "conso_jour_30j": float(conso30),
        "couverture_lot_jours": couverture,
        "ratio_couverture": couverture / max(float(jours_restants), 1.0),
        "mois": int(asof.month),
        "forme": forme,
        "categorie": categorie,
    }


def niveau(p: float, seuil_moyen: float, seuil_eleve: float = 0.6) -> str:
    seuil_eleve = max(seuil_eleve, seuil_moyen)
    return "ELEVE" if p >= seuil_eleve else "MOYEN" if p >= seuil_moyen else "FAIBLE"
