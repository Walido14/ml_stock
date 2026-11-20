#!/usr/bin/env python3
"""
Inférence Mediko : lit la base SQLite du projet (stock_medicaments, lots_medicaments,
mouvements_stock, medicaments, users), applique les 3 modèles et écrit du JSON.

  python predict.py --db mediko.db [--medecin-id 1] [--asof 2026-11-18] [--out result.json]

--medecin-id : users.id du médecin (par défaut : le premier utilisateur de rôle doctor).
Convention : une SORTIE dont le motif contient perte/expir/casse/retour n'est pas de la consommation
(voir features.LOSS_KEYWORDS) ; tout le reste (ex. "utilisation") l'est.
"""
import argparse, json, math, sqlite3, sys
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import is_consumption, lot_features, niveau, stock_features

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
Z = 1.65            # ~95 % de niveau de service pour le stock de sécurité


# ----------------------------------------------------------------- helpers (réutilisés par les tests)
def consumption_series(mv_stock: pd.DataFrame, asof: date, days: int = 90) -> np.ndarray:
    """Consommation journalière des `days` derniers jours (asof inclus), du plus ancien au plus récent."""
    cons = mv_stock[(mv_stock.type_mouvement == "SORTIE") & mv_stock.motif.map(is_consumption)] \
        .groupby("jour")["quantite"].sum()
    return np.array([cons.get(asof - timedelta(days=k), 0) for k in range(days - 1, -1, -1)], dtype=float)


def days_since_last_entry(mv_stock: pd.DataFrame, asof: date) -> int:
    ent = mv_stock[mv_stock.type_mouvement == "ENTREE"]["jour"]
    return (asof - ent.max()).days if len(ent) else 365


def default_doctor_id(con) -> int:
    row = con.execute("SELECT id FROM users WHERE role = 'doctor' ORDER BY id LIMIT 1").fetchone()
    if not row:
        raise SystemExit("Aucun utilisateur de rôle doctor dans la base.")
    return int(row[0])


def load(db: str, medecin_id: int):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)       # lecture seule : on ne modifie jamais la base
    st = pd.read_sql_query("""SELECT s.id stock_id, s.quantite_actuelle, s.quantite_minimale, s.emplacement,
              m.nom, m.dosage, m.forme, COALESCE(m.categorie,'') categorie
              FROM stock_medicaments s JOIN medicaments m ON m.id = s.medicament_id
              WHERE s.medecin_id = ?""", con, params=(medecin_id,))
    ids = [int(i) for i in st["stock_id"]]
    q = ",".join("?" * len(ids)) or "NULL"
    mv = pd.read_sql_query(f"""SELECT stock_id, type_mouvement, quantite, motif, date_mouvement
                               FROM mouvements_stock WHERE stock_id IN ({q})""", con, params=ids)
    lots = pd.read_sql_query(f"""SELECT stock_id, numero_lot, quantite, date_expiration, date_reception
                                 FROM lots_medicaments WHERE stock_id IN ({q})""", con, params=ids)
    con.close()
    mv["jour"] = pd.to_datetime(mv["date_mouvement"]).dt.date
    return st, mv, lots


# ... (Gardez tous les imports et fonctions helpers identiques) ...

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--medecin-id", type=int)
    ap.add_argument("--asof", default=str(date.today()))
    ap.add_argument("--out")
    a = ap.parse_args()
    asof = datetime.strptime(a.asof, "%Y-%m-%d").date()

    print(f"🔌 [1/5] Connexion à la base de données : {a.db}...")
    
    print("🧠 [2/5] Chargement des 3 modèles (.joblib)...")
    b_rup, b_exp, b_rea = (joblib.load(MODELS / f"{n}.joblib") for n in ("rupture", "expiration", "reapprovisionnement"))
    
    if a.medecin_id is None:
        print("   -> Recherche du médecin par défaut...")
        c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True); a.medecin_id = default_doctor_id(c); c.close()
    
    print(f"   -> Utilisation du médecin ID : {a.medecin_id} (Date de référence : {asof})")
    print("📥 [3/5] Extraction des données (stocks, mouvements, lots)...")
    st, mv, lots = load(a.db, a.medecin_id)
    mv = mv[mv["jour"] <= asof]

    print(f"   -> {len(st)} articles en stock trouvés. Préparation des features...")
    rows = []
    for r in st.itertuples():
        m = mv[mv.stock_id == r.stock_id]
        rows.append(stock_features(r.quantite_actuelle, r.quantite_minimale, consumption_series(m, asof),
                                   days_since_last_entry(m, asof), asof, r.forme, r.categorie))
    if not rows:
        print("⚠️ Aucun article trouvé pour ce médecin. Génération d'un JSON vide.")
        print(json.dumps({"medecin_id": a.medecin_id, "date": str(asof), "stocks": []})); return
    
    X = pd.DataFrame(rows)
    
    print("🤖 [4/5] Application des modèles de prédiction...")
    p_rup = b_rup["model"].predict_proba(X[b_rup["features"]])[:, 1]
    d30 = np.clip(b_rea["model"].predict(X[b_rea["features"]]), 0, None)

    lots = lots[lots.quantite > 0].copy()
    lots = lots[lots.date_expiration.notna()]
    lots["exp"] = pd.to_datetime(lots["date_expiration"]).dt.date
    lots["rec"] = pd.to_datetime(lots["date_reception"]).dt.date
    
    res = []
    print("   -> Traitement des lots et calcul des risques d'expiration...")
    for i, r in enumerate(st.itertuples()):
        L = lots[lots.stock_id == r.stock_id].sort_values("exp")
        conso30 = X.loc[i, "conso_jour_30j"]
        lot_out, q_perdu = [], 0.0
        for l in L.itertuples():
            restant = (l.exp - asof).days
            if restant <= 0:
                p, lvl = 1.0, "EXPIRE"
            else:
                avant = int(L[L.exp < l.exp]["quantite"].sum())
                f = lot_features(l.quantite, restant, (asof - l.rec).days, (l.exp - l.rec).days, avant,
                                 conso30, r.quantite_actuelle, r.forme, r.categorie, asof)
                p = float(b_exp["model"].predict_proba(pd.DataFrame([f])[b_exp["features"]])[0, 1])
                lvl = niveau(p, b_exp["seuil_moyen"], b_exp["seuil_eleve"])
            q_perdu += p * l.quantite
            lot_out.append({"numero_lot": l.numero_lot, "quantite": int(l.quantite), "date_expiration": str(l.exp),
                            "jours_restants": int(restant), "risque_expiration": round(p, 3), "niveau": lvl})

        sigma = b_rea["sigma_relatif"] * d30[i]
        securite = Z * sigma
        utile = max(r.quantite_actuelle - q_perdu, 0.0)
        a_cmd = max(0, math.ceil(d30[i] + securite - utile))
        if a_cmd > 20: a_cmd = int(math.ceil(a_cmd / 5) * 5)
        
        lvl_r = "RUPTURE" if r.quantite_actuelle == 0 else niveau(p_rup[i], b_rup["seuil_moyen"], b_rup["seuil_eleve"])
        couverture = float(X.loc[i, "jours_couverture"])
        critique = r.quantite_actuelle > 0 and couverture < 5
        
        priorite = ("URGENTE" if (lvl_r in ("RUPTURE", "ELEVE") or critique) and a_cmd > 0 else
                    "NORMALE" if a_cmd > 0 and (lvl_r == "MOYEN" or r.quantite_actuelle <= r.quantite_minimale) else
                    "FAIBLE" if a_cmd > 0 else "AUCUNE")
        
        raison = []
        if r.quantite_actuelle == 0: raison.append("stock épuisé")
        elif r.quantite_actuelle <= r.quantite_minimale: raison.append("sous le seuil minimum")
        if critique: raison.append(f"couverture de {couverture:.1f} jour(s) seulement")
        if p_rup[i] >= b_rup["seuil_moyen"]: raison.append(f"risque de rupture à 14 j ({p_rup[i]:.0%})")
        if q_perdu >= 1: raison.append(f"~{q_perdu:.0f} unité(s) risquent d'expirer (non comptées)")
        
        res.append({
            "stock_id": int(r.stock_id), "medicament": f"{r.nom} {r.dosage} ({r.forme})".replace("  ", " "),
            "categorie": r.categorie,
            "quantite_actuelle": int(r.quantite_actuelle), "quantite_minimale": int(r.quantite_minimale),
            "rupture": {"probabilite_14j": round(float(p_rup[i]), 3), "niveau": lvl_r,
                        "jours_couverture": round(float(X.loc[i, "jours_couverture"]), 1)},
            "expiration": {"quantite_a_risque": round(q_perdu, 1), "lots": lot_out},
            "reapprovisionnement": {"demande_prevue_30j": round(float(d30[i]), 1), "stock_securite": round(float(securite), 1),
                                    "quantite_a_commander": int(a_cmd), "priorite": priorite,
                                    "raison": "; ".join(raison) or "stock suffisant"},
        })
    
    order = {"URGENTE": 0, "NORMALE": 1, "FAIBLE": 2, "AUCUNE": 3}
    res.sort(key=lambda x: (order[x["reapprovisionnement"]["priorite"]], -x["rupture"]["probabilite_14j"]))
    
    print("💾 [5/5] Génération du fichier JSON final...")
    out = json.dumps({"medecin_id": a.medecin_id, "date": str(asof), "stocks": res}, ensure_ascii=False, indent=2)
    
    if a.out:
        Path(a.out).write_text(out, encoding="utf-8")
        print(f"   ✅ Succès ! Résultats sauvegardés dans : {a.out}")
    else:
        print("   ✅ Succès ! Affichage des résultats ci-dessous :\n")
        sys.stdout.write(out + "\n")

if __name__ == "__main__":
    main()