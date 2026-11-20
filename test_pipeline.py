#!/usr/bin/env python3
"""
Validation complète des modèles Mediko — à lancer après chaque entraînement :

    python test_pipeline.py            # tout (≈ 1 min)

  1. Seuils de qualité  : les modèles battent leur règle simple (metrics.json)
  2. Cohérence train/serving : features recalculées DEPUIS LA BASE == features du dataset
  3. Scénarios métier   : cas construits à la main (rupture imminente, surstock, péremption…)
  4. Invariants         : probabilités dans [0,1], quantités >= 0, JSON complet, base non modifiée
Code de sortie 0 = tout est bon, 1 = au moins un test échoue (utilisable en CI).
"""
import hashlib, json, subprocess, sys, tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3

import features as F
from predict import consumption_series, days_since_last_entry, load

HERE = Path(__file__).resolve().parent
ASOF = date(2026, 11, 17)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'OK ' if ok else 'ÉCHEC'}] {name}" + (f"  — {detail}" if detail else ""))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_predict(db, asof=ASOF):
    p = subprocess.run([sys.executable, str(HERE / "predict.py"), "--db", db, "--asof", str(asof)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=HERE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    return json.loads(p.stdout)


# ------------------------------------------------------------------ 1. seuils de qualité
def test_metrics():
    m = json.loads((HERE / "models" / "metrics.json").read_text(encoding="utf-8"))
    r, e, g = m["rupture"], m["expiration"], m["reapprovisionnement"]
    check("rupture : AUC >= 0.75", r["auc"] >= 0.75, f"AUC={r['auc']}")
    check("rupture : bat la règle « jours de couverture »", r["auc"] > r["auc_baseline_regle_simple"],
          f"{r['auc']} vs {r['auc_baseline_regle_simple']}")
    check("expiration : AUC >= 0.90 et précision moyenne >= 0.5", e["auc"] >= 0.90 and e["average_precision"] >= 0.5,
          f"AUC={e['auc']} AP={e['average_precision']}")
    check("réappro : MAE < baseline (conso 30 j × 30)", g["mae_modele"] < g["mae_baseline_conso30j_x30"],
          f"{g['mae_modele']} vs {g['mae_baseline_conso30j_x30']}")
    check("réappro : R² >= 0.80", g["r2_modele"] >= 0.80, f"R²={g['r2_modele']}")
    check("calibration rupture : Brier < taux×(1-taux)", r["brier"] < r["taux_positif"] * (1 - r["taux_positif"]) * 1.0 + 1e-9,
          f"Brier={r['brier']}")


# ------------------------------------------------------------------ 2. cohérence train / serving
def test_skew(db):
    ds = pd.read_csv(HERE / "data" / "dataset_rupture.csv")
    st, mv, _ = load(db, 1)
    ds = ds[ds.stock_id.isin(st.stock_id) & (ds.cabinet == 1)].sample(300, random_state=0)
    info = st.set_index("stock_id")
    signed = mv.assign(s=np.where(mv.type_mouvement == "SORTIE", -mv.quantite, mv.quantite))
    worst, bad_qty = 0.0, 0
    for r in ds.itertuples():
        asof = date.fromisoformat(r.date_snapshot)
        m = mv[(mv.stock_id == r.stock_id) & (mv.jour <= asof)]
        qty = int(signed[(signed.stock_id == r.stock_id) & (signed.jour <= asof)].s.sum())
        bad_qty += qty != r.quantite_actuelle
        row = F.stock_features(qty, r.quantite_minimale, consumption_series(m, asof),
                               days_since_last_entry(m, asof), asof, info.loc[r.stock_id, "forme"],
                               info.loc[r.stock_id, "categorie"])
        for k in F.STOCK_NUMERIC:
            worst = max(worst, abs(row[k] - getattr(r, k)))
    check("quantité reconstruite depuis mouvements_stock == quantité historique", bad_qty == 0, f"{bad_qty}/300 écarts")
    check("features recalculées depuis la base == features d'entraînement", worst < 1e-6, f"écart max = {worst:.2e}")


# ------------------------------------------------------------------ 3. scénarios métier
def build_scenarios(db):
    con = sqlite3.connect(db)
    con.execute("PRAGMA foreign_keys = ON")
    doctor = con.execute("SELECT id FROM users WHERE role='doctor'").fetchone()[0]
    d = lambda k: str(ASOF - timedelta(days=k))

    def make(name, lots, per_day, qmin, forme="Comprimé", cat="Antalgique", entree_il_y_a=20):
        """lots = [(numero, quantite, jours_avant_expiration)] ; per_day = consommation moyenne sur 90 j"""
        with con:
            mid = con.execute("INSERT INTO medicaments (nom, forme, categorie) VALUES (?,?,?)", (name, forme, cat)).lastrowid
            sid = con.execute("INSERT INTO stock_medicaments (medecin_id, medicament_id, quantite_minimale) VALUES (?,?,?)",
                              (doctor, mid, qmin)).lastrowid
            for num, q, exp in lots:
                con.execute("INSERT INTO lots_medicaments (stock_id, numero_lot, quantite, date_expiration, date_reception) VALUES (?,?,?,?,?)",
                            (sid, num, q, d(-exp), d(100)))
            con.execute("INSERT INTO mouvements_stock (stock_id,type_mouvement,quantite,motif,date_mouvement,utilisateur_id) VALUES (?,?,?,?,?,?)",
                        (sid, "ENTREE", 50, "achat", f"{d(entree_il_y_a)} 09:00:00", doctor))
            for k in range(90):
                if per_day > 0:
                    con.execute("INSERT INTO mouvements_stock (stock_id,type_mouvement,quantite,motif,date_mouvement,utilisateur_id) VALUES (?,?,?,?,?,?)",
                                (sid, "SORTIE", per_day, "utilisation", f"{d(k)} 10:00:00", doctor))
        return name

    make("TEST A rupture imminente", [("A1", 12, 300)], 6, 30)
    make("TEST B surstock",          [("B1", 300, 400)], 1, 10)
    make("TEST C peremption proche", [("C1", 90, 10)],  1, 5)           # 90 u. expirent dans 10 j, conso 1/j
    make("TEST D lot expire",        [("D1", 5, -3), ("D2", 40, 300)], 1, 5)
    make("TEST E stock epuise",      [], 3, 10)
    make("TEST F1 conso faible",     [("F1", 40, 300)], 1, 10)
    make("TEST F2 conso forte",      [("F2", 40, 300)], 6, 10)
    con.close()


def test_scenarios(db):
    out = run_predict(db)
    S = {s["medicament"].split(" (")[0]: s for s in out["stocks"] if s["medicament"].startswith("TEST")}
    A, B, C, D, E = (S[f"TEST {x}"] for x in ("A rupture imminente", "B surstock", "C peremption proche", "D lot expire", "E stock epuise"))
    F1, F2 = S["TEST F1 conso faible"], S["TEST F2 conso forte"]
    q = lambda s: s["reapprovisionnement"]

    check("A rupture imminente (2 j de couverture) → risque ÉLEVÉ/MOYEN, commande > 0, priorité URGENTE",
          A["rupture"]["niveau"] in ("ELEVE", "MOYEN") and q(A)["quantite_a_commander"] > 0 and q(A)["priorite"] == "URGENTE",
          f"p={A['rupture']['probabilite_14j']} cmd={q(A)['quantite_a_commander']} {q(A)['priorite']}")
    check("B surstock → risque de rupture FAIBLE, aucune commande",
          B["rupture"]["niveau"] == "FAIBLE" and q(B)["quantite_a_commander"] == 0,
          f"p={B['rupture']['probabilite_14j']} cmd={q(B)['quantite_a_commander']}")
    lot = C["expiration"]["lots"][0]
    check("C 90 unités expirant dans 10 j (conso 1/j) → risque d'expiration ÉLEVÉ", lot["niveau"] == "ELEVE" and lot["risque_expiration"] >= 0.6,
          f"p={lot['risque_expiration']}")
    check("C les unités à risque ne sont pas comptées → on propose de commander",
          C["expiration"]["quantite_a_risque"] > 30, f"à risque ≈ {C['expiration']['quantite_a_risque']}")
    lots = {l["numero_lot"]: l for l in D["expiration"]["lots"]}
    check("D lot déjà expiré → niveau EXPIRE (risque 1.0), l'autre lot est évalué", lots["D1"]["niveau"] == "EXPIRE" and lots["D2"]["niveau"] != "EXPIRE")
    check("E stock épuisé → RUPTURE + URGENTE + commande > 0",
          E["rupture"]["niveau"] == "RUPTURE" and q(E)["priorite"] == "URGENTE" and q(E)["quantite_a_commander"] > 0)
    check("F plus la consommation est forte, plus la demande prévue et le risque de rupture augmentent",
          q(F2)["demande_prevue_30j"] > q(F1)["demande_prevue_30j"] and F2["rupture"]["probabilite_14j"] > F1["rupture"]["probabilite_14j"],
          f"demande {q(F1)['demande_prevue_30j']}→{q(F2)['demande_prevue_30j']}, p {F1['rupture']['probabilite_14j']}→{F2['rupture']['probabilite_14j']}")
    return out


# ------------------------------------------------------------------ 4. invariants
def test_invariants(out, db, hash_before):
    stocks = out["stocks"]
    ok = True
    for s in stocks:
        ok &= 0 <= s["rupture"]["probabilite_14j"] <= 1
        ok &= s["reapprovisionnement"]["quantite_a_commander"] >= 0 and s["reapprovisionnement"]["demande_prevue_30j"] >= 0
        ok &= all(0 <= l["risque_expiration"] <= 1 for l in s["expiration"]["lots"])
        ok &= s["reapprovisionnement"]["priorite"] in ("URGENTE", "NORMALE", "FAIBLE", "AUCUNE")
    check("probabilités dans [0,1], quantités >= 0, priorités valides", ok, f"{len(stocks)} stocks")
    need = {"stock_id", "medicament", "quantite_actuelle", "rupture", "expiration", "reapprovisionnement"}
    check("JSON complet pour chaque stock", all(need <= s.keys() for s in stocks))
    prio = [s["reapprovisionnement"]["priorite"] for s in stocks]
    rank = {"URGENTE": 0, "NORMALE": 1, "FAIBLE": 2, "AUCUNE": 3}
    check("liste triée par priorité", [rank[p] for p in prio] == sorted(rank[p] for p in prio))
    check("la base n'a PAS été modifiée par predict.py (lecture seule)", sha(db) == hash_before)
    again = run_predict(db)
    check("résultat déterministe (2 exécutions identiques)", again == out)


def main():
    sys.stdout.reconfigure(encoding="utf-8")      # Windows : évite les erreurs d'affichage (é, →, ²)
    print("=== 1. Seuils de qualité ===");            test_metrics()
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "test.db")
        r = subprocess.run([sys.executable, str(HERE / "import_csv_to_db.py"), "--db", db, "--demo"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=HERE)
        check("import des CSV dans le schéma Mediko + contrôles d'intégrité", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])
        print("=== 2. Cohérence entraînement / base ==="); test_skew(db)
        print("=== 3. Scénarios métier ===");       build_scenarios(db); h = sha(db); out = test_scenarios(db)
        print("=== 4. Invariants ===");             test_invariants(out, db, h)
    print(f"\n{sum(results)}/{len(results)} tests réussis")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
