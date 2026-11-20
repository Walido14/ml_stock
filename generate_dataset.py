#!/usr/bin/env python3
"""
Génère les CSV propres de Mediko, prêts à être importés :

  data/medicaments.csv, stock_medicaments.csv, lots_medicaments.csv, mouvements_stock.csv
      -> exactement les colonnes des 4 tables de la migration SQL
  data/dataset_rupture.csv, dataset_expiration.csv, dataset_reapprovisionnement.csv
      -> jeux ML prêts à l'entraînement (features + cible)

!! DONNÉES SIMULÉES (cabinets fictifs, consommation saisonnière, délais fournisseur,
   FEFO, pertes par expiration). Elles servent à développer/valider le pipeline.
   Dès que Mediko a de vrais mouvements, ré-entraînez avec train_models.py.

Usage :
  python generate_dataset.py
  python generate_dataset.py --doctor-id 1 --secretary-id 2   # ids de la table users de Mediko
"""
import argparse, math, random
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from features import stock_features, lot_features

# nom, principe_actif, dosage, forme, categorie, description
CATALOGUE = [
 ("Paracétamol","Paracétamol","500 mg","Comprimé","Antalgique","Antalgique et antipyrétique, douleurs légères à modérées et fièvre."),
 ("Paracétamol","Paracétamol","1000 mg","Comprimé","Antalgique","Antalgique et antipyrétique, forme adulte."),
 ("Paracétamol","Paracétamol","24 mg/ml","Sirop","Antalgique","Antalgique et antipyrétique pédiatrique."),
 ("Tramadol","Tramadol","50 mg","Gélule","Antalgique","Antalgique opioïde faible, douleurs modérées à intenses."),
 ("Ibuprofène","Ibuprofène","400 mg","Comprimé","AINS","Anti-inflammatoire non stéroïdien, douleur et fièvre."),
 ("Ibuprofène","Ibuprofène","20 mg/ml","Sirop","AINS","Anti-inflammatoire pédiatrique."),
 ("Diclofénac","Diclofénac sodique","50 mg","Comprimé","AINS","Anti-inflammatoire, douleurs rhumatismales."),
 ("Diclofénac","Diclofénac sodique","75 mg/3 ml","Injection","AINS","Anti-inflammatoire injectable."),
 ("Kétoprofène","Kétoprofène","100 mg","Comprimé","AINS","Anti-inflammatoire, douleurs articulaires."),
 ("Naproxène","Naproxène","500 mg","Comprimé","AINS","Anti-inflammatoire, rhumatologie."),
 ("Amoxicilline","Amoxicilline","500 mg","Gélule","Antibiotique","Pénicilline à large spectre."),
 ("Amoxicilline","Amoxicilline","250 mg/5 ml","Suspension buvable","Antibiotique","Pénicilline, forme pédiatrique."),
 ("Amoxicilline/Acide clavulanique","Amoxicilline + acide clavulanique","1 g/125 mg","Comprimé","Antibiotique","Pénicilline associée à un inhibiteur de bêta-lactamases."),
 ("Azithromycine","Azithromycine","500 mg","Comprimé","Antibiotique","Macrolide, infections respiratoires."),
 ("Clarithromycine","Clarithromycine","500 mg","Comprimé","Antibiotique","Macrolide."),
 ("Ciprofloxacine","Ciprofloxacine","500 mg","Comprimé","Antibiotique","Fluoroquinolone."),
 ("Ceftriaxone","Ceftriaxone","1 g","Injection","Antibiotique","Céphalosporine de 3e génération injectable."),
 ("Céfixime","Céfixime","200 mg","Gélule","Antibiotique","Céphalosporine orale de 3e génération."),
 ("Métronidazole","Métronidazole","500 mg","Comprimé","Antibiotique","Antibiotique et antiparasitaire."),
 ("Doxycycline","Doxycycline","100 mg","Gélule","Antibiotique","Cycline."),
 ("Amlodipine","Amlodipine","5 mg","Comprimé","Antihypertenseur","Inhibiteur calcique."),
 ("Ramipril","Ramipril","5 mg","Comprimé","Antihypertenseur","IEC."),
 ("Bisoprolol","Bisoprolol","5 mg","Comprimé","Antihypertenseur","Bêta-bloquant cardiosélectif."),
 ("Losartan","Losartan","50 mg","Comprimé","Antihypertenseur","Antagoniste des récepteurs de l'angiotensine II."),
 ("Hydrochlorothiazide","Hydrochlorothiazide","25 mg","Comprimé","Antihypertenseur","Diurétique thiazidique."),
 ("Metformine","Metformine","850 mg","Comprimé","Antidiabétique","Antidiabétique oral, diabète de type 2."),
 ("Gliclazide","Gliclazide","60 mg","Comprimé","Antidiabétique","Sulfamide hypoglycémiant."),
 ("Insuline glargine","Insuline glargine","100 UI/ml","Injection","Antidiabétique","Insuline basale d'action prolongée (conservation au froid)."),
 ("Oméprazole","Oméprazole","20 mg","Gélule","Gastro-entérologie","Inhibiteur de la pompe à protons."),
 ("Pantoprazole","Pantoprazole","40 mg","Comprimé","Gastro-entérologie","Inhibiteur de la pompe à protons."),
 ("Dompéridone","Dompéridone","10 mg","Comprimé","Gastro-entérologie","Antiémétique, troubles digestifs."),
 ("Lopéramide","Lopéramide","2 mg","Gélule","Gastro-entérologie","Antidiarrhéique."),
 ("Phloroglucinol","Phloroglucinol","80 mg","Comprimé","Gastro-entérologie","Antispasmodique."),
 ("Cétirizine","Cétirizine","10 mg","Comprimé","Antihistaminique","Antihistaminique H1, allergies."),
 ("Loratadine","Loratadine","10 mg","Comprimé","Antihistaminique","Antihistaminique H1, allergies."),
 ("Desloratadine","Desloratadine","5 mg","Comprimé","Antihistaminique","Antihistaminique H1."),
 ("Salbutamol","Salbutamol","100 µg/dose","Aérosol","Respiratoire","Bronchodilatateur d'action rapide."),
 ("Ambroxol","Ambroxol","15 mg/5 ml","Sirop","Respiratoire","Fluidifiant bronchique."),
 ("Montélukast","Montélukast","10 mg","Comprimé","Respiratoire","Antiasthmatique, antagoniste des leucotriènes."),
 ("Prednisolone","Prednisolone","20 mg","Comprimé","Corticoïde","Corticoïde oral."),
 ("Dexaméthasone","Dexaméthasone","4 mg/ml","Injection","Corticoïde","Corticoïde injectable."),
 ("Hydrocortisone","Hydrocortisone","1 %","Crème","Dermatologie","Corticoïde local."),
 ("Acide fusidique","Acide fusidique","2 %","Pommade","Dermatologie","Antibiotique local."),
 ("Povidone iodée","Povidone iodée","10 %","Solution","Dermatologie","Antiseptique cutané."),
 ("Tobramycine","Tobramycine","0,3 %","Collyre","Ophtalmologie","Antibiotique ophtalmique."),
 ("Vitamine D3","Cholécalciférol","100 000 UI","Ampoule buvable","Vitamine","Supplémentation en vitamine D."),
 ("Sulfate ferreux","Fer (sulfate ferreux)","80 mg","Comprimé","Vitamine","Traitement des carences en fer."),
 ("Acide folique","Acide folique","5 mg","Comprimé","Vitamine","Vitamine B9."),
 ("Albendazole","Albendazole","400 mg","Comprimé","Antiparasitaire","Antihelminthique."),
 ("Chlorure de sodium","Chlorure de sodium","0,9 %","Solution","Dermatologie","Sérum physiologique, lavage et dilution."),
]

CAT_BASE = {"Antalgique":4.0,"AINS":2.5,"Antibiotique":2.0,"Antihypertenseur":1.5,"Antidiabétique":1.2,
            "Gastro-entérologie":1.8,"Antihistaminique":1.5,"Respiratoire":1.2,"Corticoïde":0.8,
            "Dermatologie":0.7,"Ophtalmologie":0.5,"Vitamine":1.0,"Antiparasitaire":0.4}
SHELF = {"Comprimé":(300,1000),"Gélule":(300,1000),"Sirop":(120,400),"Suspension buvable":(100,300),
         "Injection":(200,700),"Crème":(200,600),"Pommade":(200,600),"Solution":(200,700),
         "Aérosol":(300,800),"Collyre":(100,400),"Ampoule buvable":(300,800)}
LABS = ["LabPharma Tunis","MediSud","Générix","Sahel Pharma","Carthage Labs","Atlas Médical"]
EMPL = ["Armoire A","Armoire B","Étagère 1","Étagère 2","Tiroir bureau"]
DOW = [1.0, 1.05, 1.0, 1.0, 0.95, 0.5, 0.15]      # lun..dim
WINTER = {"Antibiotique","Respiratoire","Antalgique","AINS","Corticoïde"}


def season(cat, doy):
    if cat in WINTER:        return 1 + 0.5 * math.cos(2 * math.pi * (doy - 20) / 365)
    if cat == "Antihistaminique": return 1 + 0.6 * math.cos(2 * math.pi * (doy - 100) / 365)
    return 1 + 0.1 * math.cos(2 * math.pi * (doy - 20) / 365)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--end", default="2026-11-17", help="dernier jour simulé (YYYY-MM-DD)")
    ap.add_argument("--cabinets", type=int, default=4,
                    help="cabinets simulés pour l'entraînement ML ; seul le cabinet 1 est écrit dans les CSV des 4 tables")
    ap.add_argument("--step", type=int, default=3, help="pas (jours) entre deux instantanés ML")
    ap.add_argument("--doctor-id", type=int, default=1, help="users.id du médecin (rôle doctor)")
    ap.add_argument("--secretary-id", type=int, default=2, help="users.id de la secrétaire (rôle secretary)")
    a = ap.parse_args()

    rng, prng = np.random.default_rng(a.seed), random.Random(a.seed)
    counters = {}
    def new_id(kind="x"):
        counters[kind] = counters.get(kind, 0) + 1
        return counters[kind]
    end = datetime.strptime(a.end, "%Y-%m-%d").date()
    start = end - timedelta(days=a.days - 1)
    N = a.days

    cabinets = list(range(1, a.cabinets + 1))         # cabinet 1 = le cabinet "réel" écrit dans les CSV
    users = {c: (a.doctor_id, a.secretary_id) for c in cabinets}

    # --- catalogue
    meds = []
    for (nom, pa, dosage, forme, cat, desc) in CATALOGUE:
        meds.append(dict(id=new_id('med'), nom=nom, principe_actif=pa, dosage=dosage, forme=forme,
                         laboratoire=prng.choice(LABS), categorie=cat, description=desc,
                         created_at=f"{start - timedelta(days=30)} 08:00:00"))

    stocks, lots_rows, mvts = [], [], []
    snap_stock, snap_lot = [], []

    def ts(day_i, h, m=None):
        d = start + timedelta(days=day_i)
        mm = prng.randint(0, 59) if m is None else m
        return f"{d} {h:02d}:{mm:02d}:{prng.randint(0,59):02d}"

    for m_id in cabinets:
        n_st = int(rng.integers(22, 33))
        for mi in rng.choice(len(meds), size=n_st, replace=False):
            med = meds[mi]
            cat, forme = med["categorie"], med["forme"]
            lam = max(0.15, CAT_BASE[cat] * float(np.exp(rng.normal(0, 0.6))))
            overbuy = float(np.exp(rng.normal(0.25, 0.8)))
            qmin = max(3, math.ceil(lam * rng.uniform(6, 12)))
            growth = float(rng.normal(0, 0.15))
            stock_id = new_id('stock')
            u_med, u_sec = users[m_id]
            lo, hi = SHELF[forme]
            lots, pending, seq = [], [], 0
            ventes = np.zeros(N); qty_end = np.zeros(N, dtype=int)
            last_entree = 0
            emp = "Réfrigérateur" if "insuline" in med["nom"].lower() else prng.choice(EMPL)

            def make_lot(i, qty, life, motif):
                nonlocal seq
                seq += 1
                d = start + timedelta(days=i)
                lot = dict(id=new_id('lot'), num=f"L{d:%y%m}-{seq:03d}", rec=i, exp=i + life, qty0=qty,
                           qty=qty, closed=False, perdu=0)
                lots.append(lot)
                mvts.append(dict(id=0, stock_id=stock_id, type_mouvement="ENTREE", quantite=qty,
                                 motif=motif, date_mouvement=ts(i, prng.randint(9, 11)), utilisateur_id=u_sec))

            # lot initial
            make_lot(0, math.ceil(lam * rng.uniform(15, 40) * overbuy), int(rng.uniform(0.5, 1.0) * rng.integers(lo, hi)),
                     "stock initial")

            for i in range(N):
                d = start + timedelta(days=i)
                # 1) arrivées
                for o in [p for p in pending if p["arr"] == i]:
                    make_lot(i, o["qty"], int(rng.integers(lo, hi) * rng.uniform(0.2, 1.0)), "achat"); last_entree = i
                    pending.remove(o)
                # 2) expiration
                for L in lots:
                    if not L["closed"] and L["exp"] <= i:
                        if L["qty"] > 0:
                            L["perdu"] = L["qty"]
                            mvts.append(dict(id=0, stock_id=stock_id, type_mouvement="SORTIE",
                                             quantite=L["qty"], motif="perte (expiration)",
                                             date_mouvement=ts(i, 7, 30), utilisateur_id=u_sec))
                            L["qty"] = 0
                        L["closed"] = True
                # 3) demande (FEFO)
                mult = season(cat, d.timetuple().tm_yday) * DOW[d.weekday()] * (1 + growth * i / 365)
                demand = int(rng.poisson(lam * mult * rng.gamma(4, 0.25)))
                act = sorted([L for L in lots if L["qty"] > 0], key=lambda L: L["exp"])
                avail = sum(L["qty"] for L in act); sold = min(demand, avail); rest = sold
                for L in act:
                    t = min(L["qty"], rest); L["qty"] -= t; rest -= t
                    if rest == 0: break
                if sold > 0:
                    mvts.append(dict(id=0, stock_id=stock_id, type_mouvement="SORTIE", quantite=sold,
                                     motif="utilisation", date_mouvement=ts(i, prng.randint(8, 18)),
                                     utilisateur_id=prng.choice([u_med, u_sec])))
                ventes[i] = sold
                # 4) ajustements d'inventaire
                if rng.random() < 0.004:
                    act = sorted([L for L in lots if L["qty"] > 0], key=lambda L: L["exp"])
                    if act:
                        k = int(rng.integers(1, min(3, act[0]["qty"]) + 1)); act[0]["qty"] -= k
                        mvts.append(dict(id=0, stock_id=stock_id, type_mouvement="AJUSTEMENT", quantite=-k,
                                         motif="correction inventaire (casse)", date_mouvement=ts(i, 17),
                                         utilisateur_id=u_sec))
                elif rng.random() < 0.001:
                    act = sorted([L for L in lots if not L["closed"]], key=lambda L: L["exp"])
                    if act:
                        k = int(rng.integers(1, 4)); act[-1]["qty"] += k
                        mvts.append(dict(id=0, stock_id=stock_id, type_mouvement="AJUSTEMENT", quantite=k,
                                         motif="correction inventaire", date_mouvement=ts(i, 17),
                                         utilisateur_id=u_sec))
                for L in lots:
                    if not L["closed"] and L["qty"] == 0: L["closed"] = True      # lot épuisé
                q = sum(L["qty"] for L in lots); qty_end[i] = q
                # 5) réapprovisionnement (délai fournisseur variable, parfois long)
                if q <= qmin and not pending and rng.random() < 0.8:
                    lead = int(rng.integers(2, 11)) + (int(rng.integers(10, 26)) if rng.random() < 0.08 else 0)
                    oq = math.ceil(lam * rng.uniform(18, 80) * overbuy)
                    if oq > 20: oq = int(math.ceil(oq / 5) * 5)
                    pending.append(dict(arr=i + lead, qty=oq))
                # 6) instantanés ML (fin de journée)
                if i >= 60 and i % a.step == 0:
                    v = ventes[: i + 1]
                    sf = stock_features(q, qmin, v, i - last_entree, d, forme, cat)
                    snap_stock.append(dict(stock_id=stock_id, cabinet=m_id, date_snapshot=str(d), i=i, **sf))
                    c30 = sf["conso_jour_30j"]
                    for L in lots:
                        if L["qty"] > 0 and L["exp"] > i:
                            avant = sum(x["qty"] for x in lots if x["qty"] > 0 and x["exp"] < L["exp"])
                            lf = lot_features(L["qty"], L["exp"] - i, i - L["rec"], L["exp"] - L["rec"],
                                              avant, c30, q, forme, cat, d)
                            snap_lot.append(dict(stock_id=stock_id, numero_lot=L["num"], date_snapshot=str(d),
                                                 date_reception=str(start + timedelta(days=L["rec"])),
                                                 _lot=L, **lf))

            # --- sorties de simulation -> tables + labels
            for r in snap_stock:
                if r["stock_id"] != stock_id or "rupture_14j" in r: continue
                i = r.pop("i")
                r["rupture_14j"] = int((qty_end[i+1:i+15] == 0).any()) if i + 14 < N else None
                r["demande_30j"] = float(ventes[i+1:i+31].sum()) if i + 30 < N else None
            for L in lots:
                lots_rows.append(dict(id=L["id"], stock_id=stock_id, numero_lot=L["num"], quantite=L["qty"],
                                      date_expiration=str(start + timedelta(days=L["exp"])),
                                      date_reception=str(start + timedelta(days=L["rec"]))))
            stocks.append(dict(id=stock_id, medecin_id=(a.doctor_id if m_id == 1 else -m_id), medicament_id=med["id"],
                               quantite_actuelle=int(qty_end[-1]), quantite_minimale=qmin, emplacement=emp,
                               updated_at=f"{end} 18:00:00"))

    # --- jeux ML
    ds_st = pd.DataFrame(snap_stock)
    rup = ds_st[(ds_st.quantite_actuelle > 0) & ds_st.rupture_14j.notna()].drop(columns=["demande_30j"])
    rup["rupture_14j"] = rup["rupture_14j"].astype(int)
    rea = ds_st[ds_st.demande_30j.notna()].drop(columns=["rupture_14j"])

    ds_lot = pd.DataFrame(snap_lot)
    ds_lot["resolu"] = ds_lot["_lot"].map(lambda L: L["closed"])
    ds_lot["expire_avec_reste"] = ds_lot["_lot"].map(lambda L: int(L["perdu"] > 0))
    exp = ds_lot[ds_lot.resolu].drop(columns=["_lot", "resolu"])

    out = Path(a.out); out.mkdir(exist_ok=True)
    st_all = pd.DataFrame(stocks)
    demo = st_all[st_all.medecin_id == a.doctor_id]            # cabinet 1 -> base Mediko
    demo_ids = set(demo["id"])
    pd.DataFrame(meds).to_csv(out / "medicaments.csv", index=False)
    demo.to_csv(out / "stock_medicaments.csv", index=False)
    lo = pd.DataFrame(lots_rows)
    lo[lo.stock_id.isin(demo_ids)].to_csv(out / "lots_medicaments.csv", index=False)
    mv = pd.DataFrame(mvts)
    mv = mv[mv.stock_id.isin(demo_ids)].sort_values(["date_mouvement", "stock_id"]).reset_index(drop=True)
    mv["id"] = range(1, len(mv) + 1)
    mv.to_csv(out / "mouvements_stock.csv", index=False)
    rup.to_csv(out / "dataset_rupture.csv", index=False)
    exp.to_csv(out / "dataset_expiration.csv", index=False)
    rea.to_csv(out / "dataset_reapprovisionnement.csv", index=False)

    print(f"medicaments={len(meds)} | cabinet 1 (CSV tables) : stocks={len(demo)} lots={int(lo.stock_id.isin(demo_ids).sum())} mouvements={len(mv)}")
    print(f"dataset_rupture={len(rup)} (taux rupture 14j = {rup.rupture_14j.mean():.1%})")
    print(f"dataset_expiration={len(exp)} (taux perte à l'expiration = {exp.expire_avec_reste.mean():.1%})")
    print(f"dataset_reappro={len(rea)}")


if __name__ == "__main__":
    main()
