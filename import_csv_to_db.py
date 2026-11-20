#!/usr/bin/env python3
"""
Importe les CSV de démonstration dans une base SQLite au schéma Mediko (0001_initial.sql).
!! À utiliser sur une COPIE de la base ou une base de test : ce sont des données fictives. !!

  python import_csv_to_db.py --db test.db --demo            # base neuve : migration + utilisateurs de démo
  python import_csv_to_db.py --db copie_mediko.db           # base existante (médecin/secrétaire déjà créés)

- applique la migration si les tables de stock n'existent pas ;
- rattache le stock au médecin (premier users.role='doctor') et les mouvements à médecin/secrétaire ;
- refuse d'importer si le stock n'est pas vide ;
- vérifie ensuite l'intégrité (stock = somme des lots = historique des mouvements).
"""
import argparse, sqlite3, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--migration", default=str(HERE.parent / "migrations" / "0001_initial.sql"))
    ap.add_argument("--demo", action="store_true", help="crée un médecin et une secrétaire de démonstration")
    ap.add_argument("--csv-doctor-id", type=int, default=1, help="utilisateur_id 'médecin' utilisé dans les CSV")
    ap.add_argument("--csv-secretary-id", type=int, default=2, help="utilisateur_id 'secrétaire' utilisé dans les CSV")
    a = ap.parse_args()
    d = Path(a.data)

    con = sqlite3.connect(a.db)
    con.execute("PRAGMA foreign_keys = ON")
    if not con.execute("SELECT 1 FROM sqlite_master WHERE name='lots_medicaments'").fetchone():
        con.executescript(Path(a.migration).read_text(encoding="utf-8"))
        con.execute("PRAGMA foreign_keys = ON")

    def user(role):
        r = con.execute("SELECT id FROM users WHERE role=? ORDER BY id LIMIT 1", (role,)).fetchone()
        return r[0] if r else None
    if a.demo:
        with con:
            if user("doctor") is None:
                con.execute("INSERT INTO users (username, display_name, role, password_hash) VALUES ('demo_doctor','Dr Démo','doctor','x')")
            if user("secretary") is None:
                con.execute("INSERT INTO users (username, display_name, role, password_hash) VALUES ('demo_secretary','Secrétaire Démo','secretary','x')")
    doctor, secretary = user("doctor"), user("secretary")
    if doctor is None:
        sys.exit("Aucun médecin dans users (utilisez --demo sur une base de test).")
    secretary = secretary or doctor
    if con.execute("SELECT COUNT(*) FROM stock_medicaments").fetchone()[0]:
        sys.exit("Le stock n'est pas vide : importez dans une base neuve ou une copie de test.")

    med = pd.read_csv(d / "medicaments.csv", dtype=str, keep_default_na=False)
    st = pd.read_csv(d / "stock_medicaments.csv")
    lo = pd.read_csv(d / "lots_medicaments.csv", dtype={"numero_lot": str})
    mv = pd.read_csv(d / "mouvements_stock.csv", keep_default_na=False)
    uid = {a.csv_doctor_id: doctor, a.csv_secretary_id: secretary}

    with con:
        con.executemany("INSERT INTO medicaments (id,nom,principe_actif,dosage,forme,laboratoire,categorie,description,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                        [(int(r.id), r.nom, r.principe_actif, r.dosage, r.forme, r.laboratoire, r.categorie, r.description, r.created_at) for r in med.itertuples()])
        con.executemany("INSERT INTO stock_medicaments (id,medecin_id,medicament_id,quantite_minimale,emplacement) VALUES (?,?,?,?,?)",
                        [(int(r.id), doctor, int(r.medicament_id), int(r.quantite_minimale), r.emplacement) for r in st.itertuples()])
        con.executemany("INSERT INTO lots_medicaments (id,stock_id,numero_lot,quantite,date_expiration,date_reception) VALUES (?,?,?,?,?,?)",
                        [(int(r.id), int(r.stock_id), str(r.numero_lot), int(r.quantite), r.date_expiration, r.date_reception) for r in lo.itertuples()])
        con.executemany("INSERT INTO mouvements_stock (id,stock_id,type_mouvement,quantite,motif,date_mouvement,utilisateur_id) VALUES (?,?,?,?,?,?,?)",
                        [(int(r.id), int(r.stock_id), r.type_mouvement, int(r.quantite), r.motif, r.date_mouvement, uid[int(r.utilisateur_id)]) for r in mv.itertuples()])

    bad_sum = con.execute("""SELECT COUNT(*) FROM stock_medicaments s WHERE s.quantite_actuelle <>
        (SELECT COALESCE(SUM(quantite),0) FROM lots_medicaments WHERE stock_id=s.id)""").fetchone()[0]
    expected = dict(zip(st["id"], st["quantite_actuelle"]))
    bad_csv = sum(1 for i, q in con.execute("SELECT id, quantite_actuelle FROM stock_medicaments") if expected.get(i) != q)
    bad_mvt = con.execute("""SELECT COUNT(*) FROM stock_medicaments s WHERE s.quantite_actuelle <>
        (SELECT COALESCE(SUM(CASE type_mouvement WHEN 'ENTREE' THEN quantite WHEN 'SORTIE' THEN -quantite ELSE quantite END),0)
         FROM mouvements_stock WHERE stock_id=s.id)""").fetchone()[0]
    fk = con.execute("PRAGMA foreign_key_check").fetchall()
    for t in ("medicaments", "stock_medicaments", "lots_medicaments", "mouvements_stock"):
        print(f"{t:20s} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} lignes")
    print(f"médecin id={doctor} secrétaire id={secretary}")
    print(f"stock != somme des lots : {bad_sum} | stock != CSV : {bad_csv} | stock != historique : {bad_mvt} | FK violées : {len(fk)}")
    con.close()
    sys.exit(0 if not (bad_sum or bad_csv or bad_mvt or fk) else 1)


if __name__ == "__main__":
    main()
