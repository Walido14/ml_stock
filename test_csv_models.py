import joblib
import pandas as pd
from pathlib import Path
import numpy as np
import time

from features import STOCK_FEATURES, LOT_FEATURES

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

def tester_sur_csv():
    print("🚀 DÉMARRAGE DU TEST DES MODÈLES SUR FICHIERS CSV\n")
    start_time = time.time()

    print("1️⃣ Chargement des modèles depuis le dossier 'models/'...")
    b_rup = joblib.load(MODELS_DIR / "rupture.joblib")
    b_exp = joblib.load(MODELS_DIR / "expiration.joblib")
    b_rea = joblib.load(MODELS_DIR / "reapprovisionnement.joblib")
    print("   ✅ 3 modèles chargés avec succès.\n")

    # --- TEST RUPTURE ---
    print("2️⃣ TEST MODÈLE RUPTURE")
    print("   -> Lecture de 'dataset_rupture.csv'...")
    df_rup = pd.read_csv(DATA_DIR / "dataset_rupture.csv")
    print(f"   -> Fichier chargé ({len(df_rup)} lignes). Filtrage des cas à risque...")
    
    df_rup_risque = df_rup[df_rup["rupture_14j"] == 1].head(5).copy()
    X_rup = df_rup_risque[b_rup["features"]]
    
    print("   -> Prédiction en cours...")
    probas_rup = b_rup["model"].predict_proba(X_rup)[:, 1]
    df_rup_risque["Probabilité Rupture (14j)"] = np.round(probas_rup, 3)
    
    print("   ✅ Résultats (Cas réels de rupture) :")
    print(df_rup_risque[["stock_id", "rupture_14j", "Probabilité Rupture (14j)"]].to_string(index=False))
    print()

    # --- TEST EXPIRATION ---
    print("3️⃣ TEST MODÈLE EXPIRATION")
    print("   -> Lecture de 'dataset_expiration.csv'...")
    df_exp = pd.read_csv(DATA_DIR / "dataset_expiration.csv", dtype={"numero_lot": str})
    print(f"   -> Fichier chargé ({len(df_exp)} lignes). Filtrage des cas d'expiration...")
    
    df_exp_risque = df_exp[df_exp["expire_avec_reste"] == 1].head(5).copy()
    X_exp = df_exp_risque[b_exp["features"]]
    
    print("   -> Prédiction en cours...")
    probas_exp = b_exp["model"].predict_proba(X_exp)[:, 1]
    df_exp_risque["Probabilité Expiration"] = np.round(probas_exp, 3)
    
    print("   ✅ Résultats (Cas réels d'expiration) :")
    print(df_exp_risque[["stock_id", "numero_lot", "expire_avec_reste", "Probabilité Expiration"]].to_string(index=False))
    print()

    # --- TEST REAPPRO ---
    print("4️⃣ TEST MODÈLE RÉAPPROVISIONNEMENT")
    print("   -> Lecture de 'dataset_reapprovisionnement.csv'...")
    df_rea = pd.read_csv(DATA_DIR / "dataset_reapprovisionnement.csv")
    print(f"   -> Fichier chargé ({len(df_rea)} lignes). Sélection aléatoire de 5 lignes...")
    
    df_rea_sample = df_rea.sample(5).copy()
    X_rea = df_rea_sample[b_rea["features"]]
    
    print("   -> Prédiction en cours...")
    predictions_rea = np.clip(b_rea["model"].predict(X_rea), 0, None)
    df_rea_sample["Demande Prévue (30j)"] = np.round(predictions_rea, 1)
    
    print("   ✅ Résultats (Demande prévue sur 30 jours) :")
    print(df_rea_sample[["stock_id", "Demande Prévue (30j)"]].to_string(index=False))
    print()

    print(f"🎉 FIN DES TESTS (Durée: {time.time() - start_time:.2f} secondes)")

if __name__ == "__main__":
    tester_sur_csv()