# Mediko ML – stock médicaments

Installation (une fois) :

    pip install -r requirements.txt
    python train_models.py          # réentraîne avec VOTRE version de scikit-learn (les .joblib sont sensibles à la version)

Tester / valider :

    python test_pipeline.py         # 21 tests : qualité, cohérence base<->entraînement, scénarios métier, invariants

Sur la vraie base (copie !) :

    python predict.py --db "<chemin de la copie .db>" --out result.json

Données de démonstration (fictives, jamais dans une vraie base) :

    python generate_dataset.py --doctor-id 1 --secretary-id 2
    python import_csv_to_db.py --db test.db --demo

Intégration : copier ce dossier dans src-tauri/ml/, ajouter ml.rs (commande get_stock_predictions)
et <StockPredictions /> dans la page Stock.
