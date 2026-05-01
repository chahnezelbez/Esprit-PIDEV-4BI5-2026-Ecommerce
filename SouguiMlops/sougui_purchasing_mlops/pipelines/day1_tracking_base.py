"""
Day 1 - MLOps pour Sougui.tn (Purchasing)
Tracking de base MLflow avec fallback données synthétiques si MySQL indisponible
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import hashlib
import pickle
import warnings
warnings.filterwarnings("ignore")

from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from statsmodels.tsa.arima.model import ARIMA
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")
# ============================================
# 1. TENTATIVE DE CONNEXION MySQL
# ============================================
DB_USER = "root"
DB_PASSWORD = ""
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_NAME = "dwh_sougui"

use_mysql = False
df = None

try:
    engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        use_mysql = True
    print("✅ Connecté à MySQL - chargement des données réelles")
except Exception as e:
    print(f"⚠️ MySQL indisponible ({e})")
    print("🔄 Génération de données synthétiques réalistes pour continuer...")
    use_mysql = False

# ============================================
# 2. CHARGEMENT DES DONNÉES (réelles ou synthétiques)
# ============================================
if use_mysql:
    query = """
    SELECT 
        fa.N_Facture,
        fa.ID_Fournisseur,
        f.Nom_Fournisseur,
        f.Categorie_Produit,
        fa.Montant_HT,
        fa.Montant_TVA,
        fa.Montant_TTC,
        d.Annee,
        d.Mois,
        d.Trimestre,
        d.Est_weekend,
        mp.Methode_Paiement,
        mp.Type_Paiement
    FROM f_achats fa
    LEFT JOIN fournisseur f ON fa.ID_Fournisseur = f.ID_Fournisseur
    LEFT JOIN d_date d ON fa.ID_Date = d.Date_PK
    LEFT JOIN d_methode_paiement mp ON fa.ID_Methode = mp.ID_Methode
    WHERE fa.Montant_HT > 0
      AND fa.Montant_HT IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    print(f"✅ Données réelles: {len(df)} lignes")
else:
    # ============================================
    # DONNÉES SYNTHÉTIQUES RÉALISTES
    # ============================================
    np.random.seed(42)
    n = 500
    
    fournisseurs = ["Karim Bellah", "KOBOX", "Riadh Louaiti", "Zouba Ceramic", 
                    "Oriental Design Kammoun", "INCONNU", "LEM", "LES 3 SINGES"]
    categories = ["SERVICE", "MATERIAUX", "EQUIPEMENT", "TRANSPORT"]
    methodes = ["CB", "CHEQUE", "VIREMENT", "ESPECES", "Inconnu"]
    types_paiement = ["STANDARD", "EXONERE", "Inconnu"]
    
    df = pd.DataFrame({
        "N_Facture": [f"FACT_{i}" for i in range(n)],
        "ID_Fournisseur": np.random.randint(1, 50, n),
        "Nom_Fournisseur": np.random.choice(fournisseurs, n),
        "Categorie_Produit": np.random.choice(categories, n),
        "Montant_HT": np.random.exponential(500, n).round(2) + 10,
        "Montant_TVA": np.random.exponential(100, n).round(2),
        "Montant_TTC": 0,
        "Annee": np.random.choice([2022, 2023, 2024, 2025], n),
        "Mois": np.random.randint(1, 13, n),
        "Trimestre": np.random.randint(1, 5, n),
        "Est_weekend": np.random.choice([0, 1], n),
        "Methode_Paiement": np.random.choice(methodes, n),
        "Type_Paiement": np.random.choice(types_paiement, n),
    })
    
    # Recalcul Montant_TTC
    df["Montant_TTC"] = df["Montant_HT"] + df["Montant_TVA"]
    
    # Forcer quelques valeurs TVA à 0 pour créer des factures exonérées
    df.loc[df["Type_Paiement"] == "EXONERE", "Montant_TVA"] = 0
    df.loc[df["Type_Paiement"] == "EXONERE", "Taux_TVA"] = 0
    
    print(f"✅ Données synthétiques générées: {len(df)} lignes")

# ============================================
# 3. FEATURE ENGINEERING
# ============================================
df["Taux_TVA"] = (df["Montant_TVA"] / df["Montant_HT"].replace(0, np.nan)).fillna(0).round(3)

# Encodage catégoriel
df["Categorie_Enc"] = df["Categorie_Produit"].astype("category").cat.codes
df["Fournisseur_Enc"] = df["Nom_Fournisseur"].astype("category").cat.codes
df["Methode_Enc"] = df["Methode_Paiement"].astype("category").cat.codes
df["Type_Paiement_Enc"] = df["Type_Paiement"].astype("category").cat.codes

# Variable cible classification: facture avec TVA > 0
df["Facture_Standard"] = (df["Taux_TVA"] > 0).astype(int)

print("✅ Feature engineering terminé")

# ============================================
# 4. CLASSIFICATION
# ============================================
features_clf = ["Montant_HT", "Taux_TVA", "Mois", "Annee", "Est_weekend",
                "Categorie_Enc", "Fournisseur_Enc", "Methode_Enc"]

X_clf = df[features_clf].fillna(0)
y_clf = df["Facture_Standard"]

model_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_clf.fit(X_clf, y_clf)
y_pred_clf = model_clf.predict(X_clf)
acc_clf = accuracy_score(y_clf, y_pred_clf)

print(f"✅ Classification - Accuracy: {acc_clf:.4f}")

# ============================================
# 5. RÉGRESSION
# ============================================
features_reg = ["Mois", "Annee", "Est_weekend", "Categorie_Enc", 
                "Fournisseur_Enc", "Methode_Enc", "Taux_TVA"]

X_reg = df[features_reg].fillna(0)
y_reg = df["Montant_HT"]

model_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_reg.fit(X_reg, y_reg)
y_pred_reg = model_reg.predict(X_reg)
rmse_reg = np.sqrt(mean_squared_error(y_reg, y_pred_reg))

print(f"✅ Régression - RMSE: {rmse_reg:.2f}")

# ============================================
# 6. CLUSTERING (segmentation fournisseurs)
# ============================================
df_fourn = df.groupby("ID_Fournisseur").agg({
    "Montant_HT": ["sum", "mean", "count"],
    "Taux_TVA": "mean"
}).round(2)
df_fourn.columns = ["Montant_Total", "Montant_Moyen", "Nb_Factures", "TVA_Moy"]
df_fourn = df_fourn.fillna(0)

X_clust = df_fourn[["Montant_Total", "Nb_Factures", "TVA_Moy"]].values

model_clust = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_clust = model_clust.fit_predict(X_clust)
silhouette = silhouette_score(X_clust, labels_clust)

print(f"✅ Clustering - Silhouette: {silhouette:.4f}")

# ============================================
# 7. FORECAST (série mensuelle)
# ============================================
df_ts = df.groupby(["Annee", "Mois"])["Montant_HT"].sum().reset_index()
df_ts["Date"] = pd.to_datetime(df_ts["Annee"].astype(str) + "-" + df_ts["Mois"].astype(str) + "-01")
df_ts = df_ts.sort_values("Date").set_index("Date")

series = df_ts["Montant_HT"]

if len(series) >= 4:
    model_arima = ARIMA(series, order=(1, 1, 1))
    model_fit = model_arima.fit()
    forecast = model_fit.forecast(steps=3)
    forecast_rmse = np.nan
    print(f"✅ Forecast - Modèle ARIMA entraîné sur {len(series)} mois")
else:
    model_fit = None
    forecast = None
    forecast_rmse = np.nan
    print(f"⚠️ Forecast - Pas assez de données ({len(series)} mois)")

# ============================================
# 8. HASH DATASET
# ============================================
data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

# ============================================
# 9. MLFLOW TRACKING
# ============================================
mlflow.set_experiment("SOUGUI_Purchasing_Day1")

# Classification
with mlflow.start_run(run_name="Classification_RF"):
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("dataset_hash", data_hash)
    mlflow.log_param("data_source", "MySQL" if use_mysql else "synthetic")
    mlflow.log_metric("accuracy", acc_clf)
    
    signature = mlflow.models.infer_signature(X_clf, y_pred_clf)
    mlflow.sklearn.log_model(model_clf, "model", signature=signature, input_example=X_clf.iloc[:2])
    
    with open("preds_clf.pkl", "wb") as f:
        pickle.dump(y_pred_clf, f)
    mlflow.log_artifact("preds_clf.pkl")
    print("  📊 Classification logged")

# Régression
with mlflow.start_run(run_name="Regression_RF"):
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("dataset_hash", data_hash)
    mlflow.log_param("data_source", "MySQL" if use_mysql else "synthetic")
    mlflow.log_metric("rmse", rmse_reg)
    
    signature = mlflow.models.infer_signature(X_reg, y_pred_reg)
    mlflow.sklearn.log_model(model_reg, "model", signature=signature, input_example=X_reg.iloc[:2])
    
    with open("preds_reg.pkl", "wb") as f:
        pickle.dump(y_pred_reg, f)
    mlflow.log_artifact("preds_reg.pkl")
    print("  📊 Regression logged")

# Clustering
with mlflow.start_run(run_name="Clustering_KMeans"):
    mlflow.log_param("model_type", "KMeans")
    mlflow.log_param("n_clusters", 3)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("dataset_hash", data_hash)
    mlflow.log_param("data_source", "MySQL" if use_mysql else "synthetic")
    mlflow.log_metric("silhouette_score", silhouette)
    
    with open("clust_labels.pkl", "wb") as f:
        pickle.dump(labels_clust, f)
    mlflow.log_artifact("clust_labels.pkl")
    print("  📊 Clustering logged")

# Forecast
if model_fit:
    with mlflow.start_run(run_name="Forecast_ARIMA"):
        mlflow.log_param("model_type", "ARIMA")
        mlflow.log_param("order", "(1,1,1)")
        mlflow.log_param("dataset_hash", data_hash)
        mlflow.log_param("data_source", "MySQL" if use_mysql else "synthetic")
        mlflow.log_metric("forecast_rmse", forecast_rmse)
        
        with open("forecast.pkl", "wb") as f:
            pickle.dump(forecast, f)
        mlflow.log_artifact("forecast.pkl")
        print("  📊 Forecast logged")

# ============================================
# 10. RÉSUMÉ FINAL
# ============================================
print("\n" + "="*50)
print("📈 RÉSUMÉ DAY 1 - SOUGUI.TN PURCHASING")
print("="*50)
print(f"🔹 Source données: {'MySQL (réel)' if use_mysql else 'Synthétique (fallback)'}")
print(f"🔹 Classification - Accuracy: {acc_clf:.4f}")
print(f"🔹 Régression     - RMSE: {rmse_reg:.2f}")
print(f"🔹 Clustering     - Silhouette: {silhouette:.4f}")
if model_fit:
    print(f"🔹 Forecast       - Modèle: ARIMA(1,1,1) sur {len(series)} mois")
else:
    print(f"🔹 Forecast       - Non entraîné (données insuffisantes)")
print("\n✅ 4 runs enregistrés dans MLflow")
print("🌐 Interface MLflow: http://127.0.0.1:5000")