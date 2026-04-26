"""
Day 2 - MLOps pour Sougui.tn (Purchasing)
Multi-métriques + Comparaison de modèles + Artefacts visuels

Améliorations par rapport à Day 1:
- Multi-métriques pour chaque objectif
- Comparaison de plusieurs modèles (Classification: RF vs XGBoost, Régression: Linear vs RF vs XGBoost)
- Matrices de confusion, courbes ROC, graphiques résidus
- Logging structuré avec paramètres et artefacts
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import hashlib
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

# ============================================
# 1. CONNEXION MLflow
# ============================================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ============================================
# 2. CHARGEMENT DES DONNÉES (synthétiques ou MySQL)
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
        conn.execute(text("SELECT 1"))
    use_mysql = True
    print("✅ Connecté à MySQL - chargement des données réelles")
except Exception as e:
    print(f"⚠️ MySQL indisponible ({e})")
    print("🔄 Génération de données synthétiques réalistes...")
    use_mysql = False

if use_mysql:
    query = """
    SELECT 
        fa.N_Facture, fa.ID_Fournisseur, f.Nom_Fournisseur, f.Categorie_Produit,
        fa.Montant_HT, fa.Montant_TVA, fa.Montant_TTC,
        d.Annee, d.Mois, d.Trimestre, d.Est_weekend,
        mp.Methode_Paiement, mp.Type_Paiement
    FROM f_achats fa
    LEFT JOIN fournisseur f ON fa.ID_Fournisseur = f.ID_Fournisseur
    LEFT JOIN d_date d ON fa.ID_Date = d.Date_PK
    LEFT JOIN d_methode_paiement mp ON fa.ID_Methode = mp.ID_Methode
    WHERE fa.Montant_HT > 0 AND fa.Montant_HT IS NOT NULL
    """
    df = pd.read_sql(query, engine)
    print(f"✅ Données réelles: {len(df)} lignes")
else:
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
    df["Montant_TTC"] = df["Montant_HT"] + df["Montant_TVA"]
    df.loc[df["Type_Paiement"] == "EXONERE", "Montant_TVA"] = 0
    print(f"✅ Données synthétiques: {len(df)} lignes")

# ============================================
# 3. FEATURE ENGINEERING
# ============================================
df["Taux_TVA"] = (df["Montant_TVA"] / df["Montant_HT"].replace(0, np.nan)).fillna(0).round(3)
df["Categorie_Enc"] = df["Categorie_Produit"].astype("category").cat.codes
df["Fournisseur_Enc"] = df["Nom_Fournisseur"].astype("category").cat.codes
df["Methode_Enc"] = df["Methode_Paiement"].astype("category").cat.codes
df["Type_Paiement_Enc"] = df["Type_Paiement"].astype("category").cat.codes
df["Facture_Standard"] = (df["Taux_TVA"] > 0).astype(int)

data_source = "MySQL" if use_mysql else "synthetic"
data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

mlflow.set_experiment("SOUGUI_Purchasing_Day2")

# ============================================
# 4. CLASSIFICATION (multi-métriques + ROC + matrice confusion)
# ============================================
print("\n" + "="*50)
print("📊 CLASSIFICATION - Prédiction type de facture")
print("="*50)

features_clf = ["Montant_HT", "Taux_TVA", "Mois", "Annee", "Est_weekend",
                "Categorie_Enc", "Fournisseur_Enc", "Methode_Enc"]
X_clf = df[features_clf].fillna(0)
y_clf = df["Facture_Standard"]
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

models_clf = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)
}

for name, model in models_clf.items():
    with mlflow.start_run(run_name=f"Classification_{name}"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Métriques
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        
        # Logging
        mlflow.log_params({
            "model_type": name,
            "dataset_hash": data_hash,
            "data_source": data_source,
            "n_estimators": 100,
            "test_size": 0.2
        })
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        
        # Matrice de confusion
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        plt.savefig("cm.png")
        mlflow.log_artifact("cm.png")
        plt.close()
        
        # Courbe ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(fpr, tpr, label=f"ROC (AUC={metrics['roc_auc']:.3f})")
        ax.plot([0,1], [0,1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {name}")
        ax.legend()
        plt.tight_layout()
        plt.savefig("roc.png")
        mlflow.log_artifact("roc.png")
        plt.close()
        
        # Modèle
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_train.iloc[:2])
        
        print(f"  ✅ {name} - Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['roc_auc']:.4f}")

# ============================================
# 5. RÉGRESSION (multi-métriques + comparaison)
# ============================================
print("\n" + "="*50)
print("📊 RÉGRESSION - Prédiction Montant HT")
print("="*50)

features_reg = ["Mois", "Annee", "Est_weekend", "Categorie_Enc", 
                "Fournisseur_Enc", "Methode_Enc", "Taux_TVA"]
X_reg = df[features_reg].fillna(0)
y_reg = df["Montant_HT"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

models_reg = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

for name, model in models_reg.items():
    with mlflow.start_run(run_name=f"Regression_{name}"):
        model.fit(X_train_r, y_train_r)
        y_pred = model.predict(X_test_r)
        
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test_r, y_pred)),
            "mae": mean_absolute_error(y_test_r, y_pred),
            "r2": r2_score(y_test_r, y_pred)
        }
        
        mlflow.log_params({
            "model_type": name,
            "dataset_hash": data_hash,
            "data_source": data_source,
            "test_size": 0.2
        })
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        
        # Graphique résidus
        residuals = y_test_r - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(12,4))
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_title(f"Residuals - {name}")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residuals")
        
        axes[1].hist(residuals, bins=30, edgecolor="black")
        axes[1].set_title(f"Residuals Distribution - {name}")
        axes[1].set_xlabel("Residuals")
        plt.tight_layout()
        plt.savefig("residuals.png")
        mlflow.log_artifact("residuals.png")
        plt.close()
        
        # Modèle
        signature = mlflow.models.infer_signature(X_train_r, model.predict(X_train_r))
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_train_r.iloc[:2])
        
        print(f"  ✅ {name} - RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}")

# ============================================
# 6. CLUSTERING (multi-métriques)
# ============================================
print("\n" + "="*50)
print("📊 CLUSTERING - Segmentation fournisseurs")
print("="*50)

df_fourn = df.groupby("ID_Fournisseur").agg({
    "Montant_HT": ["sum", "mean", "count"],
    "Taux_TVA": "mean"
}).round(2)
df_fourn.columns = ["Montant_Total", "Montant_Moyen", "Nb_Factures", "TVA_Moy"]
df_fourn = df_fourn.fillna(0)

scaler = StandardScaler()
X_clust = scaler.fit_transform(df_fourn[["Montant_Total", "Nb_Factures", "TVA_Moy"]])

for k in [2, 3, 4]:
    with mlflow.start_run(run_name=f"Clustering_KMeans_k{k}"):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_clust)
        
        metrics = {
            "silhouette": silhouette_score(X_clust, labels) if len(set(labels)) > 1 else 0,
            "davies_bouldin": davies_bouldin_score(X_clust, labels) if len(set(labels)) > 1 else 0,
            "inertia": model.inertia_
        }
        
        mlflow.log_params({
            "n_clusters": k,
            "dataset_hash": data_hash,
            "data_source": data_source,
            "scaled": True
        })
        for k_metric, v in metrics.items():
            mlflow.log_metric(k_metric, v)
        
        # Visualisation
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_clust)
        fig, ax = plt.subplots(figsize=(8,6))
        scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="viridis", s=50)
        ax.set_title(f"Clusters (PCA) - k={k}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig(f"clusters_k{k}.png")
        mlflow.log_artifact(f"clusters_k{k}.png")
        plt.close()
        
        with open(f"clust_labels_k{k}.pkl", "wb") as f:
            pickle.dump(labels, f)
        mlflow.log_artifact(f"clust_labels_k{k}.pkl")
        
        print(f"  ✅ k={k} - Silhouette={metrics['silhouette']:.4f}, DB={metrics['davies_bouldin']:.4f}")

# ============================================
# 7. FORECAST (avec MAPE)
# ============================================
print("\n" + "="*50)
print("📊 FORECAST - Prévision achats mensuels")
print("="*50)

df_ts = df.groupby(["Annee", "Mois"])["Montant_HT"].sum().reset_index()
df_ts["Date"] = pd.to_datetime(df_ts["Annee"].astype(str) + "-" + df_ts["Mois"].astype(str) + "-01")
df_ts = df_ts.sort_values("Date").set_index("Date")
series = df_ts["Montant_HT"]

if len(series) >= 6:
    train = series[:-3] if len(series) > 3 else series
    test = series[-3:] if len(series) > 3 else series
    
    with mlflow.start_run(run_name="Forecast_ARIMA"):
        model = ARIMA(train, order=(1, 1, 1))
        fit = model.fit()
        forecast = fit.forecast(steps=len(test))
        
        rmse = np.sqrt(mean_squared_error(test, forecast)) if len(test) > 0 else np.nan
        mae = mean_absolute_error(test, forecast) if len(test) > 0 else np.nan
        mape = np.mean(np.abs((test.values - forecast) / (test.values + 1e-6))) * 100 if len(test) > 0 else np.nan
        
        mlflow.log_params({
            "model_type": "ARIMA",
            "order": "(1,1,1)",
            "dataset_hash": data_hash,
            "data_source": data_source,
            "train_months": len(train),
            "test_months": len(test)
        })
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "mape": mape})
        
        # Graphique
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(train.index, train, label="Train", linewidth=2)
        ax.plot(test.index, test, label="Actual", marker="o", linewidth=2)
        ax.plot(test.index, forecast, label="Forecast", marker="x", linestyle="--", linewidth=2)
        ax.legend()
        ax.set_title(f"ARIMA Forecast - RMSE={rmse:.2f}, MAPE={mape:.2f}%")
        ax.set_xlabel("Date")
        ax.set_ylabel("Montant HT")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("forecast.png")
        mlflow.log_artifact("forecast.png")
        plt.close()
        
        with open("forecast.pkl", "wb") as f:
            pickle.dump(forecast, f)
        mlflow.log_artifact("forecast.pkl")
        
        print(f"  ✅ ARIMA(1,1,1) - RMSE={rmse:.2f}, MAPE={mape:.2f}%")
else:
    print(f"⚠️ Forecast ignoré: seulement {len(series)} mois (minimum 6 requis)")

# ============================================
# 8. RÉSUMÉ FINAL
# ============================================
print("\n" + "="*60)
print("📈 RÉSUMÉ DAY 2 - SOUGUI.TN PURCHASING")
print("="*60)
print(f"🔹 Source données: {data_source}")
print(f"🔹 Runs créés: Classification(2), Régression(3), Clustering(3), Forecast(1)")
print(f"🔹 Total: 9 runs enregistrés dans MLflow")
print("\n📊 Comparaison des modèles disponible dans MLflow UI")
print("🌐 Interface: http://127.0.0.1:5000")
print("\n✅ Day 2 terminé avec succès!")