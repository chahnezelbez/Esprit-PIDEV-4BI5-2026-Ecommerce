"""
Day 3 - MLOps pour Sougui.tn (Purchasing)
Model Registry + Staging → Production

Objectifs:
- Enregistrer les meilleurs modèles dans MLflow Model Registry
- Transition Staging → Production
- Charger et tester un modèle depuis le registry
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

# ============================================
# 1. CONNEXION MLflow
# ============================================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ============================================
# 2. CHARGEMENT DES DONNÉES (synthétiques)
# ============================================
print("\n🔄 Génération des données...")
np.random.seed(42)
n = 500

df = pd.DataFrame({
    "ID_Fournisseur": np.random.randint(1, 50, n),
    "Montant_HT": np.random.exponential(500, n).round(2) + 10,
    "Montant_TVA": np.random.exponential(100, n).round(2),
    "Annee": np.random.choice([2022, 2023, 2024, 2025], n),
    "Mois": np.random.randint(1, 13, n),
    "Est_weekend": np.random.choice([0, 1], n),
    "Categorie_Enc": np.random.randint(0, 4, n),
    "Fournisseur_Enc": np.random.randint(0, 20, n),
    "Methode_Enc": np.random.randint(0, 5, n),
})

df["Taux_TVA"] = (df["Montant_TVA"] / df["Montant_HT"].replace(0, np.nan)).fillna(0)
df["Facture_Standard"] = (df["Taux_TVA"] > 0).astype(int)
print(f"✅ Données: {len(df)} lignes")

# ============================================
# 3. CLASSIFICATION - Meilleur modèle (RandomForest)
# ============================================
print("\n" + "="*50)
print("📊 CLASSIFICATION - Model Registry")
print("="*50)

features_clf = ["Montant_HT", "Taux_TVA", "Mois", "Annee", "Est_weekend",
                "Categorie_Enc", "Fournisseur_Enc", "Methode_Enc"]
X_clf = df[features_clf].fillna(0)
y_clf = df["Facture_Standard"]
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Entraînement du meilleur modèle (RandomForest)
model_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_clf.fit(X_train, y_train)
y_pred = model_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ RandomForest - Accuracy: {acc:.4f}")

# Enregistrement dans Model Registry
with mlflow.start_run(run_name="Classification_RF_Best") as run:
    mlflow.log_params({
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "purpose": "production_candidate"
    })
    mlflow.log_metric("accuracy", acc)
    
    signature = mlflow.models.infer_signature(X_train, model_clf.predict(X_train))
    mlflow.sklearn.log_model(
        model_clf, 
        "model", 
        signature=signature,
        registered_model_name="SOUGUI_Classifier_RF"
    )
    print(f"   ✅ Modèle enregistré: SOUGUI_Classifier_RF (version 1)")

# ============================================
# 4. RÉGRESSION - Meilleur modèle (RandomForest)
# ============================================
print("\n" + "="*50)
print("📊 RÉGRESSION - Model Registry")
print("="*50)

features_reg = ["Mois", "Annee", "Est_weekend", "Categorie_Enc", 
                "Fournisseur_Enc", "Methode_Enc", "Taux_TVA"]
X_reg = df[features_reg].fillna(0)
y_reg = df["Montant_HT"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Correction: random_state=42 (pas true42)
model_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_reg.fit(X_train_r, y_train_r)
y_pred_r = model_reg.predict(X_test_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))

print(f"✅ RandomForest - RMSE: {rmse:.2f}")

with mlflow.start_run(run_name="Regression_RF_Best") as run:
    mlflow.log_params({
        "model_type": "RandomForestRegressor",
        "n_estimators": 100,
        "purpose": "production_candidate"
    })
    mlflow.log_metric("rmse", rmse)
    
    signature = mlflow.models.infer_signature(X_train_r, model_reg.predict(X_train_r))
    mlflow.sklearn.log_model(
        model_reg, 
        "model", 
        signature=signature,
        registered_model_name="SOUGUI_Regressor_RF"
    )
    print(f"   ✅ Modèle enregistré: SOUGUI_Regressor_RF (version 1)")

# ============================================
# 5. CLUSTERING - Meilleur modèle (k=3)
# ============================================
print("\n" + "="*50)
print("📊 CLUSTERING - Model Registry")
print("="*50)

df_fourn = df.groupby("ID_Fournisseur").agg({
    "Montant_HT": ["sum", "mean", "count"],
    "Taux_TVA": "mean"
})
df_fourn.columns = ["Montant_Total", "Montant_Moyen", "Nb_Factures", "TVA_Moy"]
df_fourn = df_fourn.fillna(0)

scaler = StandardScaler()
X_clust = scaler.fit_transform(df_fourn[["Montant_Total", "Nb_Factures", "TVA_Moy"]])

model_clust = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model_clust.fit_predict(X_clust)
silhouette = silhouette_score(X_clust, labels)

print(f"✅ KMeans(k=3) - Silhouette: {silhouette:.4f}")

# Sauvegarde du modèle clustering (pickle, car pas de format MLflow standard pour KMeans)
with open("clust_model.pkl", "wb") as f:
    pickle.dump({"model": model_clust, "scaler": scaler}, f)

# Logging dans MLflow (sans registry car KMeans non supporté nativement)
with mlflow.start_run(run_name="Clustering_KMeans_Best"):
    mlflow.log_params({
        "model_type": "KMeans",
        "n_clusters": 3,
        "scaled": True
    })
    mlflow.log_metric("silhouette_score", silhouette)
    mlflow.log_artifact("clust_model.pkl")
    print(f"   ✅ Modèle sauvegardé (artefact)")

# ============================================
# 6. TRANSITION STAGING → PRODUCTION
# ============================================
print("\n" + "="*50)
print("📊 TRANSITION DES MODÈLES")
print("="*50)

from mlflow.tracking import MlflowClient

client = MlflowClient()

# Pour le classifieur
try:
    latest_version = client.get_latest_versions("SOUGUI_Classifier_RF", stages=["None"])[0].version
    client.transition_model_version_stage(
        name="SOUGUI_Classifier_RF",
        version=latest_version,
        stage="Staging"
    )
    print(f"   ✅ SOUGUI_Classifier_RF version {latest_version} → Staging")
except Exception as e:
    print(f"   ⚠️ Transition classifieur: {e}")

# Pour le régresseur
try:
    latest_version = client.get_latest_versions("SOUGUI_Regressor_RF", stages=["None"])[0].version
    client.transition_model_version_stage(
        name="SOUGUI_Regressor_RF",
        version=latest_version,
        stage="Staging"
    )
    print(f"   ✅ SOUGUI_Regressor_RF version {latest_version} → Staging")
except Exception as e:
    print(f"   ⚠️ Transition régresseur: {e}")

# ============================================
# 7. CHARGEMENT D'UN MODÈLE DEPUIS LE REGISTRY
# ============================================
print("\n" + "="*50)
print("📊 TEST - Chargement depuis Registry")
print("="*50)

# Charger le classifieur depuis Staging
model_uri = "models:/SOUGUI_Classifier_RF/Staging"
try:
    loaded_model = mlflow.sklearn.load_model(model_uri)
    test_pred = loaded_model.predict(X_test[:5])
    print(f"   ✅ Classifieur chargé depuis Staging - Prédictions: {test_pred}")
except Exception as e:
    print(f"   ⚠️ Impossible de charger le classifieur: {e}")

# Charger le régresseur depuis Staging
model_uri = "models:/SOUGUI_Regressor_RF/Staging"
try:
    loaded_model = mlflow.sklearn.load_model(model_uri)
    test_pred = loaded_model.predict(X_test_r[:5])
    print(f"   ✅ Régresseur chargé depuis Staging - Prédictions: {test_pred[:3]}...")
except Exception as e:
    print(f"   ⚠️ Impossible de charger le régresseur: {e}")

# ============================================
# 8. RÉSUMÉ FINAL
# ============================================
print("\n" + "="*60)
print("📈 RÉSUMÉ DAY 3 - MODEL REGISTRY")
print("="*60)
print(f"🔹 Modèles enregistrés:")
print(f"   - SOUGUI_Classifier_RF (accuracy={acc:.4f})")
print(f"   - SOUGUI_Regressor_RF (rmse={rmse:.2f})")
print(f"   - Clustering KMeans (k=3, silhouette={silhouette:.4f})")
print(f"\n🔹 Statut des modèles:")
print(f"   - Classifieur: Staging")
print(f"   - Régresseur: Staging")
print("\n🔹 Prochaines étapes:")
print("   1. Tester les modèles en Staging")
print("   2. Promouvoir en Production via UI MLflow ou API")
print("   3. Automatiser avec Day 4 (CI/CD pipeline)")
print("\n🌐 Interface MLflow: http://127.0.0.1:5000")
print("   → Onglet 'Models' pour voir les modèles enregistrés")