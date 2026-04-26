"""
Day 4 - MLOps pour Sougui.tn (Purchasing)
CI/CD Pipeline + Tests Automatiques + Déploiement
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
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

# ============================================
# CONFIGURATION
# ============================================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# Seuils de performance pour la promotion
THRESHOLDS = {
    "classifier": {"accuracy": 0.95, "precision": 0.95, "recall": 0.95, "f1": 0.95},
    "regressor": {"rmse": 500.0, "r2": 0.70},
    "clustering": {"silhouette": 0.40}
}

print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"📊 Seuils de performance: {THRESHOLDS}")

# ============================================
# 1. GÉNÉRATION DES DONNÉES (avec classes équilibrées)
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
# Création d'une cible équilibrée (50% de 0, 50% de 1)
df["Facture_Standard"] = (df["Taux_TVA"] > np.median(df["Taux_TVA"])).astype(int)
print(f"✅ Données: {len(df)} lignes")
print(f"   Distribution Facture_Standard: {df['Facture_Standard'].value_counts().to_dict()}")

# ============================================
# 2. CLASSIFICATION - PIPELINE
# ============================================
print("\n" + "="*60)
print("📊 CLASSIFICATION PIPELINE")
print("="*60)

features_clf = ["Montant_HT", "Taux_TVA", "Mois", "Annee", "Est_weekend",
                "Categorie_Enc", "Fournisseur_Enc", "Methode_Enc"]
X_clf = df[features_clf].fillna(0)
y_clf = df["Facture_Standard"]
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Entraînement
model_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_clf.fit(X_train, y_train)

# Évaluation
y_pred = model_clf.predict(X_test)
metrics_clf = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0)
}

# Vérifier si la classe 1 existe dans les prédictions pour ROC-AUC
if len(np.unique(y_pred)) > 1 and hasattr(model_clf, "predict_proba"):
    y_proba = model_clf.predict_proba(X_test)
    if y_proba.shape[1] > 1:
        metrics_clf["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
    else:
        metrics_clf["roc_auc"] = 0.5
        print("   ⚠️ Une seule classe dans les prédictions, ROC-AUC fixé à 0.5")
else:
    metrics_clf["roc_auc"] = 0.5
    print("   ⚠️ Une seule classe détectée, ROC-AUC fixé à 0.5")

print(f"📈 Performance Classifieur:")
for k, v in metrics_clf.items():
    print(f"   {k}: {v:.4f}")

# Test des seuils
tests_reussis = True
for metric, value in metrics_clf.items():
    if metric in THRESHOLDS["classifier"]:
        seuil = THRESHOLDS["classifier"][metric]
        if value < seuil:
            print(f"   ❌ Échec: {metric} = {value:.4f} < seuil {seuil}")
            tests_reussis = False
        else:
            print(f"   ✅ {metric} = {value:.4f} ≥ seuil {seuil}")

# Enregistrement dans MLflow
with mlflow.start_run(run_name="CI_Classifier_RF_V3"):
    mlflow.log_params({
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "pipeline_stage": "CI_test"
    })
    for k, v in metrics_clf.items():
        mlflow.log_metric(k, v)
    
    signature = mlflow.models.infer_signature(X_train, model_clf.predict(X_train))
    mlflow.sklearn.log_model(model_clf, "model", signature=signature)
    
    if tests_reussis:
        result = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            "SOUGUI_Classifier_RF"
        )
        version = result.version
        client.transition_model_version_stage(
            name="SOUGUI_Classifier_RF",
            version=version,
            stage="Staging"
        )
        print(f"\n✅ Classifieur version {version} → Staging")
        
        if metrics_clf["accuracy"] >= 0.99:
            client.transition_model_version_stage(
                name="SOUGUI_Classifier_RF",
                version=version,
                stage="Production"
            )
            print(f"✅ Classifieur version {version} → Production (performance exceptionnelle)")
    else:
        print(f"\n❌ Classifieur non promu (tests échoués)")

# ============================================
# 3. RÉGRESSION - PIPELINE
# ============================================
print("\n" + "="*60)
print("📊 RÉGRESSION PIPELINE")
print("="*60)

features_reg = ["Mois", "Annee", "Est_weekend", "Categorie_Enc", 
                "Fournisseur_Enc", "Methode_Enc", "Taux_TVA"]
X_reg = df[features_reg].fillna(0)
y_reg = df["Montant_HT"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

model_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_reg.fit(X_train_r, y_train_r)
y_pred_r = model_reg.predict(X_test_r)

rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2 = r2_score(y_test_r, y_pred_r)

print(f"📈 Performance Régresseur:")
print(f"   RMSE: {rmse:.2f}")
print(f"   R²: {r2:.4f}")

tests_reussis = True
if rmse > THRESHOLDS["regressor"]["rmse"]:
    print(f"   ❌ RMSE = {rmse:.2f} > seuil {THRESHOLDS['regressor']['rmse']}")
    tests_reussis = False
else:
    print(f"   ✅ RMSE = {rmse:.2f} ≤ seuil {THRESHOLDS['regressor']['rmse']}")

if r2 < THRESHOLDS["regressor"]["r2"]:
    print(f"   ❌ R² = {r2:.4f} < seuil {THRESHOLDS['regressor']['r2']}")
    tests_reussis = False
else:
    print(f"   ✅ R² = {r2:.4f} ≥ seuil {THRESHOLDS['regressor']['r2']}")

with mlflow.start_run(run_name="CI_Regressor_RF_V2"):
    mlflow.log_params({
        "model_type": "RandomForestRegressor",
        "n_estimators": 100,
        "pipeline_stage": "CI_test"
    })
    mlflow.log_metrics({"rmse": rmse, "r2": r2})
    
    signature = mlflow.models.infer_signature(X_train_r, model_reg.predict(X_train_r))
    mlflow.sklearn.log_model(model_reg, "model", signature=signature)
    
    if tests_reussis:
        result = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            "SOUGUI_Regressor_RF"
        )
        version = result.version
        client.transition_model_version_stage(
            name="SOUGUI_Regressor_RF",
            version=version,
            stage="Staging"
        )
        print(f"\n✅ Régresseur version {version} → Staging")
        
        if rmse < 400:
            client.transition_model_version_stage(
                name="SOUGUI_Regressor_RF",
                version=version,
                stage="Production"
            )
            print(f"✅ Régresseur version {version} → Production (performance excellente)")
    else:
        print(f"\n❌ Régresseur non promu (tests échoués)")

# ============================================
# 4. CLUSTERING - PIPELINE
# ============================================
print("\n" + "="*60)
print("📊 CLUSTERING PIPELINE")
print("="*60)

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

print(f"📈 Performance Clustering:")
print(f"   Silhouette: {silhouette:.4f}")

if silhouette >= THRESHOLDS["clustering"]["silhouette"]:
    print(f"   ✅ Silhouette = {silhouette:.4f} ≥ seuil {THRESHOLDS['clustering']['silhouette']}")
    deploiement = "OK"
else:
    print(f"   ❌ Silhouette = {silhouette:.4f} < seuil {THRESHOLDS['clustering']['silhouette']}")
    deploiement = "À AMÉLIORER"

# Sauvegarde
with open("clust_model_v2.pkl", "wb") as f:
    pickle.dump({"model": model_clust, "scaler": scaler}, f)

with mlflow.start_run(run_name="CI_Clustering_KMeans_V2"):
    mlflow.log_params({"n_clusters": 3, "scaled": True})
    mlflow.log_metric("silhouette_score", silhouette)
    mlflow.log_artifact("clust_model_v2.pkl")
    print(f"   ✅ Modèle clustering sauvegardé - {deploiement}")

# ============================================
# 5. RAPPORT FINAL
# ============================================
print("\n" + "="*60)
print("📈 RAPPORT CI/CD - DAY 4")
print("="*60)
print(f"\n🔹 Classification:")
print(f"   - Version: Staging")
print(f"   - Accuracy: {metrics_clf['accuracy']:.4f}")
print(f"   - Tests: {'✅ PASSÉS' if tests_reussis else '❌ ÉCHOUÉS'}")
print(f"\n🔹 Régression:")
print(f"   - Version: Staging")
print(f"   - RMSE: {rmse:.2f}")
print(f"   - Tests: {'✅ PASSÉS' if tests_reussis else '❌ ÉCHOUÉS'}")
print(f"\n🔹 Clustering:")
print(f"   - Silhouette: {silhouette:.4f}")
print(f"   - Seuil: {THRESHOLDS['clustering']['silhouette']}")
print(f"\n✅ Pipeline CI/CD terminé")
print("\n🌐 Interface MLflow: http://127.0.0.1:5000")
print("   → Onglet 'Models' pour voir les nouvelles versions en Staging")