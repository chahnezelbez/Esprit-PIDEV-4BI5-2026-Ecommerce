"""
Day 5 - MLOps pour Sougui.tn (Purchasing)
Optimisation des hyperparamètres avec Optuna

Objectifs:
- Optimiser les modèles de classification, régression et clustering
- Trouver les meilleurs hyperparamètres automatiquement
- Enregistrer les meilleurs modèles dans MLflow
- Atteindre les seuils de performance (accuracy ≥ 0.95, R² ≥ 0.7)
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import pickle
import warnings
import optuna
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score, r2_score
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

# ============================================
# CONFIGURATION
# ============================================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# Seuils de performance
THRESHOLDS = {
    "classifier": {"accuracy": 0.95, "precision": 0.95, "recall": 0.95, "f1": 0.95},
    "regressor": {"rmse": 500.0, "r2": 0.70},
    "clustering": {"silhouette": 0.40}
}

print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"📊 Seuils cibles: {THRESHOLDS}")

# ============================================
# 1. GÉNÉRATION DES DONNÉES
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
df["Facture_Standard"] = (df["Taux_TVA"] > np.median(df["Taux_TVA"])).astype(int)
print(f"✅ Données: {len(df)} lignes")
print(f"   Distribution Facture_Standard: {df['Facture_Standard'].value_counts().to_dict()}")

# ============================================
# 2. OPTUNA - CLASSIFICATION
# ============================================
print("\n" + "="*60)
print("📊 OPTUNA - OPTIMISATION CLASSIFICATION")
print("="*60)

features_clf = ["Montant_HT", "Taux_TVA", "Mois", "Annee", "Est_weekend",
                "Categorie_Enc", "Fournisseur_Enc", "Methode_Enc"]
X_clf = df[features_clf].fillna(0)
y_clf = df["Facture_Standard"]
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

def objective_clf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    }
    
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

print("🔍 Recherche des meilleurs paramètres...")
study_clf = optuna.create_study(direction="maximize")
study_clf.optimize(objective_clf, n_trials=30, show_progress_bar=True)

best_params_clf = study_clf.best_params
best_accuracy = study_clf.best_value

print(f"\n✅ Meilleurs paramètres trouvés:")
for k, v in best_params_clf.items():
    print(f"   {k}: {v}")
print(f"📈 Meilleure accuracy: {best_accuracy:.4f}")

# Entraînement du meilleur modèle
best_model_clf = RandomForestClassifier(**best_params_clf, random_state=42, n_jobs=-1)
best_model_clf.fit(X_train, y_train)
y_pred = best_model_clf.predict(X_test)

from sklearn.metrics import precision_score, recall_score, f1_score
metrics_clf = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0)
}

# Enregistrement dans MLflow
with mlflow.start_run(run_name="Optuna_Classifier_Best"):
    mlflow.log_params(best_params_clf)
    mlflow.log_metrics(metrics_clf)
    mlflow.log_param("optimizer", "Optuna")
    mlflow.log_param("n_trials", 30)
    
    signature = mlflow.models.infer_signature(X_train, best_model_clf.predict(X_train))
    mlflow.sklearn.log_model(best_model_clf, "model", signature=signature)
    
    # Vérification des seuils
    if metrics_clf["accuracy"] >= THRESHOLDS["classifier"]["accuracy"]:
        result = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            "SOUGUI_Classifier_Optuna"
        )
        client.transition_model_version_stage(
            name="SOUGUI_Classifier_Optuna",
            version=result.version,
            stage="Production"
        )
        print(f"\n✅ Classifieur optimisé → Production (accuracy={metrics_clf['accuracy']:.4f})")
    else:
        print(f"\n⚠️ Classifieur non promu (accuracy={metrics_clf['accuracy']:.4f} < seuil)")

# ============================================
# 3. OPTUNA - RÉGRESSION
# ============================================
print("\n" + "="*60)
print("📊 OPTUNA - OPTIMISATION RÉGRESSION")
print("="*60)

features_reg = ["Mois", "Annee", "Est_weekend", "Categorie_Enc", 
                "Fournisseur_Enc", "Methode_Enc", "Taux_TVA", "Montant_TVA"]
X_reg = df[features_reg].fillna(0)
y_reg = df["Montant_HT"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

def objective_reg(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    }
    
    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(X_train_r, y_train_r)
    y_pred = model.predict(X_test_r)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))
    return -rmse  # Minimiser RMSE

print("🔍 Recherche des meilleurs paramètres...")
study_reg = optuna.create_study(direction="maximize")
study_reg.optimize(objective_reg, n_trials=30, show_progress_bar=True)

best_params_reg = study_reg.best_params
best_rmse = -study_reg.best_value

print(f"\n✅ Meilleurs paramètres trouvés:")
for k, v in best_params_reg.items():
    print(f"   {k}: {v}")
print(f"📈 Meilleur RMSE: {best_rmse:.2f}")

# Entraînement du meilleur modèle
best_model_reg = RandomForestRegressor(**best_params_reg, random_state=42, n_jobs=-1)
best_model_reg.fit(X_train_r, y_train_r)
y_pred_r = best_model_reg.predict(X_test_r)

rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2 = r2_score(y_test_r, y_pred_r)

print(f"📈 Performance finale:")
print(f"   RMSE: {rmse:.2f}")
print(f"   R²: {r2:.4f}")

# Enregistrement dans MLflow
with mlflow.start_run(run_name="Optuna_Regressor_Best"):
    mlflow.log_params(best_params_reg)
    mlflow.log_metrics({"rmse": rmse, "r2": r2})
    mlflow.log_param("optimizer", "Optuna")
    mlflow.log_param("n_trials", 30)
    
    signature = mlflow.models.infer_signature(X_train_r, best_model_reg.predict(X_train_r))
    mlflow.sklearn.log_model(best_model_reg, "model", signature=signature)
    
    if rmse <= THRESHOLDS["regressor"]["rmse"] and r2 >= THRESHOLDS["regressor"]["r2"]:
        result = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model",
            "SOUGUI_Regressor_Optuna"
        )
        client.transition_model_version_stage(
            name="SOUGUI_Regressor_Optuna",
            version=result.version,
            stage="Production"
        )
        print(f"\n✅ Régresseur optimisé → Production (RMSE={rmse:.2f}, R²={r2:.4f})")
    else:
        print(f"\n⚠️ Régresseur non promu (RMSE={rmse:.2f}, R²={r2:.4f})")

# ============================================
# 4. OPTUNA - CLUSTERING (k optimal)
# ============================================
print("\n" + "="*60)
print("📊 OPTUNA - OPTIMISATION CLUSTERING")
print("="*60)

df_fourn = df.groupby("ID_Fournisseur").agg({
    "Montant_HT": ["sum", "mean", "count"],
    "Taux_TVA": "mean"
})
df_fourn.columns = ["Montant_Total", "Montant_Moyen", "Nb_Factures", "TVA_Moy"]
df_fourn = df_fourn.fillna(0)

scaler = StandardScaler()
X_clust = scaler.fit_transform(df_fourn[["Montant_Total", "Nb_Factures", "TVA_Moy"]])

def objective_clust(trial):
    k = trial.suggest_int("n_clusters", 2, 8)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_clust)
    
    if len(set(labels)) > 1:
        return silhouette_score(X_clust, labels)
    else:
        return -1.0

print("🔍 Recherche du nombre optimal de clusters...")
study_clust = optuna.create_study(direction="maximize")
study_clust.optimize(objective_clust, n_trials=20, show_progress_bar=True)

best_k = study_clust.best_params["n_clusters"]
best_silhouette = study_clust.best_value

print(f"\n✅ Meilleur k: {best_k}")
print(f"📈 Meilleure silhouette: {best_silhouette:.4f}")

# Entraînement du meilleur modèle
best_model_clust = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = best_model_clust.fit_predict(X_clust)

with mlflow.start_run(run_name="Optuna_Clustering_Best"):
    mlflow.log_params({"n_clusters": best_k, "optimizer": "Optuna", "n_trials": 20})
    mlflow.log_metric("silhouette_score", best_silhouette)
    
    with open("clust_model_optuna.pkl", "wb") as f:
        pickle.dump({"model": best_model_clust, "scaler": scaler}, f)
    mlflow.log_artifact("clust_model_optuna.pkl")
    
    print(f"\n✅ Clustering optimisé - k={best_k}, Silhouette={best_silhouette:.4f}")

# ============================================
# 5. COMPARAISON DES MODÈLES
# ============================================
print("\n" + "="*60)
print("📊 COMPARAISON DES PERFORMANCES")
print("="*60)

print("\n🔹 Classification:")
print(f"   Avant Optuna (RF): accuracy=1.0000")
print(f"   Après Optuna:      accuracy={best_accuracy:.4f}")

print("\n🔹 Régression:")
print(f"   Avant Optuna (RF): RMSE=373.81, R²=0.27")
print(f"   Après Optuna:      RMSE={rmse:.2f}, R²={r2:.4f}")

print("\n🔹 Clustering:")
print(f"   Avant Optuna: k=3, silhouette=0.4627")
print(f"   Après Optuna:  k={best_k}, silhouette={best_silhouette:.4f}")

# ============================================
# 6. RAPPORT FINAL
# ============================================
print("\n" + "="*60)
print("📈 RÉSUMÉ DAY 5 - OPTUNA OPTIMIZATION")
print("="*60)
print(f"\n🔹 Modèles optimisés enregistrés:")
print(f"   - SOUGUI_Classifier_Optuna (Production)" if best_accuracy >= THRESHOLDS["classifier"]["accuracy"] else "   - Classifieur: ⚠️ Seuil non atteint")
print(f"   - SOUGUI_Regressor_Optuna (Production)" if rmse <= THRESHOLDS["regressor"]["rmse"] and r2 >= THRESHOLDS["regressor"]["r2"] else "   - Régresseur: ⚠️ Seuil non atteint")
print(f"   - Clustering: k={best_k}, silhouette={best_silhouette:.4f}")

print("\n🔹 Améliorations:")
print(f"   - Régression: R² passé de 0.27 à {r2:.4f}")

print("\n🌐 Interface MLflow: http://127.0.0.1:5000")
print("   → Onglet 'Models' pour voir les modèles optimisés")