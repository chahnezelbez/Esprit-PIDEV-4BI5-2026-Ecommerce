"""
DAY 5 - MLOps pour Sougui.tn (Général Manager)
Optimisation des hyperparamètres avec Optuna
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import warnings
import optuna
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

THRESHOLDS = {"classifier": {"accuracy": 0.95}, "regressor": {"rmse": 200.0, "r2": 0.80}}
print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ============================================
# 1. GÉNÉRATION DES DONNÉES
# ============================================
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "Recency": np.random.randint(1, 365, n),
    "Frequency": np.random.randint(1, 20, n),
    "Monetary": np.random.exponential(500, n).round(2),
    "Avg_Order_Value": np.random.exponential(150, n).round(2),
    "Customer_Age_Days": np.random.randint(30, 1095, n),
    "Pct_Weekend": np.random.random(n),
    "Nb_Categories": np.random.randint(1, 8, n),
    "Is_Online_Buyer": np.random.choice([0, 1], n),
})
df["Churn"] = ((df["Recency"] > 90) | (df["Frequency"] < 3)).astype(int)
print(f"✅ Données: {len(df)} lignes")

# ============================================
# 2. OPTUNA - CLASSIFICATION
# ============================================
print("\n" + "="*60)
print("📊 OPTUNA - OPTIMISATION CLASSIFICATION")
print("="*60)

features_clf = ["Recency", "Customer_Age_Days", "Pct_Weekend", "Frequency", 
                "Nb_Categories", "Monetary", "Avg_Order_Value"]
X_clf = df[features_clf].fillna(0)
y_clf = df["Churn"]
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
    return accuracy_score(y_test, model.predict(X_test))

print("🔍 Recherche des meilleurs paramètres...")
study_clf = optuna.create_study(direction="maximize")
study_clf.optimize(objective_clf, n_trials=30, show_progress_bar=True)

print(f"\n✅ Meilleurs paramètres: {study_clf.best_params}")
print(f"📈 Meilleure accuracy: {study_clf.best_value:.4f}")

best_model_clf = RandomForestClassifier(**study_clf.best_params, random_state=42, n_jobs=-1)
best_model_clf.fit(X_train, y_train)
y_pred = best_model_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

with mlflow.start_run(run_name="Optuna_Classifier_Best"):
    mlflow.log_params(study_clf.best_params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("optimizer", "Optuna")
    mlflow.log_param("n_trials", 30)
    signature = mlflow.models.infer_signature(X_train, best_model_clf.predict(X_train))
    mlflow.sklearn.log_model(best_model_clf, "model", signature=signature)
    
    if acc >= THRESHOLDS["classifier"]["accuracy"]:
        result = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "SOUGUI_Classifier_Optuna")
        client.transition_model_version_stage(name="SOUGUI_Classifier_Optuna", version=result.version, stage="Production")
        print(f"\n✅ Classifieur optimisé → Production (accuracy={acc:.4f})")

# ============================================
# 3. OPTUNA - RÉGRESSION
# ============================================
print("\n" + "="*60)
print("📊 OPTUNA - OPTIMISATION RÉGRESSION")
print("="*60)

features_reg = ["Recency", "Frequency", "Monetary", "Customer_Age_Days", 
                "Pct_Weekend", "Nb_Categories", "Is_Online_Buyer"]
X_reg = df[features_reg].fillna(0)
y_reg = df["Avg_Order_Value"]
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
    rmse = np.sqrt(mean_squared_error(y_test_r, model.predict(X_test_r)))
    return -rmse

print("🔍 Recherche des meilleurs paramètres...")
study_reg = optuna.create_study(direction="maximize")
study_reg.optimize(objective_reg, n_trials=30, show_progress_bar=True)

print(f"\n✅ Meilleurs paramètres: {study_reg.best_params}")
print(f"📈 Meilleur RMSE: {-study_reg.best_value:.2f}")

best_model_reg = RandomForestRegressor(**study_reg.best_params, random_state=42, n_jobs=-1)
best_model_reg.fit(X_train_r, y_train_r)
y_pred_r = best_model_reg.predict(X_test_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2 = r2_score(y_test_r, y_pred_r)

print(f"📈 Performance finale: RMSE={rmse:.2f}, R²={r2:.4f}")

with mlflow.start_run(run_name="Optuna_Regressor_Best"):
    mlflow.log_params(study_reg.best_params)
    mlflow.log_metrics({"rmse": rmse, "r2": r2})
    mlflow.log_param("optimizer", "Optuna")
    mlflow.log_param("n_trials", 30)
    signature = mlflow.models.infer_signature(X_train_r, best_model_reg.predict(X_train_r))
    mlflow.sklearn.log_model(best_model_reg, "model", signature=signature)
    
    if rmse <= THRESHOLDS["regressor"]["rmse"] and r2 >= THRESHOLDS["regressor"]["r2"]:
        result = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "SOUGUI_Regressor_Optuna")
        client.transition_model_version_stage(name="SOUGUI_Regressor_Optuna", version=result.version, stage="Production")
        print(f"\n✅ Régresseur optimisé → Production (RMSE={rmse:.2f}, R²={r2:.4f})")

print("\n✅ Day 5 terminé. Optimisation Optuna complète.")