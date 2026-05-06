"""
DAY 4 - MLOps pour Sougui.tn (Général Manager)
CI/CD Pipeline + Tests Automatiques + Déploiement
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# Seuils de performance pour la promotion
THRESHOLDS = {
    "classifier": {"accuracy": 0.95, "precision": 0.95, "recall": 0.95, "f1": 0.95},
    "regressor": {"rmse": 200.0, "r2": 0.80}
}

print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"📊 Seuils de performance: {THRESHOLDS}")

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
# 2. CLASSIFICATION - PIPELINE CI/CD
# ============================================
print("\n" + "="*60)
print("📊 CLASSIFICATION PIPELINE")
print("="*60)

features_clf = ["Recency", "Customer_Age_Days", "Pct_Weekend", "Frequency", 
                "Nb_Categories", "Monetary", "Avg_Order_Value"]
X_clf = df[features_clf].fillna(0)
y_clf = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

model_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_clf.fit(X_train, y_train)
y_pred = model_clf.predict(X_test)

from sklearn.metrics import precision_score, recall_score, f1_score
metrics_clf = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0)
}

print(f"📈 Performance Classifieur:")
for k, v in metrics_clf.items():
    print(f"   {k}: {v:.4f}")

tests_reussis = True
for metric, value in metrics_clf.items():
    if metric in THRESHOLDS["classifier"]:
        seuil = THRESHOLDS["classifier"][metric]
        if value < seuil:
            print(f"   ❌ Échec: {metric} = {value:.4f} < seuil {seuil}")
            tests_reussis = False
        else:
            print(f"   ✅ {metric} = {value:.4f} ≥ seuil {seuil}")

with mlflow.start_run(run_name="CI_Classifier_RF"):
    mlflow.log_params({"model_type": "RandomForestClassifier", "n_estimators": 100, "pipeline_stage": "CI_test"})
    for k, v in metrics_clf.items():
        mlflow.log_metric(k, v)
    signature = mlflow.models.infer_signature(X_train, model_clf.predict(X_train))
    mlflow.sklearn.log_model(model_clf, "model", signature=signature)
    
    if tests_reussis:
        result = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "SOUGUI_Classifier_RF")
        client.transition_model_version_stage(name="SOUGUI_Classifier_RF", version=result.version, stage="Staging")
        print(f"\n✅ Classifieur version {result.version} → Staging")
        if metrics_clf["accuracy"] >= 0.98:
            client.transition_model_version_stage(name="SOUGUI_Classifier_RF", version=result.version, stage="Production")
            print(f"✅ Classifieur version {result.version} → Production")
    else:
        print(f"\n❌ Classifieur non promu (tests échoués)")

# ============================================
# 3. RÉGRESSION - PIPELINE CI/CD
# ============================================
print("\n" + "="*60)
print("📊 RÉGRESSION PIPELINE")
print("="*60)

features_reg = ["Recency", "Frequency", "Monetary", "Customer_Age_Days", 
                "Pct_Weekend", "Nb_Categories", "Is_Online_Buyer"]
X_reg = df[features_reg].fillna(0)
y_reg = df["Avg_Order_Value"]
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

with mlflow.start_run(run_name="CI_Regressor_RF"):
    mlflow.log_params({"model_type": "RandomForestRegressor", "n_estimators": 100, "pipeline_stage": "CI_test"})
    mlflow.log_metrics({"rmse": rmse, "r2": r2})
    signature = mlflow.models.infer_signature(X_train_r, model_reg.predict(X_train_r))
    mlflow.sklearn.log_model(model_reg, "model", signature=signature)
    
    if tests_reussis:
        result = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", "SOUGUI_Regressor_RF")
        client.transition_model_version_stage(name="SOUGUI_Regressor_RF", version=result.version, stage="Staging")
        print(f"\n✅ Régresseur version {result.version} → Staging")
        if rmse < 150:
            client.transition_model_version_stage(name="SOUGUI_Regressor_RF", version=result.version, stage="Production")
            print(f"✅ Régresseur version {result.version} → Production")
    else:
        print(f"\n❌ Régresseur non promu (tests échoués)")

print("\n✅ Day 4 terminé. Pipeline CI/CD opérationnel.")