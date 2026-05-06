"""
DAY 3 - MLOps pour Sougui.tn (Général Manager)
Model Registry + Versioning + Promotion Staging → Production
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()
print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ============================================
# 1. GÉNÉRATION DES DONNÉES
# ============================================
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "client_id": np.random.randint(1, 500, n),
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

features_clf = ["Recency", "Customer_Age_Days", "Pct_Weekend", "Frequency", 
                "Nb_Categories", "Monetary", "Avg_Order_Value"]
X_clf = df[features_clf].fillna(0)
y_clf = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

features_reg = ["Recency", "Frequency", "Monetary", "Customer_Age_Days", 
                "Pct_Weekend", "Nb_Categories", "Is_Online_Buyer"]
X_reg = df[features_reg].fillna(0)
y_reg = df["Avg_Order_Value"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# ============================================
# 2. CLASSIFICATION - Meilleur modèle (RandomForest)
# ============================================
print("\n" + "="*50)
print("📊 CLASSIFICATION - Model Registry")
print("="*50)

model_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_clf.fit(X_train, y_train)
acc = accuracy_score(y_test, model_clf.predict(X_test))
print(f"✅ RandomForest - Accuracy: {acc:.4f}")

with mlflow.start_run(run_name="Classification_RF_Best") as run:
    mlflow.log_params({"model_type": "RandomForestClassifier", "n_estimators": 100, "purpose": "production_candidate"})
    mlflow.log_metric("accuracy", acc)
    signature = mlflow.models.infer_signature(X_train, model_clf.predict(X_train))
    mlflow.sklearn.log_model(model_clf, "model", signature=signature, registered_model_name="SOUGUI_Classifier_RF")
    print(f"  ✅ Modèle enregistré: SOUGUI_Classifier_RF (version 1)")

# ============================================
# 3. RÉGRESSION - Meilleur modèle (RandomForest)
# ============================================
print("\n" + "="*50)
print("📊 RÉGRESSION - Model Registry")
print("="*50)

model_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_reg.fit(X_train_r, y_train_r)
rmse = np.sqrt(mean_squared_error(y_test_r, model_reg.predict(X_test_r)))
print(f"✅ RandomForest - RMSE: {rmse:.2f}")

with mlflow.start_run(run_name="Regression_RF_Best") as run:
    mlflow.log_params({"model_type": "RandomForestRegressor", "n_estimators": 100, "purpose": "production_candidate"})
    mlflow.log_metric("rmse", rmse)
    signature = mlflow.models.infer_signature(X_train_r, model_reg.predict(X_train_r))
    mlflow.sklearn.log_model(model_reg, "model", signature=signature, registered_model_name="SOUGUI_Regressor_RF")
    print(f"  ✅ Modèle enregistré: SOUGUI_Regressor_RF (version 1)")

# ============================================
# 4. TRANSITION STAGING → PRODUCTION
# ============================================
print("\n" + "="*50)
print("📊 TRANSITION DES MODÈLES")
print("="*50)

for model_name in ["SOUGUI_Classifier_RF", "SOUGUI_Regressor_RF"]:
    try:
        latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
        client.transition_model_version_stage(name=model_name, version=latest_version, stage="Staging")
        print(f"  ✅ {model_name} version {latest_version} → Staging")
        
        # Promotion automatique en Production si performance excellente
        if model_name == "SOUGUI_Classifier_RF" and acc >= 0.95:
            client.transition_model_version_stage(name=model_name, version=latest_version, stage="Production")
            print(f"  ✅ {model_name} version {latest_version} → Production (accuracy={acc:.4f})")
        elif model_name == "SOUGUI_Regressor_RF" and rmse <= 200:
            client.transition_model_version_stage(name=model_name, version=latest_version, stage="Production")
            print(f"  ✅ {model_name} version {latest_version} → Production (RMSE={rmse:.2f})")
    except Exception as e:
        print(f"  ⚠️ Transition {model_name}: {e}")

# ============================================
# 5. TEST DE CHARGEMENT DEPUIS LE REGISTRY
# ============================================
print("\n" + "="*50)
print("📊 TEST - Chargement depuis Registry")
print("="*50)

try:
    model_uri = "models:/SOUGUI_Classifier_RF/Staging"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    test_pred = loaded_model.predict(X_test[:5])
    print(f"  ✅ Classifieur chargé depuis Staging - Prédictions: {test_pred}")
except Exception as e:
    print(f"  ⚠️ Impossible de charger le classifieur: {e}")

try:
    model_uri = "models:/SOUGUI_Regressor_RF/Staging"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    test_pred = loaded_model.predict(X_test_r[:5])
    print(f"  ✅ Régresseur chargé depuis Staging - Prédictions: {test_pred[:3]}...")
except Exception as e:
    print(f"  ⚠️ Impossible de charger le régresseur: {e}")

print("\n✅ Day 3 terminé. Modèles versionnés et promus.")