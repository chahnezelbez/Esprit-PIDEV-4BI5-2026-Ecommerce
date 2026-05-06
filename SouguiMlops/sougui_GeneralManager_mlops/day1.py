"""
DAY 1 - MLOps pour Sougui.tn (Général Manager)
Tracking de base MLflow avec fallback données synthétiques
Cas d'usage: Classification (Churn), Régression (Prix), Clustering (Segmentation)
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import hashlib
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ============================================
# 1. CHARGEMENT DES DONNÉES
# ============================================
use_mysql = False
df = None

try:
    engine = create_engine("mysql+pymysql://root:@127.0.0.1:3306/dwh_sougui")
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    use_mysql = True
    print("✅ Connecté à MySQL - chargement des données réelles")
    df = pd.read_sql("""
        SELECT client_id, Recency, Frequency, Monetary, Avg_Order_Value,
               Customer_Age_Days, Pct_Weekend, Nb_Categories, Is_Online_Buyer, Churn
        FROM rfm_client
        LIMIT 1000
    """, engine)
except Exception as e:
    print(f"⚠️ MySQL indisponible ({e})")
    print("🔄 Génération de données synthétiques...")
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
    use_mysql = False

data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
data_source = "MySQL" if use_mysql else "synthetic"
mlflow.set_experiment("SOUGUI_GM_Day1")

# ============================================
# 2. CLASSIFICATION - Churn client
# ============================================
features_clf = ["Recency", "Customer_Age_Days", "Pct_Weekend", "Frequency", 
                "Nb_Categories", "Monetary", "Avg_Order_Value"]
X_clf = df[features_clf].fillna(0)
y_clf = df["Churn"]

model_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_clf.fit(X_clf, y_clf)
acc_clf = accuracy_score(y_clf, model_clf.predict(X_clf))

with mlflow.start_run(run_name="Classification_Churn_RF"):
    mlflow.log_params({"model_type": "RandomForestClassifier", "n_estimators": 100,
                       "dataset_hash": data_hash, "data_source": data_source})
    mlflow.log_metric("accuracy", acc_clf)
    signature = mlflow.models.infer_signature(X_clf, model_clf.predict(X_clf))
    mlflow.sklearn.log_model(model_clf, "model", signature=signature)
    print(f"  ✅ Classification Churn - Accuracy: {acc_clf:.4f}")

# ============================================
# 3. RÉGRESSION - Prix produits
# ============================================
features_reg = ["Recency", "Frequency", "Monetary", "Avg_Order_Value", 
                "Customer_Age_Days", "Pct_Weekend", "Nb_Categories"]
X_reg = df[features_reg].fillna(0)
y_reg = df["Avg_Order_Value"]

model_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_reg.fit(X_reg, y_reg)
rmse_reg = np.sqrt(mean_squared_error(y_reg, model_reg.predict(X_reg)))

with mlflow.start_run(run_name="Regression_Price_RF"):
    mlflow.log_params({"model_type": "RandomForestRegressor", "n_estimators": 100,
                       "dataset_hash": data_hash, "data_source": data_source})
    mlflow.log_metric("rmse", rmse_reg)
    signature = mlflow.models.infer_signature(X_reg, model_reg.predict(X_reg))
    mlflow.sklearn.log_model(model_reg, "model", signature=signature)
    print(f"  ✅ Régression Prix - RMSE: {rmse_reg:.2f}")

# ============================================
# 4. CLUSTERING - Segmentation clients
# ============================================
X_clust = df[["Monetary", "Frequency", "Recency"]].values
scaler = StandardScaler()
X_clust_scaled = scaler.fit_transform(X_clust)

model_clust = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_clust = model_clust.fit_predict(X_clust_scaled)
silhouette = silhouette_score(X_clust_scaled, labels_clust)

with mlflow.start_run(run_name="Clustering_Segmentation_KMeans"):
    mlflow.log_params({"n_clusters": 4, "dataset_hash": data_hash, "data_source": data_source})
    mlflow.log_metric("silhouette_score", silhouette)
    with open("clust_labels.pkl", "wb") as f:
        pickle.dump(labels_clust, f)
    mlflow.log_artifact("clust_labels.pkl")
    print(f"  ✅ Clustering Segmentation - Silhouette: {silhouette:.4f}")

print("\n" + "="*60)
print("📈 RÉSUMÉ DAY 1")
print("="*60)
print(f"🔹 Source données: {data_source}")
print(f"🔹 Classification Churn - Accuracy: {acc_clf:.4f}")
print(f"🔹 Régression Prix - RMSE: {rmse_reg:.2f}")
print(f"🔹 Clustering - Silhouette: {silhouette:.4f}")
print("\n✅ 3 runs enregistrés dans MLflow")
print("🌐 Interface MLflow: http://127.0.0.1:5000")