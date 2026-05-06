"""
DAY 6 - MLOps pour Sougui.tn (Général Manager)
Déploiement et Monitoring avec SQLite
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import sqlite3
import time
import warnings
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

MONITORING_DB = "monitoring.db"
PRODUCTION_THRESHOLDS = {
    "classifier": {"accuracy": 0.95, "latency_ms": 100},
    "regressor": {"rmse": 200.0, "r2": 0.80, "latency_ms": 100}
}

print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ============================================
# 1. INITIALISATION BASE SQLITE
# ============================================
def init_monitoring_db():
    conn = sqlite3.connect(MONITORING_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS monitoring (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model_type TEXT,
            metric_name TEXT,
            metric_value REAL,
            threshold REAL,
            status TEXT,
            latency_ms REAL,
            details TEXT
        )
    """)
    conn.commit()
    conn.close()

init_monitoring_db()
print("✅ Base SQLite initialisée")

def save_metric_to_db(model_type, metric_name, metric_value, threshold, status, latency_ms=None, details=None):
    conn = sqlite3.connect(MONITORING_DB)
    conn.execute("""
        INSERT INTO monitoring (timestamp, model_type, metric_name, metric_value, threshold, status, latency_ms, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), model_type, metric_name, metric_value, threshold, status, latency_ms, details))
    conn.commit()
    conn.close()

# ============================================
# 2. GÉNÉRATION DES DONNÉES
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
# 3. RÉCUPÉRATION DES MODÈLES PRODUCTION
# ============================================
def get_production_model(model_name):
    try:
        latest_version = client.get_latest_versions(model_name, stages=["Production"])[0]
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"  ✅ {model_name} - version {latest_version.version} chargé")
        return model, latest_version.version
    except Exception as e:
        print(f"  ⚠️ {model_name} non trouvé en Production: {e}")
        return None, None

print("\n" + "="*60)
print("📊 RÉCUPÉRATION DES MODÈLES PRODUCTION")
print("="*60)

model_clf, version_clf = get_production_model("SOUGUI_Classifier_Optuna")
model_reg, version_reg = get_production_model("SOUGUI_Regressor_Optuna")

if model_clf is None:
    print("🔄 Fallback: entraînement d'un modèle temporaire")
    features_clf = ["Recency", "Customer_Age_Days", "Pct_Weekend", "Frequency", 
                    "Nb_Categories", "Monetary", "Avg_Order_Value"]
    X_clf = df[features_clf].fillna(0)
    y_clf = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    model_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_clf.fit(X_train, y_train)
    version_clf = "local_fallback"

# ============================================
# 4. INFERENCE ET MONITORING
# ============================================
print("\n" + "="*60)
print("📊 INFERENCE ET MONITORING")
print("="*60)

# Classification
features_clf = ["Recency", "Customer_Age_Days", "Pct_Weekend", "Frequency", 
                "Nb_Categories", "Monetary", "Avg_Order_Value"]
X_test_clf = df[features_clf].fillna(0).iloc[-200:]
y_true_clf = df["Churn"].iloc[-200:]

start_time = time.time()
y_pred_clf = model_clf.predict(X_test_clf)
latency_clf = (time.time() - start_time) * 1000

accuracy = accuracy_score(y_true_clf, y_pred_clf)
status = "OK" if accuracy >= PRODUCTION_THRESHOLDS["classifier"]["accuracy"] else "ALERTE"

save_metric_to_db("classifier", "accuracy", accuracy, PRODUCTION_THRESHOLDS["classifier"]["accuracy"], status, latency_clf, f"version_{version_clf}")
print(f"\n📈 Classification:")
print(f"   Accuracy: {accuracy:.4f} (seuil: {PRODUCTION_THRESHOLDS['classifier']['accuracy']})")
print(f"   Latence: {latency_clf:.2f} ms | Statut: {'✅ OK' if status == 'OK' else '❌ ALERTE'}")

# Régression
if model_reg:
    features_reg = ["Recency", "Frequency", "Monetary", "Customer_Age_Days", 
                    "Pct_Weekend", "Nb_Categories", "Is_Online_Buyer"]
    X_test_reg = df[features_reg].fillna(0).iloc[-200:]
    y_true_reg = df["Avg_Order_Value"].iloc[-200:]
    
    start_time = time.time()
    y_pred_reg = model_reg.predict(X_test_reg)
    latency_reg = (time.time() - start_time) * 1000
    
    rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
    r2 = r2_score(y_true_reg, y_pred_reg)
    
    status = "OK" if (rmse <= PRODUCTION_THRESHOLDS["regressor"]["rmse"] and 
                     r2 >= PRODUCTION_THRESHOLDS["regressor"]["r2"]) else "ALERTE"
    
    save_metric_to_db("regressor", "rmse", rmse, PRODUCTION_THRESHOLDS["regressor"]["rmse"], status, latency_reg, f"version_{version_reg}")
    save_metric_to_db("regressor", "r2", r2, PRODUCTION_THRESHOLDS["regressor"]["r2"], status, latency_reg, f"version_{version_reg}")
    
    print(f"\n📈 Régression:")
    print(f"   RMSE: {rmse:.2f} (seuil: {PRODUCTION_THRESHOLDS['regressor']['rmse']})")
    print(f"   R²: {r2:.4f} (seuil: {PRODUCTION_THRESHOLDS['regressor']['r2']})")
    print(f"   Latence: {latency_reg:.2f} ms | Statut: {'✅ OK' if status == 'OK' else '❌ ALERTE'}")

# ============================================
# 5. DÉTECTION DE DÉRIVE
# ============================================
print("\n" + "="*60)
print("📊 DÉTECTION DE DÉRIVE")
print("="*60)

from scipy import stats

np.random.seed(42)
df_train_ref = pd.DataFrame({"Recency": np.random.randint(1, 365, 500), "Frequency": np.random.randint(1, 20, 500)})
df_test_current = pd.DataFrame({"Recency": df["Recency"].iloc[-200:], "Frequency": df["Frequency"].iloc[-200:]})

drift_detected = False
for col in ["Recency", "Frequency"]:
    ks_stat, p_value = stats.ks_2samp(df_train_ref[col], df_test_current[col])
    if p_value < 0.05:
        print(f"  ⚠️ Dérive détectée sur {col}: p-value={p_value:.4f}")
        drift_detected = True
    else:
        print(f"  ✅ {col}: distribution stable (p-value={p_value:.4f})")

if drift_detected:
    save_metric_to_db("system", "data_drift", 1.0, 0, "ALERTE", details="distribution_changed")
    print("\n⚠️ ALERTE: Dérive des données détectée!")
else:
    print("\n✅ Pas de dérive détectée. Modèle stable.")

# ============================================
# 6. CONSULTATION HISTORIQUE
# ============================================
print("\n" + "="*60)
print("📊 HISTORIQUE DES MONITORING")
print("="*60)

conn = sqlite3.connect(MONITORING_DB)
cursor = conn.cursor()
cursor.execute("SELECT * FROM monitoring ORDER BY id DESC LIMIT 10")
rows = cursor.fetchall()
conn.close()

print("\nDernières entrées:")
for row in rows:
    print(f"   {row[1]} | {row[2]} | {row[3]}={row[4]:.4f} | {row[6]}")

print("\n✅ Day 6 terminé. Monitoring actif.")