"""
Day 6 - MLOps pour Sougui.tn (Purchasing)
Déploiement et Monitoring avec SQLite

Objectifs:
- Déployer les modèles optimisés via MLflow Models (API REST)
- Monitorer les performances en production
- Sauvegarder les métriques dans SQLite
- Détecter la dérive des données
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import sqlite3
import json
import time
import requests
import pickle
import warnings
from datetime import datetime
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

# ============================================
# CONFIGURATION
# ============================================
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# Base SQLite pour le monitoring
MONITORING_DB = "monitoring.db"

# Configuration du serveur MLflow Models
MLFLOW_MODELS_URI = "http://127.0.0.1:5001"

# Seuils de performance en production
PRODUCTION_THRESHOLDS = {
    "classifier": {"accuracy": 0.95, "latency_ms": 100},
    "regressor": {"rmse": 200.0, "r2": 0.80, "latency_ms": 100},
    "clustering": {"silhouette": 0.40}
}

print(f"✅ MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"📊 Seuils production: {PRODUCTION_THRESHOLDS}")

# ============================================
# 1. INITIALISATION DE LA BASE SQLITE
# ============================================
def init_monitoring_db():
    """Crée la table de monitoring si elle n'existe pas"""
    conn = sqlite3.connect(MONITORING_DB)
    cursor = conn.cursor()
    cursor.execute("""
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
    print("✅ Base SQLite initialisée")

init_monitoring_db()

def save_metric_to_db(model_type, metric_name, metric_value, threshold, status, latency_ms=None, details=None):
    """Sauvegarde une métrique dans SQLite"""
    conn = sqlite3.connect(MONITORING_DB)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO monitoring (timestamp, model_type, metric_name, metric_value, threshold, status, latency_ms, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().isoformat(), model_type, metric_name, metric_value, threshold, status, latency_ms, details))
    conn.commit()
    conn.close()

# ============================================
# 2. RÉCUPÉRATION DES MODÈLES PRODUCTION
# ============================================
print("\n" + "="*60)
print("📊 RÉCUPÉRATION DES MODÈLES PRODUCTION")
print("="*60)

def get_production_model(model_name):
    """Récupère la dernière version en Production d'un modèle"""
    try:
        latest_version = client.get_latest_versions(model_name, stages=["Production"])[0]
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"   ✅ {model_name} - version {latest_version.version} chargé")
        return model, latest_version.version
    except Exception as e:
        print(f"   ⚠️ {model_name} non trouvé en Production: {e}")
        return None, None

# Récupération des modèles
model_clf, version_clf = get_production_model("SOUGUI_Classifier_Optuna")
model_reg, version_reg = get_production_model("SOUGUI_Regressor_Optuna")

# Pour le clustering (pas de registry standard)
try:
    with open("clust_model_optuna.pkl", "rb") as f:
        clust_data = pickle.load(f)
    model_clust = clust_data["model"]
    scaler = clust_data["scaler"]
    print(f"   ✅ Clustering modèle chargé")
except Exception as e:
    print(f"   ⚠️ Clustering non trouvé: {e}")
    model_clust = None

# ============================================
# 3. TEST DES MODÈLES VIA API (si serveur actif)
# ============================================
print("\n" + "="*60)
print("📊 TEST DES MODÈLES VIA API REST")
print("="*60)

def test_api_model(model_name, input_data):
    """Teste un modèle déployé via l'API MLflow Models"""
    try:
        response = requests.post(
            f"{MLFLOW_MODELS_URI}/invocations",
            headers={"Content-Type": "application/json"},
            json={"dataframe_split": input_data.to_dict(orient="split")}
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"   ⚠️ API {model_name} erreur: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"   ⚠️ Serveur MLflow Models non démarré sur port 5001")
        return None

# Génération d'un petit échantillon pour le test
sample_data = pd.DataFrame({
    "Montant_HT": [500, 1000, 1500],
    "Taux_TVA": [0.19, 0.07, 0.19],
    "Mois": [6, 6, 6],
    "Annee": [2025, 2025, 2025],
    "Est_weekend": [0, 1, 0],
    "Categorie_Enc": [1, 2, 1],
    "Fournisseur_Enc": [5, 5, 5],
    "Methode_Enc": [2, 2, 2],
})

print("🔍 Test des API...")
api_result = test_api_model("SOUGUI_Classifier_Optuna", sample_data)
if api_result:
    print(f"   ✅ API Classification OK - Prédictions: {api_result}")
else:
    print("   ⚠️ Démarrez le serveur MLflow Models pour tester l'API")

# ============================================
# 4. SIMULATION DE DONNÉES PRODUCTION
# ============================================
print("\n" + "="*60)
print("📊 SIMULATION DE DONNÉES PRODUCTION")
print("="*60)

np.random.seed(999)  # seed différent pour simuler des données réelles
n_test = 100

df_test = pd.DataFrame({
    "ID_Fournisseur": np.random.randint(1, 50, n_test),
    "Montant_HT": np.random.exponential(500, n_test).round(2) + 10,
    "Montant_TVA": np.random.exponential(100, n_test).round(2),
    "Taux_TVA": np.random.random(n_test),
    "Annee": np.random.choice([2025, 2026], n_test),
    "Mois": np.random.randint(1, 13, n_test),
    "Est_weekend": np.random.choice([0, 1], n_test),
    "Categorie_Enc": np.random.randint(0, 4, n_test),
    "Fournisseur_Enc": np.random.randint(0, 20, n_test),
    "Methode_Enc": np.random.randint(0, 5, n_test),
})

# Cible pour évaluation
df_test["Facture_Standard"] = (df_test["Taux_TVA"] > 0.5).astype(int)

print(f"✅ Données test: {len(df_test)} lignes")

# ============================================
# 5. INFERENCE ET MONITORING
# ============================================
print("\n" + "="*60)
print("📊 INFERENCE ET MONITORING")
print("="*60)

# Classification
if model_clf:
    features_clf = ["Montant_HT", "Taux_TVA", "Mois", "Annee", "Est_weekend",
                    "Categorie_Enc", "Fournisseur_Enc", "Methode_Enc"]
    X_test_clf = df_test[features_clf].fillna(0)
    y_true_clf = df_test["Facture_Standard"]
    
    start_time = time.time()
    y_pred_clf = model_clf.predict(X_test_clf)
    latency_clf = (time.time() - start_time) * 1000
    
    accuracy = accuracy_score(y_true_clf, y_pred_clf)
    
    status = "OK" if accuracy >= PRODUCTION_THRESHOLDS["classifier"]["accuracy"] else "ALERTE"
    
    save_metric_to_db(
        model_type="classifier",
        metric_name="accuracy",
        metric_value=accuracy,
        threshold=PRODUCTION_THRESHOLDS["classifier"]["accuracy"],
        status=status,
        latency_ms=latency_clf,
        details=f"version_{version_clf}"
    )
    
    print(f"\n📈 Classification:")
    print(f"   Accuracy: {accuracy:.4f} (seuil: {PRODUCTION_THRESHOLDS['classifier']['accuracy']})")
    print(f"   Latence: {latency_clf:.2f} ms")
    print(f"   Statut: {'✅ OK' if status == 'OK' else '❌ ALERTE'}")

# Régression
if model_reg:
    features_reg = ["Mois", "Annee", "Est_weekend", "Categorie_Enc", 
                    "Fournisseur_Enc", "Methode_Enc", "Taux_TVA", "Montant_TVA"]
    X_test_reg = df_test[features_reg].fillna(0)
    y_true_reg = df_test["Montant_HT"]
    
    start_time = time.time()
    y_pred_reg = model_reg.predict(X_test_reg)
    latency_reg = (time.time() - start_time) * 1000
    
    rmse = np.sqrt(mean_squared_error(y_true_reg, y_pred_reg))
    r2 = r2_score(y_true_reg, y_pred_reg)
    
    status = "OK" if (rmse <= PRODUCTION_THRESHOLDS["regressor"]["rmse"] and 
                     r2 >= PRODUCTION_THRESHOLDS["regressor"]["r2"]) else "ALERTE"
    
    save_metric_to_db(
        model_type="regressor",
        metric_name="rmse",
        metric_value=rmse,
        threshold=PRODUCTION_THRESHOLDS["regressor"]["rmse"],
        status=status,
        latency_ms=latency_reg,
        details=f"version_{version_reg}, r2={r2:.4f}"
    )
    
    save_metric_to_db(
        model_type="regressor",
        metric_name="r2",
        metric_value=r2,
        threshold=PRODUCTION_THRESHOLDS["regressor"]["r2"],
        status=status,
        latency_ms=latency_reg,
        details=f"version_{version_reg}, rmse={rmse:.2f}"
    )
    
    print(f"\n📈 Régression:")
    print(f"   RMSE: {rmse:.2f} (seuil: {PRODUCTION_THRESHOLDS['regressor']['rmse']})")
    print(f"   R²: {r2:.4f} (seuil: {PRODUCTION_THRESHOLDS['regressor']['r2']})")
    print(f"   Latence: {latency_reg:.2f} ms")
    print(f"   Statut: {'✅ OK' if status == 'OK' else '❌ ALERTE'}")

# Clustering
if model_clust:
    df_fourn_test = df_test.groupby("ID_Fournisseur").agg({
        "Montant_HT": ["sum", "mean", "count"],
        "Taux_TVA": "mean"
    })
    df_fourn_test.columns = ["Montant_Total", "Montant_Moyen", "Nb_Factures", "TVA_Moy"]
    df_fourn_test = df_fourn_test.fillna(0)
    
    X_clust_test = scaler.transform(df_fourn_test[["Montant_Total", "Nb_Factures", "TVA_Moy"]])
    labels_test = model_clust.predict(X_clust_test)
    
    silhouette = silhouette_score(X_clust_test, labels_test) if len(set(labels_test)) > 1 else 0
    
    status = "OK" if silhouette >= PRODUCTION_THRESHOLDS["clustering"]["silhouette"] else "ALERTE"
    
    save_metric_to_db(
        model_type="clustering",
        metric_name="silhouette",
        metric_value=silhouette,
        threshold=PRODUCTION_THRESHOLDS["clustering"]["silhouette"],
        status=status,
        details=f"n_clusters={len(set(labels_test))}"
    )
    
    print(f"\n📈 Clustering:")
    print(f"   Silhouette: {silhouette:.4f} (seuil: {PRODUCTION_THRESHOLDS['clustering']['silhouette']})")
    print(f"   Nb clusters: {len(set(labels_test))}")
    print(f"   Statut: {'✅ OK' if status == 'OK' else '❌ ALERTE'}")

# ============================================
# 6. DÉTECTION DE DÉRIVE
# ============================================
print("\n" + "="*60)
print("📊 DÉTECTION DE DÉRIVE (Data Drift)")
print("="*60)

from scipy import stats

np.random.seed(42)
n_train = 500
df_train_ref = pd.DataFrame({
    "Montant_HT": np.random.exponential(500, n_train) + 10,
    "Taux_TVA": np.random.random(n_train)
})

drift_detected = False
for col in ["Montant_HT", "Taux_TVA"]:
    ks_stat, p_value = stats.ks_2samp(df_train_ref[col], df_test[col])
    if p_value < 0.05:
        print(f"   ⚠️ Dérive détectée sur {col}: p-value={p_value:.4f}")
        drift_detected = True
    else:
        print(f"   ✅ {col}: distribution stable (p-value={p_value:.4f})")

if drift_detected:
    save_metric_to_db(
        model_type="system",
        metric_name="data_drift",
        metric_value=1,
        threshold=0,
        status="ALERTE",
        details="distribution_changed"
    )

# ============================================
# 7. CONSULTATION DE L'HISTORIQUE SQLite
# ============================================
print("\n" + "="*60)
print("📊 HISTORIQUE DES MONITORING (SQLite)")
print("="*60)

conn = sqlite3.connect(MONITORING_DB)
cursor = conn.cursor()
cursor.execute("SELECT * FROM monitoring ORDER BY id DESC LIMIT 10")
rows = cursor.fetchall()
conn.close()

print("\nDernières entrées dans la base:")
print("-" * 80)
for row in rows:
    print(f"   {row[1]} | {row[2]} | {row[3]}={row[4]:.4f} | {row[6]}")

# ============================================
# 8. RAPPORT FINAL
# ============================================
print("\n" + "="*60)
print("📈 RAPPORT DAY 6 - MONITORING COMPLET")
print("="*60)
print(f"\n🔹 Base SQLite: {MONITORING_DB}")
print(f"🔹 Modèles en Production: OK")
print(f"🔹 Data Drift: {'⚠️ Détecté' if drift_detected else '✅ Aucun'}")
print(f"\n✅ Monitoring actif - Les métriques sont sauvegardées dans SQLite")
print("\n🌐 Interface MLflow: http://127.0.0.1:5000")
print("🔧 Pour tester l'API, démarrez le serveur:")
print("   mlflow models serve --model-uri models:/SOUGUI_Classifier_Optuna/Production --host 127.0.0.1 --port 5001")