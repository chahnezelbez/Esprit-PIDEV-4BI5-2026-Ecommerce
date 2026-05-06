"""
DAY 2 - MLOps pour Sougui.tn (Général Manager)
Multi-métriques + Comparaison de modèles + Artefacts visuels
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import hashlib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,
    silhouette_score, davies_bouldin_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
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

data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
mlflow.set_experiment("SOUGUI_GM_Day2")

# ============================================
# 2. CLASSIFICATION - Comparaison de modèles
# ============================================
print("\n" + "="*50)
print("📊 CLASSIFICATION - Comparaison de modèles")
print("="*50)

features_clf = ["Recency", "Customer_Age_Days", "Pct_Weekend", "Frequency", 
                "Nb_Categories", "Monetary", "Avg_Order_Value"]
X_clf = df[features_clf].fillna(0)
y_clf = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

models_clf = {
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)
}

for name, model in models_clf.items():
    with mlflow.start_run(run_name=f"Classification_{name}"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_proba)) > 1 else 0.5
        }
        
        mlflow.log_params({"model_type": name, "dataset_hash": data_hash})
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig("cm.png")
        mlflow.log_artifact("cm.png")
        plt.close()
        
        # Courbe ROC
        if len(np.unique(y_proba)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(figsize=(8,6))
            plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
            plt.plot([0,1], [0,1], "k--")
            plt.title(f"ROC Curve - {name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig("roc.png")
            mlflow.log_artifact("roc.png")
            plt.close()
        
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        print(f"  ✅ {name} - Accuracy={metrics['accuracy']:.4f}, AUC={metrics['roc_auc']:.4f}")

# ============================================
# 3. RÉGRESSION - Comparaison de modèles
# ============================================
print("\n" + "="*50)
print("📊 RÉGRESSION - Comparaison de modèles")
print("="*50)

features_reg = ["Recency", "Frequency", "Monetary", "Customer_Age_Days", 
                "Pct_Weekend", "Nb_Categories", "Is_Online_Buyer"]
X_reg = df[features_reg].fillna(0)
y_reg = df["Avg_Order_Value"]
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
        
        mlflow.log_params({"model_type": name, "dataset_hash": data_hash})
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        
        # Graphique des résidus
        residuals = y_test_r - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(12,4))
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_title(f"Residuals - {name}")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residuals")
        axes[1].hist(residuals, bins=30, edgecolor="black")
        axes[1].set_title(f"Residuals Distribution - {name}")
        plt.tight_layout()
        plt.savefig("residuals.png")
        mlflow.log_artifact("residuals.png")
        plt.close()
        
        signature = mlflow.models.infer_signature(X_train_r, model.predict(X_train_r))
        mlflow.sklearn.log_model(model, "model", signature=signature)
        print(f"  ✅ {name} - RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.4f}")

# ============================================
# 4. CLUSTERING - Optimisation du nombre de clusters
# ============================================
print("\n" + "="*50)
print("📊 CLUSTERING - Optimisation du nombre de clusters")
print("="*50)

X_clust = df[["Monetary", "Frequency", "Recency"]].values
scaler = StandardScaler()
X_clust_scaled = scaler.fit_transform(X_clust)

for k in [2, 3, 4, 5, 6]:
    with mlflow.start_run(run_name=f"Clustering_KMeans_k{k}"):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_clust_scaled)
        
        metrics = {
            "silhouette": silhouette_score(X_clust_scaled, labels) if len(set(labels)) > 1 else 0,
            "davies_bouldin": davies_bouldin_score(X_clust_scaled, labels) if len(set(labels)) > 1 else 0,
            "inertia": model.inertia_
        }
        
        mlflow.log_params({"n_clusters": k, "dataset_hash": data_hash})
        for k_metric, v in metrics.items():
            mlflow.log_metric(k_metric, v)
        
        # Visualisation PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_clust_scaled)
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="viridis", s=50)
        plt.title(f"Clusters (PCA) - k={k}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.savefig(f"clusters_k{k}.png")
        mlflow.log_artifact(f"clusters_k{k}.png")
        plt.close()
        
        print(f"  ✅ k={k} - Silhouette={metrics['silhouette']:.4f}")

print("\n✅ Day 2 terminé. Tous les modèles comparés et artefacts sauvegardés.")