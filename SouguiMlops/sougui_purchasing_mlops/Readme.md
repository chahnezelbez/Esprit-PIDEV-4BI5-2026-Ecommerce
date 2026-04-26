<div align="center">

# 🛒 Sougui.tn – Purchasing MLOps Pipeline

### Plateforme Artisanale Tunisienne | Département Achats

![Version](https://img.shields.io/badge/version-1.0.0-c2793a?style=flat-square)
![MLflow](https://img.shields.io/badge/MLflow-2.0+-blue?style=flat-square)
![Optuna](https://img.shields.io/badge/Optuna-3.0+-purple?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-yellow?style=flat-square)
![Status](https://img.shields.io/badge/status-Production-green?style=flat-square)

**Version** : 1.0.0 · **Dernière mise à jour** : Avril 2025

</div>

---

## 📋 Description

Ce projet implémente une **chaîne MLOps complète** pour le département **Purchasing (Achats)** de Sougui.tn.

> 🎯 **Vision** : Passer d'expériences ML ad hoc à un pipeline industriel, traçable, reproductible et déployable en production.

L'objectif est de :
- Automatiser le suivi des modèles ML (Classification, Régression, Clustering, Forecast)
- Garantir la reproductibilité et la traçabilité de chaque expérience
- Mettre en place un cycle de vie complet des modèles (Staging → Production)
- Optimiser les performances automatiquement avec **Optuna**
- Monitorer les modèles en production via une base SQLite dédiée

---

## 🎯 Les 4 objectifs métier

| # | Objectif | Modèle | Métrique cible | Business value |
|---|----------|--------|----------------|----------------|
| 1 | **Classification** | RandomForest / XGBoost | Accuracy ≥ 0.95 | Distinguer factures standard vs exonérées |
| 2 | **Régression** | RandomForest / XGBoost | RMSE ≤ 200, R² ≥ 0.80 | Prédire le montant HT des achats |
| 3 | **Clustering** | KMeans | Silhouette ≥ 0.40 | Segmenter les fournisseurs par profil |
| 4 | **Forecast** | ARIMA | MAPE ≤ 20% | Prévoir les volumes d'achats mensuels |

---

## 🏗️ Architecture du projet

```
sougui_purchasing_mlops/
│
├── pipelines/                         # Scripts jour par jour
│   ├── day1_tracking_base.py          # Experiment tracking basique
│   ├── day2_multi_metrics.py          # Multi-métriques + artefacts visuels
│   ├── day3_model_registry.py         # Versioning & Model Registry
│   ├── day4_cicd_pipeline.py          # CI/CD – promotion automatique
│   ├── day5_optuna_optimization.py    # Hyperparameter tuning
│   └── day6_deployment_monitoring.py  # Déploiement & monitoring
│
├── models/                            # Modèles sauvegardés localement
├── mlflow_server/
│   └── mlflow.db                      # Backend SQLite MLflow
├── requirements.txt
└── README.md
```

---

## 🔄 Architecture MLOps – Vue d'ensemble

```
┌──────────────────────────────────────────────────────────────────┐
│                   SOUGUI.TN – Purchasing MLOps                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🗄️  Source de données                                           │
│       MySQL (dwh_sougui) ──► Fallback : données synthétiques     │
│                    ↓                                             │
│  🔄  Pipeline automatisé                                         │
│       Preprocessing → Training → Evaluation                      │
│                    ↓                                             │
│  📈  MLflow Tracking                                             │
│       Params · Métriques · Artifacts · Registry                  │
│                    ↓                                             │
│  🧪  Optuna – Hyperparameter Tuning                              │
│       Recherche automatique · 30+ trials                         │
│                    ↓                                             │
│  🚀  Model Registry                                              │
│       Staging ──────────────────► Production                     │
│                    ↓                                             │
│                                                                  │
│                    ↓                                             │
│  🔮  API FastAPI / MLflow Models Serve                           │
│       http://127.0.0.1:5001                                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Days 1 à 6 – Détail technique

### 📘 Day 1 – Experiment Tracking basique

| Élément | Détail |
|---------|--------|
| **Problème** | Suivi manuel des expériences → non reproductible |
| **Solution** | MLflow log paramètres, métriques et artefacts |
| **Output** | ✅ 4 runs (Classification, Régression, Clustering, Forecast) |

---

### 📘 Day 2 – Multi-métriques + Artifacts

| Élément | Détail |
|---------|--------|
| **Problème** | Une seule métrique ne suffit pas |
| **Solution** | Log de 5 métriques + images PNG (confusion matrix, ROC, résidus) |
| **Output** | ✅ Artifacts visibles et comparables dans MLflow UI |

---

### 📘 Day 3 – Model Registry + Versions

| Élément | Détail |
|---------|--------|
| **Problème** | Difficile de gérer les versions et la mise en production |
| **Solution** | MLflow Registry avec versions (v1, v2, v3) et promotion |
| **Output** | ✅ 3 versions par modèle, v2 en Production |

---

### 📘 Day 4 – Persistance SQLite

| Élément | Détail |
|---------|--------|
| **Problème** | Les données MLflow ne persistent pas après redémarrage |
| **Solution** | Backend SQLite + serveur centralisé |
| **Output** | ✅ Base `mlflow.db` persistante (244 KB) |

---

### 📘 Day 5 – Optuna + Hyperparameter Tuning

| Élément | Détail |
|---------|--------|
| **Problème** | Optimisation manuelle inefficace |
| **Solution** | Optuna explore automatiquement l'espace des paramètres |
| **Output** | ✅ 30 runs automatiques, meilleurs paramètres identifiés |

---

### 📘 Day 6 – MLflow Model Format + Inference

| Élément | Détail |
|---------|--------|
| **Problème** | Format pickle non standardisé |
| **Solution** | Format MLflow + chargement depuis Registry + inférence |
| **Output** | ✅ Prédictions pour 5 nouveaux clients (churn, prix, segment, ventes) |

---

## 🖼️ Artefacts disponibles

| Run / Expérience | Artefacts générés |
|------------------|-------------------|
| `Classification_RandomForest` | `cm.png`, `roc.png` |
| `Classification_XGBoost` | `cm.png`, `roc.png` |
| `Regression_LinearRegression` | `residuals.png` |
| `Regression_RandomForest` | `residuals.png` |
| `Regression_XGBoost` | `residuals.png` |
| `Clustering_KMeans_k2` | `clusters_k2.png` |
| `Clustering_KMeans_k3` | `clusters_k3.png` |
| `Clustering_KMeans_k4` | `clusters_k4.png` |

---

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone <repository-url>
cd sougui_purchasing_mlops
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Démarrer les serveurs MLflow

**Terminal 1 – Serveur de tracking**
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow_server/mlflow.db \
  --default-artifact-root ./mlflow_server/artifacts \
  --host 127.0.0.1 \
  --port 5000
```

**Terminal 2 – Serveur de déploiement** *(optionnel)*
```bash
mlflow models serve \
  --model-uri models:/SOUGUI_Classifier_Optuna/Production \
  --host 127.0.0.1 \
  --port 5001
```

---

## 📊 Exécution des pipelines

### 📘 Day 1 – Tracking de base

```bash
python pipelines/day1_tracking_base.py
```

| Élément | Détail |
|---------|--------|
| **Objectif** | Poser les fondations du suivi d'expériences |
| **Output** | ✅ 4 runs enregistrés (Classification, Régression, Clustering, Forecast) |

---

### 📘 Day 2 – Multi-métriques + comparaison

```bash
python pipelines/day2_multi_metrics.py
```

| Élément | Détail |
|---------|--------|
| **Objectif** | Enrichir le tracking avec plusieurs métriques et visualisations |
| **Output** | ✅ 9 runs, artefacts visuels (matrices de confusion, courbes ROC) |

---

### 📘 Day 3 – Model Registry

```bash
python pipelines/day3_model_registry.py
```

| Élément | Détail |
|---------|--------|
| **Objectif** | Versionner et gouverner les modèles |
| **Output** | ✅ Modèles enregistrés en Staging avec versions v1 / v2 / v3 |

---

### 📘 Day 4 – CI/CD Pipeline

```bash
python pipelines/day4_cicd_pipeline.py
```

| Élément | Détail |
|---------|--------|
| **Objectif** | Automatiser la promotion des modèles validés |
| **Output** | ✅ Promotion automatique Staging → Production selon seuils métriques |

---

### 📘 Day 5 – Optimisation Optuna

```bash
python pipelines/day5_optuna_optimization.py
```

| Élément | Détail |
|---------|--------|
| **Objectif** | Trouver automatiquement les meilleurs hyperparamètres |
| **Output** | ✅ 30+ trials · Meilleurs paramètres identifiés et loggués |

---

### 📘 Day 6 – Déploiement & Monitoring

```bash
python pipelines/day6_deployment_monitoring.py
```

| Élément | Détail |
|---------|--------|
| **Objectif** | Surveiller les modèles en production |
| **Output** | ✅ Métriques dans `monitoring.db` · Alertes drift activées |

---

## 📈 Résultats des optimisations Optuna

| Modèle | Métrique | Avant Optuna | Après Optuna | Amélioration |
|--------|----------|:------------:|:------------:|:------------:|
| Régression | RMSE | 373.81 | **169.28** | 🟢 **−55%** |
| Régression | R² | 0.27 | **0.8512** | 🟢 **+215%** |
| Classification | Accuracy | 1.0000 | **1.0000** | ✅ Parfait |
| Clustering | Silhouette | 0.4627 | **0.4627** | ➡️ Stable |

---

## 📦 Modèles enregistrés en Registry

| Modèle | Stage | Performance |
|--------|-------|-------------|
| `SOUGUI_Classifier_Optuna` | **Production** | Accuracy = 1.0000 |
| `SOUGUI_Regressor_Optuna` | **Production** | RMSE = 169.28 · R² = 0.8512 |
| `SOUGUI_Classifier_RF` | Staging | Accuracy = 1.0000 |
| `SOUGUI_Regressor_RF` | Staging | RMSE = 373.81 · R² = 0.27 |

---

## 🗄️ Bases de données

| Base | Rôle | Type |
|------|------|------|
| `mlflow_server/mlflow.db` | Runs, métriques, paramètres, artifacts | SQLite |
| `dwh_sougui` | Données source Achats | MySQL (XAMPP) |

### Schéma – table monitoring




---

## 🔧 Configuration MySQL

```python
DB_USER     = "root"
DB_PASSWORD = ""
DB_HOST     = "127.0.0.1"
DB_PORT     = 3306
DB_NAME     = "dwh_sougui"
```

> 💡 Si MySQL n'est pas disponible, le pipeline bascule automatiquement sur des **données synthétiques** pour garantir la continuité des expériences.

---

## 🌐 Interfaces & accès

| Interface | URL | Description |
|-----------|-----|-------------|
| MLflow Tracking UI | http://127.0.0.1:5000 | Suivi des runs et registry |
| MLflow Models API | http://127.0.0.1:5001 | Endpoint d'inférence |

---

## 🛠️ Dépendances principales

```
mlflow>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
optuna>=3.0.0
statsmodels>=0.14.0
pymysql>=1.0.0
sqlalchemy>=2.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
```

---

## 📞 Support & ressources

- **MLflow UI** : http://127.0.0.1:5000
- **Logs pipelines** : dossier `pipelines/`

---

<div align="center">

**© 2025 Sougui.tn – Tous droits réservés**
Projet réalisé dans le cadre du cours **S12 – MLOps** · Département **Purchasing**

</div>
