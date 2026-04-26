Voici un fichier **README.md** complet pour votre projet MLOps **Sougui.tn – Purchasing**.

---

## 📄 Fichier `README.md`

```markdown
# 🛒 Sougui.tn – Purchasing MLOps Pipeline

**Version**: 1.0.0  
**Dernière mise à jour**: Avril 2025

---

## 📋 Description

Ce projet implémente une **chaîne MLOps complète** pour le département **Purchasing (Achats)** de Sougui.tn.

L'objectif est de :
- Automatiser le suivi des modèles ML (Classification, Régression, Clustering, Forecast)
- Garantir la reproductibilité et la traçabilité
- Mettre en place un cycle de vie des modèles (Staging → Production)
- Optimiser les performances avec Optuna
- Monitorer les modèles en production

---

## 🎯 Les 4 objectifs métier

| Objectif | Modèle | Métrique cible | Business value |
|----------|--------|----------------|----------------|
| **Classification** | RandomForest / XGBoost | Accuracy ≥ 0.95 | Distinguer factures standard vs exonérées |
| **Régression** | RandomForest / XGBoost | RMSE ≤ 200, R² ≥ 0.80 | Prédire le montant HT des achats |
| **Clustering** | KMeans | Silhouette ≥ 0.40 | Segmenter les fournisseurs |
| **Forecast** | ARIMA | MAPE ≤ 20% | Prévoir les achats mensuels |

---

## 🏗️ Architecture du projet

```
sougui_purchasing_mlops/
│
├── pipelines/                    # Scripts jour par jour
│   ├── day1_tracking_base.py
│   ├── day2_multi_metrics.py
│   ├── day3_model_registry.py
│   ├── day4_cicd_pipeline.py
│   ├── day5_optuna_optimization.py
│   └── day6_deployment_monitoring.py
│
├── models/                       # Stockage des modèles
├── mlflow_server/                # Base SQLite MLflow
├── monitoring.db                 # Base SQLite pour monitoring
├── requirements.txt              # Dépendances
└── README.md
```

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
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Démarrer les serveurs MLflow

**Terminal 1 - Serveur de tracking**
```bash
mlflow server --backend-store-uri sqlite:///mlflow_server/mlflow.db --host 127.0.0.1 --port 5000
```

**Terminal 2 - Serveur de déploiement (optionnel)**
```bash
mlflow models serve --model-uri models:/SOUGUI_Classifier_Optuna/Production --host 127.0.0.1 --port 5001
```

---

## 📊 Exécution des pipelines

### Day 1 – Tracking de base
```bash
python pipelines/day1_tracking_base.py
```
✅ 4 runs enregistrés (Classification, Régression, Clustering, Forecast)

### Day 2 – Multi-métriques + comparaison
```bash
python pipelines/day2_multi_metrics.py
```
✅ 9 runs, artefacts visuels (matrices confusion, courbes ROC)

### Day 3 – Model Registry
```bash
python pipelines/day3_model_registry.py
```
✅ Modèles versionnés en Staging

### Day 4 – CI/CD Pipeline
```bash
python pipelines/day4_cicd_pipeline.py
```
✅ Promotion automatique Staging → Production

### Day 5 – Optimisation Optuna
```bash
python pipelines/day5_optuna_optimization.py
```
✅ Recherche automatique des meilleurs hyperparamètres

### Day 6 – Monitoring
```bash
python pipelines/day6_deployment_monitoring.py
```
✅ Sauvegarde des métriques dans SQLite, détection de dérive

---

## 📈 Résultats des optimisations

| Métrique | Avant Optuna | Après Optuna | Amélioration |
|----------|--------------|--------------|--------------|
| Régression - RMSE | 373.81 | **169.28** | **-55%** |
| Régression - R² | 0.27 | **0.8512** | **+215%** |
| Classification - Accuracy | 1.0000 | 1.0000 | ✅ Parfait |
| Clustering - Silhouette | 0.4627 | 0.4627 | Stable |

---

## 🗄️ Bases de données

| Base | Rôle | Localisation |
|------|------|--------------|
| `mlflow_server/mlflow.db` | Tracking des runs, métriques, paramètres | SQLite |
| `monitoring.db` | Métriques de production, alertes | SQLite |
| `dwh_sougui` (MySQL) | Données source achats | XAMPP |

---

## 🔧 Configuration MySQL

```python
DB_USER = "root"
DB_PASSWORD = ""
DB_HOST = "127.0.0.1"
DB_PORT = 3306
DB_NAME = "dwh_sougui"
```

> 💡 Si MySQL n'est pas disponible, le pipeline utilise automatiquement des **données synthétiques**.

---

## 🌐 Accès MLflow UI

| Interface | URL |
|-----------|-----|
| Tracking UI | http://127.0.0.1:5000 |
| Models API | http://127.0.0.1:5001 |

---

## 📊 Modèles enregistrés

| Modèle | Stage | Métrique |
|--------|-------|----------|
| SOUGUI_Classifier_Optuna | Production | Accuracy = 1.0000 |
| SOUGUI_Regressor_Optuna | Production | RMSE = 169.28, R² = 0.8512 |
| SOUGUI_Classifier_RF | Staging | Accuracy = 1.0000 |
| SOUGUI_Regressor_RF | Staging | RMSE = 373.81, R² = 0.27 |

---

## 📁 Structure des logs (SQLite - monitoring.db)

```sql
CREATE TABLE monitoring (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    model_type TEXT,
    metric_name TEXT,
    metric_value REAL,
    threshold REAL,
    status TEXT,
    latency_ms REAL,
    details TEXT
);
```

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

## 👥 Équipe

**Responsable MLOps** : [Votre Nom]  
**Décideur Purchasing** : [Nom du décideur]

---

## 📅 Prochaines étapes

- [ ] Activer MySQL pour charger les données réelles
- [ ] Ré-entraîner les modèles sur les vraies données
- [ ] Configurer les alertes Slack
- [ ] Mettre en place un scheduler (Airflow / cron)
- [ ] Ajouter des tests unitaires
- [ ] Déployer l'API MLflow Models en production

---

## 📞 Support

Pour toute question :  
- MLflow UI : http://127.0.0.1:5000
- Base monitoring : `monitoring.db`

---

**© 2025 Sougui.tn – Tous droits réservés**
```

---

## ✅ À faire maintenant

1. **Copiez ce contenu** dans un fichier `README.md` à la racine de votre projet
2. **Ajoutez votre nom** dans la section Équipe
3. **Committez le fichier** :
```bash
git add README.md
git commit -m "Add README - MLOps Purchasing pipeline"
```

---

**Le projet est maintenant complet et documenté !** 🎉