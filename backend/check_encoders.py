import joblib
import pickle
import numpy as np
from pathlib import Path

print("=" * 60)
print("INSPECTION DES MODÈLES ET ENCODEURS")
print("=" * 60)

base_path_purchase = Path("models/decideur_purchase")
base_path_commercial = Path("models/decideur_commercial_b2c")
base_path_marketing = Path("models/decideur_marketing")
base_path_gm = Path("models/decideur_GM")

# ========== DECIDEUR PURCHASE ==========
print("\n" + "=" * 60)
print("DÉCIDEUR PURCHASE")
print("=" * 60)

# 1. Classification
print("\n📂 Classification")
print("-" * 40)
try:
    model_clf = joblib.load(base_path_purchase / "classification/modele.joblib")
    if hasattr(model_clf, "feature_names_in_"):
        print("Features attendues :", list(model_clf.feature_names_in_))
    else:
        print("Nombre de features :", model_clf.n_features_in_)
    
    cat_enc = joblib.load(base_path_purchase / "classification/encodeur_categorie.joblib")
    four_enc = joblib.load(base_path_purchase / "classification/encodeur_fournisseur.joblib")
    print("Catégories disponibles :", list(cat_enc.classes_))
    print("Fournisseurs disponibles :", list(four_enc.classes_))
except Exception as e:
    print(f"Erreur : {e}")

# 2. Régression
print("\n📂 Régression")
print("-" * 40)
try:
    model_reg = joblib.load(base_path_purchase / "regression/modele.joblib")
    if hasattr(model_reg, "feature_names_in_"):
        print("Features attendues :", list(model_reg.feature_names_in_))
    else:
        print("Nombre de features :", model_reg.n_features_in_)
    
    cat_enc = joblib.load(base_path_purchase / "regression/encodeur_categorie.joblib")
    four_enc = joblib.load(base_path_purchase / "regression/encodeur_fournisseur.joblib")
    meth_enc = joblib.load(base_path_purchase / "regression/encodeur_methode.joblib")
    print("Catégories disponibles :", list(cat_enc.classes_))
    print("Fournisseurs disponibles :", list(four_enc.classes_))
    print("Méthodes de paiement disponibles :", list(meth_enc.classes_))
except Exception as e:
    print(f"Erreur : {e}")

# 3. Clustering (scaler)
print("\n📂 Clustering (scaler)")
print("-" * 40)
try:
    scaler = joblib.load(base_path_purchase / "clustering/scaler.joblib")
    if hasattr(scaler, "feature_names_in_"):
        print("Features attendues :", list(scaler.feature_names_in_))
    else:
        print("Nombre de features :", scaler.n_features_in_)
except Exception as e:
    print(f"Erreur : {e}")

# 4. Modèle KMeans
print("\n📂 Clustering (modèle KMeans)")
print("-" * 40)
try:
    kmeans = joblib.load(base_path_purchase / "clustering/modele.joblib")
    if hasattr(kmeans, "feature_names_in_"):
        print("Features attendues :", list(kmeans.feature_names_in_))
    else:
        print("Nombre de features :", kmeans.n_features_in_)
except Exception as e:
    print(f"Erreur : {e}")

# ========== DECIDEUR COMMERCIAL B2C ==========
print("\n" + "=" * 60)
print("DÉCIDEUR COMMERCIAL B2C")
print("=" * 60)

# 5. Anomalie Detection
print("\n📂 Anomalie Detection")
print("-" * 40)
anomaly_path = base_path_commercial / "regression/anomalie"
try:
    scaler_anom = joblib.load(anomaly_path / "scaler.pkl")
    print("✅ Scaler chargé")
    if hasattr(scaler_anom, "feature_names_in_"):
        print("Features attendues par le scaler :", list(scaler_anom.feature_names_in_))
    else:
        print("Nombre de features du scaler :", scaler_anom.n_features_in_)
    
    model_anom = joblib.load(anomaly_path / "iso_forest.pkl")
    print("✅ Modèle Isolation Forest chargé")
    if hasattr(model_anom, "feature_names_in_"):
        print("Features attendues par le modèle :", list(model_anom.feature_names_in_))
    else:
        print("Nombre de features du modèle :", model_anom.n_features_in_)
except Exception as e:
    print(f"Erreur : {e}")

# 6. Régression (si présente)
reg_path = base_path_commercial / "regression"
if (reg_path / "best_model.pkl").exists():
    print("\n📂 Régression")
    print("-" * 40)
    try:
        model_reg_com = joblib.load(reg_path / "best_model.pkl")
        if hasattr(model_reg_com, "feature_names_in_"):
            print("Features attendues :", list(model_reg_com.feature_names_in_))
        else:
            print("Nombre de features :", model_reg_com.n_features_in_)
    except Exception as e:
        print(f"Erreur : {e}")

# 7. Classification (si présente)
classif_path = base_path_commercial / "classification"
if (classif_path / "best_model.pkl").exists():
    print("\n📂 Classification")
    print("-" * 40)
    try:
        model_clf_com = joblib.load(classif_path / "best_model.pkl")
        if hasattr(model_clf_com, "feature_names_in_"):
            print("Features attendues :", list(model_clf_com.feature_names_in_))
        else:
            print("Nombre de features :", model_clf_com.n_features_in_)
    except Exception as e:
        print(f"Erreur : {e}")

# ========== DECIDEUR MARKETING ==========
print("\n" + "=" * 60)
print("DÉCIDEUR MARKETING")
print("=" * 60)

# 1. Clustering marketing
print("\n📂 Clustering")
print("-" * 40)
try:
    scaler_mkt = joblib.load(base_path_marketing / "clustering/scaler.joblib")
    print("✅ Scaler chargé")
    if hasattr(scaler_mkt, "feature_names_in_"):
        print("Features attendues par le scaler :", list(scaler_mkt.feature_names_in_))
    else:
        print("Nombre de features du scaler :", scaler_mkt.n_features_in_)

    model_kmeans_mkt = joblib.load(base_path_marketing / "clustering/modele.joblib")
    print("✅ Modèle KMeans chargé")
    if hasattr(model_kmeans_mkt, "feature_names_in_"):
        print("Features attendues par le modèle :", list(model_kmeans_mkt.feature_names_in_))
    else:
        print("Nombre de features du modèle :", model_kmeans_mkt.n_features_in_)

    feat_names = joblib.load(base_path_marketing / "clustering/feature_names.joblib")
    print("Noms des features (feature_names.joblib) :", list(feat_names))
except Exception as e:
    print(f"Erreur : {e}")

# 2. Timeseries marketing (SARIMA)
print("\n📂 Timeseries")
print("-" * 40)
try:
    with open(base_path_marketing / "timeseries/sarima_model.pkl", "rb") as f:
        sarima = pickle.load(f)
    print("✅ Modèle SARIMA chargé")
    print(f"Type du modèle : {type(sarima)}")
    
    config = joblib.load(base_path_marketing / "timeseries/sarima_config.joblib")
    print("Configuration SARIMA :", config)
except Exception as e:
    print(f"Erreur : {e}")

# 3. Régression marketing
print("\n📂 Régression")
print("-" * 40)
try:
    model_reg_mkt = joblib.load(base_path_marketing / "regression/modele.joblib")
    print("✅ Modèle de régression chargé")
    if hasattr(model_reg_mkt, "feature_names_in_"):
        print("Features attendues :", list(model_reg_mkt.feature_names_in_))
    else:
        print("Nombre de features :", model_reg_mkt.n_features_in_)
    
    if hasattr(model_reg_mkt, "named_steps"):
        print("Pipeline steps :", list(model_reg_mkt.named_steps.keys()))
    
    if (base_path_marketing / "regression/scaler.joblib").exists():
        scaler_reg = joblib.load(base_path_marketing / "regression/scaler.joblib")
        print("✅ Scaler présent pour la régression")
        if hasattr(scaler_reg, "feature_names_in_"):
            print("  Features scaler :", list(scaler_reg.feature_names_in_))
    else:
        print("⚠️ Aucun scaler séparé trouvé pour la régression (peut être dans pipeline)")
except Exception as e:
    print(f"Erreur : {e}")

# 4. Classification marketing
print("\n📂 Classification")
print("-" * 40)
try:
    model_clf_mkt = joblib.load(base_path_marketing / "classification/modele.joblib")
    print("✅ Modèle de classification chargé")
    if hasattr(model_clf_mkt, "feature_names_in_"):
        print("Features attendues :", list(model_clf_mkt.feature_names_in_))
    else:
        print("Nombre de features :", model_clf_mkt.n_features_in_)
    
    if hasattr(model_clf_mkt, "named_steps"):
        print("Pipeline steps :", list(model_clf_mkt.named_steps.keys()))
    
    if (base_path_marketing / "classification/scaler.joblib").exists():
        scaler_clf = joblib.load(base_path_marketing / "classification/scaler.joblib")
        print("✅ Scaler présent pour la classification")
    else:
        print("⚠️ Aucun scaler séparé trouvé pour la classification")
except Exception as e:
    print(f"Erreur : {e}")

# ========== DECIDEUR GM ==========
print("\n" + "=" * 60)
print("DÉCIDEUR GM")
print("=" * 60)

# Helper pour extraire modèle et scaler d'un dict
def extract_model_and_scaler(obj):
    if isinstance(obj, dict):
        print("  (Objet dictionnaire détecté)")
        model = obj.get('model') or obj.get('regressor') or obj.get('classifier') or obj.get('estimator')
        scaler = obj.get('scaler') or obj.get('preprocessor')
        if model is None:
            # Si pas de clé standard, prendre la première valeur non métadata
            for k, v in obj.items():
                if k not in ['metadata', 'config', 'feature_names'] and hasattr(v, 'predict'):
                    model = v
                    break
        return model, scaler
    else:
        return obj, None

# 1. Classification
print("\n📂 Classification")
print("-" * 40)
try:
    loaded = joblib.load(base_path_gm / "classification/modele_classification.joblib")
    model_clf_gm, scaler_clf_gm = extract_model_and_scaler(loaded)
    if model_clf_gm is None:
        print("❌ Impossible d'extraire un modèle de classification")
    else:
        print("✅ Modèle de classification extrait")
        if hasattr(model_clf_gm, "feature_names_in_"):
            print("Features attendues :", list(model_clf_gm.feature_names_in_))
        elif hasattr(model_clf_gm, "n_features_in_"):
            print("Nombre de features :", model_clf_gm.n_features_in_)
        if hasattr(model_clf_gm, "named_steps"):
            print("Pipeline steps :", list(model_clf_gm.named_steps.keys()))
        if scaler_clf_gm is not None:
            print("✅ Scaler/Preprocessor trouvé dans le dictionnaire")
        else:
            print("⚠️ Aucun scaler/preprocessor trouvé")
except Exception as e:
    print(f"Erreur : {e}")

# 2. Régression
print("\n📂 Régression")
print("-" * 40)
try:
    loaded = joblib.load(base_path_gm / "regression/modele_regression.joblib")
    model_reg_gm, scaler_reg_gm = extract_model_and_scaler(loaded)
    if model_reg_gm is None:
        print("❌ Impossible d'extraire un modèle de régression")
    else:
        print("✅ Modèle de régression extrait")
        if hasattr(model_reg_gm, "feature_names_in_"):
            print("Features attendues :", list(model_reg_gm.feature_names_in_))
        elif hasattr(model_reg_gm, "n_features_in_"):
            print("Nombre de features :", model_reg_gm.n_features_in_)
        if hasattr(model_reg_gm, "named_steps"):
            print("Pipeline steps :", list(model_reg_gm.named_steps.keys()))
        if scaler_reg_gm is not None:
            print("✅ Scaler/Preprocessor trouvé dans le dictionnaire")
        else:
            print("⚠️ Aucun scaler/preprocessor trouvé")
except Exception as e:
    print(f"Erreur : {e}")

# 3. Clustering
print("\n📂 Clustering")
print("-" * 40)
try:
    loaded = joblib.load(base_path_gm / "clustering/modele_clustering.joblib")
    model_clust_gm, scaler_clust_gm = extract_model_and_scaler(loaded)
    if model_clust_gm is None:
        print("❌ Impossible d'extraire un modèle de clustering")
    else:
        print("✅ Modèle de clustering extrait")
        if hasattr(model_clust_gm, "feature_names_in_"):
            print("Features attendues :", list(model_clust_gm.feature_names_in_))
        elif hasattr(model_clust_gm, "n_features_in_"):
            print("Nombre de features :", model_clust_gm.n_features_in_)
        if scaler_clust_gm is not None:
            print("✅ Scaler trouvé dans le dictionnaire")
        else:
            print("❌ Aucun scaler trouvé – risque d'erreur")
except Exception as e:
    print(f"Erreur : {e}")

# 4. Anomalie
print("\n📂 Anomalie")
print("-" * 40)
try:
    loaded = joblib.load(base_path_gm / "anomalie/modele_anomalie.joblib")
    model_anom_gm, scaler_anom_gm = extract_model_and_scaler(loaded)
    if model_anom_gm is None:
        print("❌ Impossible d'extraire un modèle d'anomalie")
    else:
        print("✅ Modèle d'anomalie extrait")
        if hasattr(model_anom_gm, "feature_names_in_"):
            print("Features attendues :", list(model_anom_gm.feature_names_in_))
        elif hasattr(model_anom_gm, "n_features_in_"):
            print("Nombre de features :", model_anom_gm.n_features_in_)
        if scaler_anom_gm is not None:
            print("✅ Scaler trouvé dans le dictionnaire")
        else:
            print("❌ Aucun scaler trouvé – risque d'erreur")
except Exception as e:
    print(f"Erreur : {e}")

# 5. Recommandation
print("\n📂 Recommandation")
print("-" * 40)
try:
    loaded = joblib.load(base_path_gm / "recommandation/modele_recommandation.joblib")
    model_rec_gm, _ = extract_model_and_scaler(loaded)
    if model_rec_gm is None:
        print("❌ Impossible d'extraire un modèle de recommandation")
    else:
        print("✅ Modèle de recommandation extrait")
        print(f"Type : {type(model_rec_gm).__name__}")
        if hasattr(model_rec_gm, "n_users") or hasattr(model_rec_gm, "n_items"):
            print("  Attributs : n_users / n_items présents")
        if hasattr(model_rec_gm, "user_mapping") or hasattr(model_rec_gm, "item_mapping"):
            print("  Mappings utilisateur/produit trouvés")
        else:
            print("⚠️ Aucun mapping utilisateur/produit détecté")
    
    # Vérifier fichiers de mapping séparés
    if (base_path_gm / "recommandation/user_mapping.joblib").exists():
        print("✅ user_mapping.joblib trouvé")
    else:
        print("❌ user_mapping.joblib manquant")
    if (base_path_gm / "recommandation/item_mapping.joblib").exists():
        print("✅ item_mapping.joblib trouvé")
    else:
        print("❌ item_mapping.joblib manquant")
except Exception as e:
    print(f"Erreur : {e}")

print("\n" + "=" * 60)
print("✅ Inspection terminée")