# =====================================================
# IMPORTS
# =====================================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from typing import List, Dict, Any
import time
from monitoring import setup_monitoring, drift_detector, PREDICTION_TIME, MODEL_PREDICTIONS, BASELINES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# CRÉER APP
# =====================================================
app = FastAPI(
    title="Sougui API - Décideurs Multi-Tâches",
    description="API pour les modèles ML multi-décideurs (Purchase, Commercial B2C, Marketing, GM, B2B, Financier)",
    version="3.5.0"
)
app = setup_monitoring(app)

# =====================================================
# CORS MIDDLEWARE
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:4200",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:4200",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ------------------------------------------------------------------
# Chemins des modèles
# ------------------------------------------------------------------
BASE_MODEL_PATH = Path("models")

PURCHASE_PATH        = BASE_MODEL_PATH / "decideur_purchase"
CLASSIF_PURCHASE_PATH = PURCHASE_PATH / "classification"
REGR_PURCHASE_PATH    = PURCHASE_PATH / "regression"
CLUST_PURCHASE_PATH   = PURCHASE_PATH / "clustering"

COMMERCIAL_PATH        = BASE_MODEL_PATH / "decideur_commercial_b2c"
CLASSIF_COMMERCIAL_PATH = COMMERCIAL_PATH / "classification"
REGR_COMMERCIAL_PATH    = COMMERCIAL_PATH / "regression"
FORECAST_COMMERCIAL_PATH = REGR_COMMERCIAL_PATH / "forecast"
ANOMALY_COMMERCIAL_PATH  = REGR_COMMERCIAL_PATH / "anomalie"

MARKETING_PATH          = BASE_MODEL_PATH / "decideur_marketing"
CLUST_MARKETING_PATH    = MARKETING_PATH / "clustering"
TIMESERIES_MARKETING_PATH = MARKETING_PATH / "timeseries"
REGR_MARKETING_PATH     = MARKETING_PATH / "regression"
CLASSIF_MARKETING_PATH  = MARKETING_PATH / "classification"

GM_PATH         = BASE_MODEL_PATH / "decideur_GM"
CLASSIF_GM_PATH = GM_PATH / "classification"
REGR_GM_PATH    = GM_PATH / "regression"
CLUST_GM_PATH   = GM_PATH / "clustering"
ANOMALY_GM_PATH = GM_PATH / "anomalie"

B2B_PATH        = BASE_MODEL_PATH / "decideur_B2B"
ANOMALY_B2B_PATH = B2B_PATH / "anomaly"
CLASSIF_B2B_PATH = B2B_PATH / "classification"
RISKS_B2B_PATH   = B2B_PATH / "classification_risks"
CLUST_B2B_PATH   = B2B_PATH / "clustering"
FORECAST_B2B_PATH = B2B_PATH / "forecast"
REGR_B2B_PATH    = B2B_PATH / "regression"

FIN_PATH         = BASE_MODEL_PATH / "decideur_fin"
CLUST_FIN_PATH   = FIN_PATH / "clustering"
FORECAST_FIN_PATH = FIN_PATH / "forcasting"

# ------------------------------------------------------------------
# SCHÉMAS - PURCHASE
# ------------------------------------------------------------------
class ClassificationFeaturesP(BaseModel):
    Montant_HT: float
    Taux_TVA: float
    Marge_TVA: float
    Mois: int
    Annee: int
    Semaine: int
    Est_weekend: int
    fournisseur: str
    categorie: str

class RegressionFeaturesP(BaseModel):
    Mois: int
    Annee: int
    Semaine: int
    Est_weekend: int
    fournisseur: str
    categorie: str
    methode: str
    Taux_TVA: float

class ClusteringFeaturesP(BaseModel):
    Nb_Factures: float
    Montant_Total: float
    Montant_Moyen: float
    Montant_Max: float
    TVA_Moy: float

# ------------------------------------------------------------------
# SCHÉMAS - COMMERCIAL B2C
# ------------------------------------------------------------------
class RegressionFeaturesC(BaseModel):
    feat_avg_price: float
    feat_max_price: float
    feat_free_shipping: float
    feat_shipping_pct: float
    feat_payment_encoded: float
    feat_is_tunis: float
    feat_nb_products: float
    feat_total_qty: float
    feat_has_promo: float
    feat_is_gift: float

class ClassificationFeaturesC(BaseModel):
    feat_is_peak_season: float
    feat_payment_encoded: float
    feat_shipping_pct: float
    feat_is_tunis: float
    feat_avg_price: float
    feat_max_price: float
    feat_has_note: float
    feat_has_promo: float
    feat_free_shipping: float
    feat_discount: float

class AnomalyDetectionFeaturesC(BaseModel):
    feat_nb_products: float
    feat_avg_price: float
    feat_max_price: float
    feat_discount: float
    feat_shipping_pct: float
    target_value: float

class ForecastInputC(BaseModel):
    periods: int = Field(default=30, ge=1, le=365)

# ------------------------------------------------------------------
# SCHÉMAS - MARKETING
# ------------------------------------------------------------------
class ClusteringFeaturesM(BaseModel):
    price_current: float
    discount_depth: float
    rating_value: float
    reviews_count: float
    sales_qty: float
    sales_revenue: float
    order_lines: float
    sales_velocity: float
    review_signal: float

class TimeseriesInputM(BaseModel):
    periods: int = Field(default=12, ge=1, le=36)

class RegressionFeaturesM(BaseModel):
    discount_depth: float
    rating_value: float
    reviews_count: float
    name_len: float
    desc_len: float
    sales_qty: float
    sales_revenue: float
    order_lines: float
    avg_qty: float
    avg_unit_price: float
    days_on_sale: float
    sales_velocity: float
    revenue_per_orderline: float
    review_signal: float
    broad_category: str
    stock_status: str
    main_payment: str
    main_delivery: str

class ClassificationFeaturesM(BaseModel):
    price_current: float
    discount_depth: float
    rating_value: float
    reviews_count: float
    name_len: float
    desc_len: float
    sales_qty: float
    sales_revenue: float
    order_lines: float
    avg_qty: float
    avg_unit_price: float
    days_on_sale: float
    sales_velocity: float
    revenue_per_orderline: float
    review_signal: float
    broad_category: str
    stock_status: str
    main_payment: str
    main_delivery: str

# ------------------------------------------------------------------
# SCHÉMAS - GM
# ------------------------------------------------------------------
class ClassificationFeaturesGM(BaseModel):
    Recency: float
    Customer_Age_Days: float
    Pct_Weekend: float
    Frequency: float
    Nb_Categories: float
    Monetary: float
    Avg_Price: float

class RegressionFeaturesGM(BaseModel):
    est_weekend: float
    mois: float
    trimestre: float
    quantite: float
    categorie_id_2: float
    categorie_id_3: float
    categorie_id_4: float
    categorie_id_5: float
    categorie_id_6: float
    categorie_id_7: float
    categorie_id_8: float
    categorie_id_9: float
    categorie_id_10: float
    categorie_id_11: float
    categorie_id_12: float
    canal_id_3: float
    canal_id_4: float
    gouvernorat_Ben_Arous: float
    gouvernorat_Bizerte: float
    gouvernorat_INCONNU: float
    gouvernorat_Monastir: float
    gouvernorat_Nabeul: float
    gouvernorat_Sfax: float
    gouvernorat_Sousse: float
    gouvernorat_Tunis: float

class ClusteringFeaturesGM(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    Avg_Order_Value: float
    Nb_Categories: float
    Pct_Weekend: float
    Is_Online_Buyer: float

class AnomalyFeaturesGM(BaseModel):
    prix_unitaire: float
    quantite: float
    montant_total: float
    mois: int
    est_weekend: int

# ------------------------------------------------------------------
# SCHÉMAS - B2B
# ------------------------------------------------------------------
class B2BFeaturesArray(BaseModel):
    features: List[float]

class B2BClassificationPipelineFeatures(BaseModel):
    montant_ht: float
    tva: float
    total_ttc: float
    ca_total_client: float
    nombre_de_marches_client: float
    annee: float
    mois_sin: float
    mois_cos: float
    segment_client: str
    client: str
    ecart_negatif_flag: float

# ------------------------------------------------------------------
# SCHÉMAS - FINANCIER
# ------------------------------------------------------------------
class ClusteringFeaturesFin(BaseModel):
    nb_commandes: float
    ca_total: float
    panier_moyen: float
    total_remise: float
    total_remboursement: float
    livraison_moyenne: float
    total_articles: float
    nb_categories: float
    nb_produits: float
    taux_finalisation: float
    recence: float

class ForecastInputFin(BaseModel):
    periods: int = Field(default=12, ge=1, le=36)

# ------------------------------------------------------------------
# CHARGEMENT DES MODÈLES
# ------------------------------------------------------------------
models   = {}
encoders = {}
scalers  = {}
metadata = {}

@app.on_event("startup")
async def load_models():
    try:
        # ===== PURCHASE =====
        logger.info("📦 Chargement des modèles PURCHASE...")
        models["purchase_classif"]          = joblib.load(CLASSIF_PURCHASE_PATH / "modele.joblib")
        encoders["purchase_classif_four"]   = joblib.load(CLASSIF_PURCHASE_PATH / "encodeur_fournisseur.joblib")
        encoders["purchase_classif_cat"]    = joblib.load(CLASSIF_PURCHASE_PATH / "encodeur_categorie.joblib")
        logger.info("  ✓ Classification PURCHASE chargée")

        models["purchase_reg"]              = joblib.load(REGR_PURCHASE_PATH / "modele.joblib")
        encoders["purchase_reg_four"]       = joblib.load(REGR_PURCHASE_PATH / "encodeur_fournisseur.joblib")
        encoders["purchase_reg_cat"]        = joblib.load(REGR_PURCHASE_PATH / "encodeur_categorie.joblib")
        encoders["purchase_reg_meth"]       = joblib.load(REGR_PURCHASE_PATH / "encodeur_methode.joblib")
        logger.info("  ✓ Régression PURCHASE chargée")

        models["purchase_clust"]            = joblib.load(CLUST_PURCHASE_PATH / "modele.joblib")
        scalers["purchase_clust"]           = joblib.load(CLUST_PURCHASE_PATH / "scaler.joblib")
        logger.info("  ✓ Clustering PURCHASE chargé")

        # ===== COMMERCIAL B2C =====
        logger.info("🏪 Chargement des modèles COMMERCIAL B2C...")
        try:
            models["commercial_reg"] = joblib.load(REGR_COMMERCIAL_PATH / "best_model.pkl")
            with open(REGR_COMMERCIAL_PATH / "metadata.json") as f:
                metadata["commercial_reg"] = json.load(f)
            logger.info("  ✓ Régression COMMERCIAL B2C chargée")
        except FileNotFoundError as e:
            logger.warning(f"  ⚠ Régression COMMERCIAL B2C non trouvée: {e}")

        try:
            models["commercial_classif"]  = joblib.load(CLASSIF_COMMERCIAL_PATH / "best_model.pkl")
            scalers["commercial_classif"] = joblib.load(CLASSIF_COMMERCIAL_PATH / "scaler.pkl")
            with open(CLASSIF_COMMERCIAL_PATH / "metadata.json") as f:
                metadata["commercial_classif"] = json.load(f)
            logger.info("  ✓ Classification COMMERCIAL B2C chargée")
        except FileNotFoundError:
            logger.warning("  ⚠ Classification COMMERCIAL B2C non trouvée")

        try:
            models["commercial_forecast"] = joblib.load(FORECAST_COMMERCIAL_PATH / "model_prophet.pkl")
            with open(FORECAST_COMMERCIAL_PATH / "metadata.json") as f:
                metadata["commercial_forecast"] = json.load(f)
            logger.info("  ✓ Forecast COMMERCIAL B2C chargé")
        except FileNotFoundError:
            logger.warning("  ⚠ Forecast COMMERCIAL B2C non trouvé")

        try:
            models["commercial_anomaly"]  = joblib.load(ANOMALY_COMMERCIAL_PATH / "iso_forest.pkl")
            scalers["commercial_anomaly"] = joblib.load(ANOMALY_COMMERCIAL_PATH / "scaler.pkl")
            with open(ANOMALY_COMMERCIAL_PATH / "metadata.json") as f:
                metadata["commercial_anomaly"] = json.load(f)
            logger.info("  ✓ Anomalie Detection COMMERCIAL B2C chargée")
        except FileNotFoundError:
            logger.warning("  ⚠ Anomalie Detection COMMERCIAL B2C non trouvée")

        # ===== MARKETING =====
        logger.info("📊 Chargement des modèles MARKETING...")
        try:
            models["marketing_clust"]  = joblib.load(CLUST_MARKETING_PATH / "modele.joblib")
            scalers["marketing_clust"] = joblib.load(CLUST_MARKETING_PATH / "scaler.joblib")
            if (CLUST_MARKETING_PATH / "feature_names.joblib").exists():
                metadata["marketing_clust_features"] = list(joblib.load(CLUST_MARKETING_PATH / "feature_names.joblib"))
            logger.info("  ✓ Clustering MARKETING chargé")
        except Exception as e:
            logger.warning(f"  ⚠ Clustering MARKETING non chargé: {e}")

        try:
            import pickle
            with open(TIMESERIES_MARKETING_PATH / "sarima_model.pkl", "rb") as f:
                models["marketing_timeseries"] = pickle.load(f)
            metadata["marketing_timeseries_config"] = joblib.load(TIMESERIES_MARKETING_PATH / "sarima_config.joblib")
            logger.info("  ✓ Timeseries MARKETING (SARIMA) chargée")
        except Exception as e:
            logger.warning(f"  ⚠ Timeseries MARKETING non chargée: {e}")

        try:
            models["marketing_reg"] = joblib.load(REGR_MARKETING_PATH / "modele.joblib")
            logger.info("  ✓ Régression MARKETING chargée")
        except Exception as e:
            logger.warning(f"  ⚠ Régression MARKETING non chargée: {e}")

        try:
            models["marketing_classif"] = joblib.load(CLASSIF_MARKETING_PATH / "modele.joblib")
            logger.info("  ✓ Classification MARKETING chargée")
        except Exception as e:
            logger.warning(f"  ⚠ Classification MARKETING non chargée: {e}")

        # ===== GM =====
        logger.info("🎲 Chargement des modèles GM...")
        try:
            models["gm_classif"] = joblib.load(CLASSIF_GM_PATH / "modele.joblib")
            logger.info("  ✓ Classification GM chargée")
        except Exception as e:
            logger.warning(f"  ⚠ Classification GM non chargée: {e}")

        try:
            models["gm_reg"]  = joblib.load(REGR_GM_PATH / "modele.joblib")
            scalers["gm_reg"] = joblib.load(REGR_GM_PATH / "scaler.joblib")
            logger.info("  ✓ Régression GM chargée")
        except Exception as e:
            logger.warning(f"  ⚠ Régression GM non chargée: {e}")

        try:
            models["gm_clust"]  = joblib.load(CLUST_GM_PATH / "modele.joblib")
            scalers["gm_clust"] = joblib.load(CLUST_GM_PATH / "scaler.joblib")
            logger.info("  ✓ Clustering GM chargé")
        except Exception as e:
            logger.warning(f"  ⚠ Clustering GM non chargé: {e}")

        try:
            models["gm_anomaly"] = joblib.load(ANOMALY_GM_PATH / "modele_anomalie.joblib")
            logger.info("  ✓ Anomalie GM chargée")
        except Exception as e:
            logger.warning(f"  ⚠ Anomalie GM non chargée: {e}")

        # ===== B2B =====
        logger.info("📊 Chargement des modèles B2B...")
        try:
            models["b2b_anomaly"]  = joblib.load(ANOMALY_B2B_PATH / "anomaly_1_isolation_forest.joblib")
            scalers["b2b_anomaly"] = joblib.load(ANOMALY_B2B_PATH / "anomaly_scaler.joblib")
            with open(ANOMALY_B2B_PATH / "anomaly_meta.json") as f:
                metadata["b2b_anomaly"] = json.load(f)
            logger.info("  ✓ Anomaly B2B chargée")
        except Exception as e:
            logger.warning(f"  ⚠ Anomaly B2B non chargée: {e}")

        try:
            models["b2b_classif"] = joblib.load(CLASSIF_B2B_PATH / "classif_1_random_forest.joblib")
            with open(CLASSIF_B2B_PATH / "classif_meta.json") as f:
                metadata["b2b_classif"] = json.load(f)
            logger.info("  ✓ Classification B2B chargée")
        except Exception as e:
            logger.warning(f"  ⚠ Classification B2B non chargée: {e}")

        try:
            models["b2b_risks"] = joblib.load(RISKS_B2B_PATH / "classif_risk_randomforest.joblib")
            logger.info("  ✓ Classification Risks B2B chargée")
        except Exception as e:
            logger.warning(f"  ⚠ Classification Risks B2B non chargée: {e}")

        try:
            models["b2b_clust"]  = joblib.load(CLUST_B2B_PATH / "cluster_2_kmeans.joblib")
            scalers["b2b_clust"] = joblib.load(CLUST_B2B_PATH / "cluster_scaler.joblib")
            with open(CLUST_B2B_PATH / "cluster_meta.json") as f:
                metadata["b2b_clust"] = json.load(f)
            logger.info("  ✓ Clustering B2B chargé")
        except Exception as e:
            logger.warning(f"  ⚠ Clustering B2B non chargé: {e}")

        try:
            models["b2b_forecast"]  = joblib.load(FORECAST_B2B_PATH / "forecast_1_xgboost_regressor.joblib")
            scalers["b2b_forecast"] = joblib.load(FORECAST_B2B_PATH / "forecast_scaler.joblib")
            with open(FORECAST_B2B_PATH / "forecast_meta.json") as f:
                metadata["b2b_forecast"] = json.load(f)
            logger.info("  ✓ Forecast B2B chargé")
        except Exception as e:
            logger.warning(f"  ⚠ Forecast B2B non chargé: {e}")

        try:
            models["b2b_reg"] = joblib.load(REGR_B2B_PATH / "regression_1_xgboost.joblib")
            with open(REGR_B2B_PATH / "regression_meta.json") as f:
                metadata["b2b_reg"] = json.load(f)
            logger.info("  ✓ Regression B2B chargée")
        except Exception as e:
            logger.warning(f"  ⚠ Regression B2B non chargée: {e}")

        # ===== FINANCIER =====
        logger.info("💰 Chargement des modèles FINANCIER...")
        try:
            scalers["fin_clust"] = joblib.load(CLUST_FIN_PATH / "scale_clusteringr.joblib")
            models["fin_clust"]  = joblib.load(CLUST_FIN_PATH / "clustering_model.joblib")
            logger.info("  ✓ Clustering FINANCIER chargé")
        except Exception as e:
            logger.warning(f"  ⚠ Clustering FINANCIER non chargé: {e}")

        try:
            forecast_path = FORECAST_FIN_PATH / "forecasting_best_model.joblib"
            if forecast_path.exists():
                models["fin_forecast"] = joblib.load(forecast_path)
                logger.info("  ✓ Forecasting FINANCIER chargé")
            else:
                logger.warning(f"  ⚠ Fichier forecasting introuvable: {forecast_path}")
        except Exception as e:
            logger.warning(f"  ⚠ Forecasting FINANCIER non chargé: {e}")

        logger.info("✅ Tous les modèles disponibles sont chargés !")

    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement des modèles: {e}")
        raise RuntimeError(f"Impossible de charger les modèles: {e}")

# ------------------------------------------------------------------
# FONCTIONS UTILITAIRES
# ------------------------------------------------------------------
def encode_field(value: str, encoder, field_name: str):
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        raise HTTPException(status_code=400, detail=f"Valeur inconnue pour {field_name}: {value}")

# ------------------------------------------------------------------
# PRÉPARATEURS DE FEATURES - PURCHASE
# ------------------------------------------------------------------
def prepare_classif_features_p(f: ClassificationFeaturesP):
    four_enc = encode_field(f.fournisseur, encoders["purchase_classif_four"], "fournisseur")
    cat_enc  = encode_field(f.categorie,   encoders["purchase_classif_cat"],  "categorie")
    return np.array([[f.Montant_HT, f.Taux_TVA, f.Marge_TVA, f.Mois, f.Annee,
                      f.Semaine, f.Est_weekend, four_enc, cat_enc]])

def prepare_regression_features_p(f: RegressionFeaturesP):
    four_enc = encode_field(f.fournisseur, encoders["purchase_reg_four"], "fournisseur")
    cat_enc  = encode_field(f.categorie,   encoders["purchase_reg_cat"],  "categorie")
    meth_enc = encode_field(f.methode,     encoders["purchase_reg_meth"], "methode")
    return np.array([[f.Mois, f.Annee, f.Semaine, f.Est_weekend,
                      four_enc, cat_enc, meth_enc, f.Taux_TVA]])

def prepare_clustering_features_p(f: ClusteringFeaturesP):
    X = np.array([[f.Nb_Factures, f.Montant_Total, f.Montant_Moyen, f.Montant_Max, f.TVA_Moy]])
    return scalers["purchase_clust"].transform(X)

# ------------------------------------------------------------------
# PRÉPARATEURS DE FEATURES - COMMERCIAL B2C
# ------------------------------------------------------------------
def prepare_regression_features_c(f: RegressionFeaturesC):
    X = np.array([[f.feat_avg_price, f.feat_max_price, f.feat_free_shipping, f.feat_shipping_pct,
                   f.feat_payment_encoded, f.feat_is_tunis, f.feat_nb_products, f.feat_total_qty,
                   f.feat_has_promo, f.feat_is_gift]])
    if metadata.get("commercial_reg", {}).get("requires_scaling", False):
        X = scalers["commercial_reg"].transform(X)
    return X

def prepare_classification_features_c(f: ClassificationFeaturesC):
    X = np.array([[f.feat_is_peak_season, f.feat_payment_encoded, f.feat_shipping_pct, f.feat_is_tunis,
                   f.feat_avg_price, f.feat_max_price, f.feat_has_note, f.feat_has_promo,
                   f.feat_free_shipping, f.feat_discount]])
    if metadata.get("commercial_classif", {}).get("requires_scaling", False):
        X = scalers["commercial_classif"].transform(X)
    return X

# ------------------------------------------------------------------
# PRÉPARATEURS DE FEATURES - MARKETING
# ------------------------------------------------------------------
def prepare_clustering_features_m(f: ClusteringFeaturesM):
    feature_order = ['price_current','discount_depth','rating_value','reviews_count',
                     'sales_qty','sales_revenue','order_lines','sales_velocity','review_signal']
    return scalers["marketing_clust"].transform(pd.DataFrame({col: [getattr(f, col)] for col in feature_order}))

def prepare_regression_features_m(f: RegressionFeaturesM):
    feature_order = ['discount_depth','rating_value','reviews_count','name_len','desc_len',
                     'sales_qty','sales_revenue','order_lines','avg_qty','avg_unit_price',
                     'days_on_sale','sales_velocity','revenue_per_orderline','review_signal',
                     'broad_category','stock_status','main_payment','main_delivery']
    return pd.DataFrame({col: [getattr(f, col)] for col in feature_order})

def prepare_classification_features_m(f: ClassificationFeaturesM):
    feature_order = ['price_current','discount_depth','rating_value','reviews_count','name_len',
                     'desc_len','sales_qty','sales_revenue','order_lines','avg_qty','avg_unit_price',
                     'days_on_sale','sales_velocity','revenue_per_orderline','review_signal',
                     'broad_category','stock_status','main_payment','main_delivery']
    return pd.DataFrame({col: [getattr(f, col)] for col in feature_order})

# ------------------------------------------------------------------
# PRÉPARATEURS DE FEATURES - GM
# ------------------------------------------------------------------
def prepare_regression_features_gm(f: RegressionFeaturesGM):
    return np.array([[
        f.est_weekend, f.mois, f.trimestre, f.quantite,
        f.categorie_id_2, f.categorie_id_3, f.categorie_id_4, f.categorie_id_5,
        f.categorie_id_6, f.categorie_id_7, f.categorie_id_8, f.categorie_id_9,
        f.categorie_id_10, f.categorie_id_11, f.categorie_id_12,
        f.canal_id_3, f.canal_id_4,
        f.gouvernorat_Ben_Arous, f.gouvernorat_Bizerte, f.gouvernorat_INCONNU,
        f.gouvernorat_Monastir, f.gouvernorat_Nabeul, f.gouvernorat_Sfax,
        f.gouvernorat_Sousse, f.gouvernorat_Tunis
    ]])

def prepare_clustering_features_gm(f: ClusteringFeaturesGM):
    return np.array([[f.Recency, f.Frequency, f.Monetary, f.Avg_Order_Value,
                      f.Nb_Categories, f.Pct_Weekend, f.Is_Online_Buyer]])

# ------------------------------------------------------------------
# PRÉPARATEURS DE FEATURES - FINANCIER
# ------------------------------------------------------------------
def prepare_clustering_features_fin(f: ClusteringFeaturesFin):
    feature_order = ['nb_commandes','ca_total','panier_moyen','total_remise',
                     'total_remboursement','livraison_moyenne','total_articles',
                     'nb_categories','nb_produits','taux_finalisation','recence']
    return np.array([[getattr(f, feat) for feat in feature_order]])

# ======================================================================
# ENDPOINTS - PURCHASE
# ======================================================================
@app.post("/decideur-purchase/classification/predict")
async def predict_classif_purchase(features: ClassificationFeaturesP):
    start = time.time()
    X = prepare_classif_features_p(features)

    t0 = time.time()
    prediction = models["purchase_classif"].predict(X)[0]
    PREDICTION_TIME.labels(decideur="purchase", task="classification").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="purchase", task="classification", prediction_class=str(prediction)).inc()

    try:
        proba_arr = models["purchase_classif"].predict_proba(X)[0]
        confidence = float(max(proba_arr))
        drift_detector.update_prediction("purchase", "classification", confidence)
        logger.info(f"Purchase classification - confiance: {confidence:.3f}")
        proba = proba_arr.tolist()
    except Exception as e:
        logger.warning(f"Confiance purchase classification indisponible: {e}")
        proba = None

    logger.info(f"Purchase classification terminée en {time.time()-start:.3f}s")
    return {"decideur": "purchase", "task": "classification",
            "prediction": int(prediction), "probabilities": proba}


@app.post("/decideur-purchase/regression/predict")
async def predict_reg_purchase(features: RegressionFeaturesP):
    start = time.time()
    X = prepare_regression_features_p(features)

    t0 = time.time()
    prediction = models["purchase_reg"].predict(X)[0]
    PREDICTION_TIME.labels(decideur="purchase", task="regression").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="purchase", task="regression", prediction_class="value").inc()

    logger.info(f"Purchase regression terminée en {time.time()-start:.3f}s")
    return {"decideur": "purchase", "task": "regression", "predicted_value": float(prediction)}


@app.post("/decideur-purchase/clustering/predict")
async def predict_clust_purchase(features: ClusteringFeaturesP):
    start = time.time()
    X_scaled = prepare_clustering_features_p(features)

    t0 = time.time()
    cluster = models["purchase_clust"].predict(X_scaled)[0]
    PREDICTION_TIME.labels(decideur="purchase", task="clustering").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="purchase", task="clustering", prediction_class=str(cluster)).inc()

    logger.info(f"Purchase clustering terminé en {time.time()-start:.3f}s")
    return {"decideur": "purchase", "task": "clustering", "cluster": int(cluster)}


# ======================================================================
# ENDPOINTS - COMMERCIAL B2C
# ======================================================================
@app.post("/decideur-commercial/regression/predict")
async def predict_reg_commercial(features: RegressionFeaturesC):
    start = time.time()
    X = prepare_regression_features_c(features)

    t0 = time.time()
    prediction_log = models["commercial_reg"].predict(X)[0]
    PREDICTION_TIME.labels(decideur="commercial", task="regression").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="commercial", task="regression", prediction_class="value").inc()

    prediction_tnd = np.expm1(prediction_log)
    logger.info(f"Commercial regression terminée en {time.time()-start:.3f}s")
    return {"decideur": "commercial_b2c", "task": "regression",
            "predicted_value_log": float(prediction_log),
            "predicted_value_tnd": float(prediction_tnd),
            "note": "Valeur en TND"}


@app.post("/decideur-commercial/classification/predict")
async def predict_classif_commercial(features: ClassificationFeaturesC):
    if "commercial_classif" not in models:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    start = time.time()
    X = prepare_classification_features_c(features)

    t0 = time.time()
    prediction = models["commercial_classif"].predict(X)[0]
    PREDICTION_TIME.labels(decideur="commercial", task="classification").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="commercial", task="classification", prediction_class=str(prediction)).inc()

    try:
        proba_arr = models["commercial_classif"].predict_proba(X)[0]
        confidence = float(max(proba_arr))
        drift_detector.update_prediction("commercial", "classification", confidence)
        logger.info(f"Commercial classification - confiance: {confidence:.3f}")
        proba = proba_arr.tolist()
    except Exception as e:
        logger.warning(f"Confiance commercial classification indisponible: {e}")
        proba = None

    classes = {0: "En cours", 1: "Terminée"}
    logger.info(f"Commercial classification terminée en {time.time()-start:.3f}s")
    return {"decideur": "commercial_b2c", "task": "classification",
            "prediction": int(prediction),
            "label": classes.get(int(prediction), "Unknown"),
            "probabilities": proba}


@app.post("/decideur-commercial/anomaly/detect")
async def detect_anomaly_commercial(features: AnomalyDetectionFeaturesC):
    if "commercial_anomaly" not in models:
        raise HTTPException(status_code=503, detail="Modèle d'anomalie non chargé")
    start = time.time()
    X = np.array([[features.feat_nb_products, features.feat_avg_price, features.feat_max_price,
                   features.feat_discount, features.feat_shipping_pct, features.target_value]])
    X_scaled = scalers["commercial_anomaly"].transform(X)

    t0 = time.time()
    anomaly_prediction = models["commercial_anomaly"].predict(X_scaled)[0]
    anomaly_score      = models["commercial_anomaly"].score_samples(X_scaled)[0]
    PREDICTION_TIME.labels(decideur="commercial", task="anomaly_detection").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="commercial", task="anomaly_detection",
                              prediction_class=str(anomaly_prediction)).inc()

    if int(anomaly_prediction) == -1:
        logger.warning(f"🚨 Anomalie COMMERCIAL détectée - score: {anomaly_score:.3f}")

    logger.info(f"Commercial anomaly detection terminée en {time.time()-start:.3f}s")
    return {"decideur": "commercial_b2c", "task": "anomaly_detection",
            "anomaly_score": float(anomaly_score),
            "is_anomaly": int(anomaly_prediction) == -1,
            "prediction": int(anomaly_prediction),
            "note": "-1 = anomalie, 1 = normal"}


@app.post("/decideur-commercial/forecast/predict")
async def forecast_commercial(input_data: ForecastInputC):
    if "commercial_forecast" not in models:
        raise HTTPException(status_code=503, detail="Modèle de forecast non chargé")
    start = time.time()
    try:
        future         = models["commercial_forecast"].make_future_dataframe(periods=input_data.periods)
        forecast_result = models["commercial_forecast"].predict(future)
        future_forecast = forecast_result.tail(input_data.periods)
        predictions = [{"ds": str(row["ds"].date()),
                        "yhat": float(row["yhat"]),
                        "yhat_lower": float(row["yhat_lower"]),
                        "yhat_upper": float(row["yhat_upper"])}
                       for _, row in future_forecast.iterrows()]

        PREDICTION_TIME.labels(decideur="commercial", task="forecast").observe(time.time() - start)
        MODEL_PREDICTIONS.labels(decideur="commercial", task="forecast", prediction_class="forecast").inc()

        logger.info(f"Commercial forecast terminé en {time.time()-start:.3f}s")
        return {"decideur": "commercial_b2c", "task": "forecast",
                "periods": input_data.periods, "predictions": predictions}
    except Exception as e:
        logger.error(f"Erreur forecast commercial: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================================
# ENDPOINTS - MARKETING
# ======================================================================
@app.post("/decideur-marketing/clustering/predict")
async def predict_clust_marketing(features: ClusteringFeaturesM):
    if "marketing_clust" not in models:
        raise HTTPException(status_code=503, detail="Modèle de clustering marketing non chargé")
    start = time.time()
    X_scaled = prepare_clustering_features_m(features)

    t0 = time.time()
    cluster = models["marketing_clust"].predict(X_scaled)[0]
    PREDICTION_TIME.labels(decideur="marketing", task="clustering").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="marketing", task="clustering", prediction_class=str(cluster)).inc()

    logger.info(f"Marketing clustering terminé en {time.time()-start:.3f}s")
    return {"decideur": "marketing", "task": "clustering",
            "cluster": int(cluster), "n_clusters": models["marketing_clust"].n_clusters}


@app.post("/decideur-marketing/timeseries/forecast")
async def forecast_marketing(input_data: TimeseriesInputM):
    if "marketing_timeseries" not in models:
        raise HTTPException(status_code=503, detail="Modèle SARIMA marketing non chargé")
    start = time.time()
    try:
        model          = models["marketing_timeseries"]
        forecast_result = model.forecast(steps=input_data.periods)
        forecast_list  = forecast_result.tolist() if hasattr(forecast_result, "tolist") else list(forecast_result)
        config         = metadata.get("marketing_timeseries_config", {})

        PREDICTION_TIME.labels(decideur="marketing", task="timeseries").observe(time.time() - start)
        MODEL_PREDICTIONS.labels(decideur="marketing", task="timeseries", prediction_class="forecast").inc()

        logger.info(f"Marketing timeseries terminée en {time.time()-start:.3f}s")
        return {"decideur": "marketing", "task": "timeseries_forecast",
                "periods": input_data.periods, "forecast": forecast_list,
                "order": config.get("order"),
                "seasonal_order": config.get("seasonal_order"),
                "last_train_date": config.get("last_train_date")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prévision: {e}")


@app.post("/decideur-marketing/regression/predict")
async def predict_reg_marketing(features: RegressionFeaturesM):
    if "marketing_reg" not in models:
        raise HTTPException(status_code=503, detail="Modèle de régression marketing non chargé")
    start = time.time()
    df = prepare_regression_features_m(features)

    t0 = time.time()
    prediction = models["marketing_reg"].predict(df)[0]
    PREDICTION_TIME.labels(decideur="marketing", task="regression").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="marketing", task="regression", prediction_class="value").inc()

    logger.info(f"Marketing regression terminée en {time.time()-start:.3f}s")
    return {"decideur": "marketing", "task": "regression", "predicted_value": float(prediction)}


@app.post("/decideur-marketing/classification/predict")
async def predict_classif_marketing(features: ClassificationFeaturesM):
    if "marketing_classif" not in models:
        raise HTTPException(status_code=503, detail="Modèle de classification marketing non chargé")
    start = time.time()
    df = prepare_classification_features_m(features)

    t0 = time.time()
    prediction = models["marketing_classif"].predict(df)[0]
    PREDICTION_TIME.labels(decideur="marketing", task="classification").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="marketing", task="classification", prediction_class=str(prediction)).inc()

    try:
        proba_arr = models["marketing_classif"].predict_proba(df)[0]
        confidence = float(max(proba_arr))
        drift_detector.update_prediction("marketing", "classification", confidence)
        logger.info(f"Marketing classification - confiance: {confidence:.3f}")
        proba = proba_arr.tolist()
    except Exception as e:
        logger.warning(f"Confiance marketing classification indisponible: {e}")
        proba = None

    logger.info(f"Marketing classification terminée en {time.time()-start:.3f}s")
    return {"decideur": "marketing", "task": "classification",
            "prediction": int(prediction), "probabilities": proba}


# ======================================================================
# ENDPOINTS - GM
# ======================================================================
@app.post("/decideur-gm/classification/predict")
async def predict_classif_gm(features: ClassificationFeaturesGM):
    if "gm_classif" not in models:
        raise HTTPException(status_code=503, detail="Modèle de classification GM non chargé")
    start = time.time()
    X = np.array([[features.Recency, features.Customer_Age_Days, features.Pct_Weekend,
                   features.Frequency, features.Nb_Categories, features.Monetary, features.Avg_Price]])

    t0 = time.time()
    prediction = models["gm_classif"].predict(X)[0]
    PREDICTION_TIME.labels(decideur="gm", task="classification").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="gm", task="classification", prediction_class=str(prediction)).inc()

    try:
        proba_arr = models["gm_classif"].predict_proba(X)[0]
        confidence = float(max(proba_arr))
        drift_detector.update_prediction("gm", "classification", confidence)
        logger.info(f"GM classification - confiance: {confidence:.3f}")
        proba = proba_arr.tolist()
    except Exception as e:
        logger.warning(f"Confiance GM classification indisponible: {e}")
        proba = None

    logger.info(f"GM classification terminée en {time.time()-start:.3f}s")
    return {"decideur": "gm", "task": "classification",
            "prediction": int(prediction), "probabilities": proba}


@app.post("/decideur-gm/regression/predict")
async def predict_reg_gm(features: RegressionFeaturesGM):
    if "gm_reg" not in models or "gm_reg" not in scalers:
        raise HTTPException(status_code=503, detail="Modèle de régression GM non chargé")
    start = time.time()
    X        = prepare_regression_features_gm(features)
    X_scaled = scalers["gm_reg"].transform(X)

    t0 = time.time()
    prediction = models["gm_reg"].predict(X_scaled)[0]
    PREDICTION_TIME.labels(decideur="gm", task="regression").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="gm", task="regression", prediction_class="value").inc()

    logger.info(f"GM regression terminée en {time.time()-start:.3f}s")
    return {"decideur": "gm", "task": "regression", "predicted_value": float(prediction)}


@app.post("/decideur-gm/clustering/predict")
async def predict_clust_gm(features: ClusteringFeaturesGM):
    if "gm_clust" not in models or "gm_clust" not in scalers:
        raise HTTPException(status_code=503, detail="Modèle de clustering GM non chargé")
    start = time.time()
    X        = prepare_clustering_features_gm(features)
    X_scaled = scalers["gm_clust"].transform(X)

    t0 = time.time()
    cluster = models["gm_clust"].predict(X_scaled)[0]
    PREDICTION_TIME.labels(decideur="gm", task="clustering").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="gm", task="clustering", prediction_class=str(cluster)).inc()

    logger.info(f"GM clustering terminé en {time.time()-start:.3f}s")
    return {"decideur": "gm", "task": "clustering", "cluster": int(cluster)}


@app.post("/decideur-gm/anomaly/detect")
async def detect_anomaly_gm(features: AnomalyFeaturesGM):
    if "gm_anomaly" not in models:
        raise HTTPException(status_code=503, detail="Modèle d'anomalie GM non chargé")
    start = time.time()
    X = np.array([[features.prix_unitaire, features.quantite, features.montant_total,
                   features.mois, features.est_weekend]])

    t0 = time.time()
    anomaly_pred  = models["gm_anomaly"].predict(X)[0]
    anomaly_score = models["gm_anomaly"].score_samples(X)[0]
    PREDICTION_TIME.labels(decideur="gm", task="anomaly_detection").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="gm", task="anomaly_detection",
                              prediction_class=str(anomaly_pred)).inc()

    if int(anomaly_pred) == -1:
        logger.warning(f"🚨 Anomalie GM détectée - score: {anomaly_score:.3f}")

    logger.info(f"GM anomaly detection terminée en {time.time()-start:.3f}s")
    return {"decideur": "gm", "task": "anomaly_detection",
            "anomaly_score": float(anomaly_score),
            "is_anomaly": int(anomaly_pred) == -1,
            "prediction": int(anomaly_pred),
            "note": "-1 = anomalie, 1 = normal"}


# ======================================================================
# ENDPOINTS - B2B
# ======================================================================
@app.post("/decideur-b2b/anomaly/detect")
async def detect_anomaly_b2b(features: B2BFeaturesArray):
    if "b2b_anomaly" not in models or "b2b_anomaly" not in scalers:
        raise HTTPException(503, "Modèle d'anomalie B2B non chargé")
    start = time.time()
    X = np.array([features.features])
    expected = scalers["b2b_anomaly"].n_features_in_
    if len(X[0]) != expected:
        raise HTTPException(400, f"Features incorrectes: attendu {expected}, reçu {len(X[0])}")
    X_scaled = scalers["b2b_anomaly"].transform(X)

    t0 = time.time()
    pred  = models["b2b_anomaly"].predict(X_scaled)[0]
    score = models["b2b_anomaly"].score_samples(X_scaled)[0]
    PREDICTION_TIME.labels(decideur="b2b", task="anomaly_detection").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="b2b", task="anomaly_detection", prediction_class=str(pred)).inc()

    if int(pred) == -1:
        logger.warning(f"🚨 Anomalie B2B détectée - score: {score:.3f}")

    logger.info(f"B2B anomaly detection terminée en {time.time()-start:.3f}s")
    return {"decideur": "b2b", "task": "anomaly_detection",
            "prediction": int(pred), "is_anomaly": int(pred) == -1, "anomaly_score": float(score)}


@app.post("/decideur-b2b/classification/predict")
async def predict_classif_b2b(features: B2BFeaturesArray):
    if "b2b_classif" not in models:
        raise HTTPException(503, "Modèle de classification B2B non chargé")
    start = time.time()
    X = np.array([features.features])
    expected = models["b2b_classif"].n_features_in_
    if len(X[0]) != expected:
        raise HTTPException(400, f"Features incorrectes: attendu {expected}, reçu {len(X[0])}")

    t0 = time.time()
    pred = models["b2b_classif"].predict(X)[0]
    PREDICTION_TIME.labels(decideur="b2b", task="classification").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="b2b", task="classification", prediction_class=str(pred)).inc()

    try:
        proba_arr  = models["b2b_classif"].predict_proba(X)[0]
        confidence = float(max(proba_arr))
        drift_detector.update_prediction("b2b", "classification", confidence)
        logger.info(f"B2B classification - confiance: {confidence:.3f}")
        proba = proba_arr.tolist()
    except Exception as e:
        logger.warning(f"Confiance B2B classification indisponible: {e}")
        proba = None

    logger.info(f"B2B classification terminée en {time.time()-start:.3f}s")
    return {"decideur": "b2b", "task": "classification",
            "prediction": int(pred), "probabilities": proba}


@app.post("/decideur-b2b/classification-risks/predict")
async def predict_risk_b2b(features: B2BFeaturesArray):
    if "b2b_risks" not in models:
        raise HTTPException(503, "Modèle de classification risques B2B non chargé")
    start = time.time()
    X = np.array([features.features])
    expected = models["b2b_risks"].n_features_in_
    if len(X[0]) != expected:
        raise HTTPException(400, f"Features incorrectes: attendu {expected}, reçu {len(X[0])}")

    t0 = time.time()
    pred = models["b2b_risks"].predict(X)[0]
    PREDICTION_TIME.labels(decideur="b2b", task="classification_risks").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="b2b", task="classification_risks", prediction_class=str(pred)).inc()

    try:
        proba_arr  = models["b2b_risks"].predict_proba(X)[0]
        confidence = float(max(proba_arr))
        drift_detector.update_prediction("b2b", "classification_risks", confidence)
        logger.info(f"B2B risks - confiance: {confidence:.3f}")
        proba = proba_arr.tolist()
    except Exception as e:
        logger.warning(f"Confiance B2B risks indisponible: {e}")
        proba = None

    logger.info(f"B2B classification risks terminée en {time.time()-start:.3f}s")
    return {"decideur": "b2b", "task": "classification_risks",
            "prediction": int(pred), "probabilities": proba}


@app.post("/decideur-b2b/clustering/predict")
async def cluster_b2b(features: B2BFeaturesArray):
    if "b2b_clust" not in models:
        raise HTTPException(503, "Modèle de clustering B2B non chargé")
    start = time.time()
    X = np.array([features.features])
    expected = models["b2b_clust"].n_features_in_
    if len(X[0]) != expected:
        raise HTTPException(400, f"Features incorrectes: attendu {expected}, reçu {len(X[0])}")

    t0 = time.time()
    cluster = models["b2b_clust"].predict(X)[0]
    PREDICTION_TIME.labels(decideur="b2b", task="clustering").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="b2b", task="clustering", prediction_class=str(cluster)).inc()

    logger.info(f"B2B clustering terminé en {time.time()-start:.3f}s")
    return {"decideur": "b2b", "task": "clustering", "cluster": int(cluster)}


@app.post("/decideur-b2b/forecast/predict")
async def forecast_b2b(features: B2BFeaturesArray):
    if "b2b_forecast" not in models:
        raise HTTPException(503, "Modèle de forecast B2B non chargé")
    start = time.time()
    X = np.array([features.features])
    expected = models["b2b_forecast"].n_features_in_
    if len(X[0]) != expected:
        raise HTTPException(400, f"Features incorrectes: attendu {expected}, reçu {len(X[0])}")

    t0 = time.time()
    pred = models["b2b_forecast"].predict(X)[0]
    PREDICTION_TIME.labels(decideur="b2b", task="forecast").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="b2b", task="forecast", prediction_class="value").inc()

    logger.info(f"B2B forecast terminé en {time.time()-start:.3f}s")
    return {"decideur": "b2b", "task": "forecast", "predicted_value": float(pred)}


@app.post("/decideur-b2b/regression/predict")
async def regress_b2b(features: B2BFeaturesArray):
    if "b2b_reg" not in models:
        raise HTTPException(503, "Modèle de régression B2B non chargé")
    start = time.time()
    X = np.array([features.features])
    expected = models["b2b_reg"].n_features_in_
    if len(X[0]) != expected:
        raise HTTPException(400, f"Features incorrectes: attendu {expected}, reçu {len(X[0])}")

    t0 = time.time()
    pred = models["b2b_reg"].predict(X)[0]
    PREDICTION_TIME.labels(decideur="b2b", task="regression").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="b2b", task="regression", prediction_class="value").inc()

    logger.info(f"B2B regression terminée en {time.time()-start:.3f}s")
    return {"decideur": "b2b", "task": "regression", "predicted_value": float(pred)}


@app.post("/decideur-b2b/classification/pipeline")
async def classify_pipeline_b2b(features: B2BClassificationPipelineFeatures):
    raise HTTPException(501, "Endpoint non utilisé. Utilisez /decideur-b2b/classification/predict")


# ======================================================================
# ENDPOINTS - FINANCIER
# ======================================================================
@app.post("/decideur-fin/clustering/predict")
async def cluster_fin(features: ClusteringFeaturesFin):
    if "fin_clust" not in models or "fin_clust" not in scalers:
        raise HTTPException(503, "Modèle de clustering financier non chargé")
    start = time.time()
    X        = prepare_clustering_features_fin(features)
    X_scaled = scalers["fin_clust"].transform(X)

    t0 = time.time()
    cluster = models["fin_clust"].predict(X_scaled)[0]
    PREDICTION_TIME.labels(decideur="fin", task="clustering").observe(time.time() - t0)
    MODEL_PREDICTIONS.labels(decideur="fin", task="clustering", prediction_class=str(cluster)).inc()

    logger.info(f"Fin clustering terminé en {time.time()-start:.3f}s")
    return {"decideur": "fin", "task": "clustering", "cluster": int(cluster)}


@app.post("/decideur-fin/forecast/predict")
async def forecast_fin(input_data: ForecastInputFin):
    if "fin_forecast" not in models:
        raise HTTPException(503, "Modèle de forecasting financier non chargé")
    start = time.time()
    try:
        model          = models["fin_forecast"]
        forecast_result = model.forecast(steps=input_data.periods)
        forecast_list  = forecast_result.tolist() if hasattr(forecast_result, "tolist") else list(forecast_result)

        PREDICTION_TIME.labels(decideur="fin", task="forecast").observe(time.time() - start)
        MODEL_PREDICTIONS.labels(decideur="fin", task="forecast", prediction_class="forecast").inc()

        logger.info(f"Fin forecast terminé en {time.time()-start:.3f}s")
        return {"decideur": "fin", "task": "forecast",
                "periods": input_data.periods, "forecast": forecast_list}
    except Exception as e:
        raise HTTPException(500, detail=f"Erreur de prévision: {e}")


# ======================================================================
# ENDPOINTS GÉNÉRAUX
# ======================================================================
@app.get("/")
def home():
    return {
        "status": "✅ API Sougui is running",
        "version": "3.5.0",
        "decideurs": ["purchase", "commercial_b2c", "marketing", "gm", "b2b", "fin"],
        "tasks": ["classification", "regression", "clustering", "anomaly_detection",
                  "forecast", "timeseries_forecast", "classification_risks"]
    }

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys()), "count": len(models)}

@app.get("/info")
def info():
    return {
        "purchase": {
            "classification": "✓" if "purchase_classif" in models else "✗",
            "regression":     "✓" if "purchase_reg"    in models else "✗",
            "clustering":     "✓" if "purchase_clust"  in models else "✗",
        },
        "commercial_b2c": {
            "regression":       "✓" if "commercial_reg"     in models else "✗",
            "classification":   "✓" if "commercial_classif" in models else "✗",
            "anomaly_detection":"✓" if "commercial_anomaly" in models else "✗",
            "forecast":         "✓" if "commercial_forecast"in models else "✗",
        },
        "marketing": {
            "clustering":   "✓" if "marketing_clust"      in models else "✗",
            "timeseries":   "✓" if "marketing_timeseries" in models else "✗",
            "regression":   "✓" if "marketing_reg"        in models else "✗",
            "classification":"✓" if "marketing_classif"   in models else "✗",
        },
        "gm": {
            "classification":   "✓" if "gm_classif" in models else "✗",
            "regression":       "✓" if "gm_reg"     in models else "✗",
            "clustering":       "✓" if "gm_clust"   in models else "✗",
            "anomaly_detection":"✓" if "gm_anomaly" in models else "✗",
        },
        "b2b": {
            "anomaly":              "✓" if "b2b_anomaly" in models else "✗",
            "classification":       "✓" if "b2b_classif" in models else "✗",
            "classification_risks": "✓" if "b2b_risks"   in models else "✗",
            "clustering":           "✓" if "b2b_clust"   in models else "✗",
            "forecast":             "✓" if "b2b_forecast" in models else "✗",
            "regression":           "✓" if "b2b_reg"     in models else "✗",
        },
        "fin": {
            "clustering": "✓" if "fin_clust"    in models else "✗",
            "forecast":   "✓" if "fin_forecast" in models else "✗",
            "classification": "✗ (incompatibilité)",
            "regression":     "✗ (incompatibilité)",
        },
        "metadata": metadata,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)