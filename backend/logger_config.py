# logger_config.py
import logging
import json
from datetime import datetime
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

class JSONFormatter(logging.Formatter):
    """Formateur JSON pour des logs exploitables"""
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Ajouter les champs extra si présents
        for key in ["decideur", "task", "confidence", "drift_score",
                    "latency", "prediction", "anomaly_score", "trigger"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        return json.dumps(log_data, ensure_ascii=False)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console (lisible)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    ))

    # Fichier JSON (exploitable)
    file_handler = logging.FileHandler("logs/mlops.json", encoding="utf-8")
    file_handler.setFormatter(JSONFormatter())

    # Fichier erreurs séparé
    error_handler = logging.FileHandler("logs/errors.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())

    if not logger.handlers:
        logger.addHandler(console)
        logger.addHandler(file_handler)
        logger.addHandler(error_handler)

    return logger


# Logger principal
mlops_logger = get_logger("mlops")

def log_prediction(decideur, task, confidence, latency, prediction):
    """Log structuré d'une prédiction"""
    extra = {"decideur": decideur, "task": task,
             "confidence": round(confidence, 4),
             "latency": round(latency, 4),
             "prediction": str(prediction)}
    mlops_logger.info(
        f"Prédiction {decideur}/{task} | conf={confidence:.3f} | lat={latency*1000:.1f}ms",
        extra=extra
    )

def log_anomaly(decideur, task, score):
    """Log d'une anomalie détectée"""
    extra = {"decideur": decideur, "task": task,
             "anomaly_score": round(score, 4), "trigger": "anomaly_detected"}
    mlops_logger.warning(
        f"🚨 ANOMALIE {decideur}/{task} | score={score:.4f}",
        extra=extra
    )

def log_drift(decideur, task, confidence, baseline):
    """Log d'un drift de confiance détecté"""
    drop_pct = ((baseline - confidence) / baseline) * 100
    extra = {"decideur": decideur, "task": task,
             "confidence": round(confidence, 4),
             "baseline": baseline,
             "drop_pct": round(drop_pct, 2),
             "trigger": "drift_detected"}
    mlops_logger.warning(
        f"📉 DRIFT {decideur}/{task} | conf={confidence:.3f} baseline={baseline:.3f} "
        f"baisse={drop_pct:.1f}%",
        extra=extra
    )
    # Si baisse > 5%, log retraining trigger
    if drop_pct > 5:
        retrain_extra = {**extra, "trigger": "retrain_required"}
        mlops_logger.error(
            f"🔁 RETRAINING REQUIS {decideur}/{task} | baisse={drop_pct:.1f}% > 5%",
            extra=retrain_extra
        )

def log_error(endpoint, error_type, detail):
    """Log d'une erreur API"""
    extra = {"endpoint": endpoint, "error_type": error_type, "trigger": "api_error"}
    mlops_logger.error(
        f"❌ ERREUR {endpoint} | {error_type}: {detail}",
        extra=extra
    )