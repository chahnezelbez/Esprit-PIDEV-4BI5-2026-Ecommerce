# monitoring.py — VERSION CORRIGÉE ET COMPLÈTE
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from fastapi import FastAPI, Request, Response
from starlette.types import ASGIApp, Receive, Scope, Send
import time
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime
import psutil

logger = logging.getLogger(__name__)

# =============================================
# MÉTRIQUES PROMETHEUS
# =============================================

# 1. MÉTRIQUES DE TRAFIC
REQUESTS_TOTAL = Counter(
    'http_requests_total', 
    'Total des requêtes HTTP', 
    ['method', 'endpoint', 'status']
)

REQUESTS_IN_PROGRESS = Gauge(
    'http_requests_in_progress', 
    'Requêtes en cours', 
    ['endpoint']
)

# 2. MÉTRIQUES DE PERFORMANCE
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds', 
    'Latence des requêtes en secondes',
    ['method', 'endpoint'], 
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
)

PREDICTION_TIME = Histogram(
    'prediction_duration_seconds', 
    'Temps de prédiction des modèles',
    ['decideur', 'task'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1)
)

# 3. MÉTRIQUES DE STABILITÉ
ERROR_COUNTER = Counter(
    'error_total', 
    'Nombre total d\'erreurs', 
    ['endpoint', 'error_type']
)

# 4. MÉTRIQUES DE SANTÉ DES MODÈLES
MODEL_CONFIDENCE = Gauge(
    'model_confidence', 
    'Confiance moyenne du modèle', 
    ['decideur', 'task']
)

MODEL_PREDICTIONS = Counter(
    'model_predictions_total', 
    'Nombre total de prédictions',
    ['decideur', 'task', 'prediction_class']
)

# 5. MÉTRIQUES DE DRIFT
DATA_DRIFT_SCORE = Gauge(
    'data_drift_score', 
    'Score de drift des données', 
    ['decideur', 'task']
)

# 6. MÉTRIQUES SYSTÈME
CPU_USAGE = Gauge('cpu_usage_percent', 'Utilisation CPU (%)')
MEMORY_USAGE = Gauge('memory_usage_percent', 'Utilisation mémoire (%)')
API_HEALTH = Gauge('api_health_status', 'Statut de santé API (1=OK, 0=KO)')

# =============================================
# BASELINES
# =============================================
BASELINES = {
    'latency': {
        'purchase': 0.1,
        'commercial': 0.15, 
        'marketing': 0.12,
        'gm': 0.13,
        'b2b': 0.11,
        'fin': 0.1
    },
    'confidence': {
        'purchase': 0.75,
        'commercial': 0.70,
        'marketing': 0.72,
        'gm': 0.73,
        'b2b': 0.74,
        'fin': 0.71
    }
}

# =============================================
# DÉTECTEUR DE DRIFT
# =============================================
class DriftDetector:
    def __init__(self):
        self.confidence_history: Dict[str, List] = {}
        self.drift_history: Dict[str, List] = {}
    
    def update_prediction(self, decideur: str, task: str, confidence: float):
        key = f"{decideur}_{task}"
        
        if key not in self.confidence_history:
            self.confidence_history[key] = []
        
        self.confidence_history[key].append(confidence)
        
        if len(self.confidence_history[key]) > 500:
            self.confidence_history[key].pop(0)
        
        if len(self.confidence_history[key]) > 0:
            avg_confidence = np.mean(self.confidence_history[key])
            MODEL_CONFIDENCE.labels(decideur=decideur, task=task).set(avg_confidence)
            
            baseline = BASELINES['confidence'].get(decideur, 0.70)
            if avg_confidence < baseline * 0.95:
                logger.warning(f"⚠️ Dégradation confiance {key}: {avg_confidence:.3f} (baseline: {baseline:.3f})")
    
    def update_drift_score(self, decideur: str, task: str, score: float):
        DATA_DRIFT_SCORE.labels(decideur=decideur, task=task).set(score)
        if score > 1.5:
            logger.warning(f"🚨 Drift détecté {decideur}/{task}: score={score:.2f}")

drift_detector = DriftDetector()

# =============================================
# MIDDLEWARE DE MONITORING
# =============================================
class MonitoringMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        endpoint = request.url.path
        
        # Ignorer les OPTIONS (preflight CORS)
        if request.method == "OPTIONS":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
                REQUESTS_TOTAL.labels(
                    method=request.method,
                    endpoint=endpoint,
                    status=status_code
                ).inc()
                if status_code >= 400:
                    ERROR_COUNTER.labels(
                        endpoint=endpoint, 
                        error_type=f"http_{status_code}"
                    ).inc()
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            ERROR_COUNTER.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
            raise
        finally:
            latency = time.time() - start_time
            REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(latency)
            REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()

# =============================================
# CONFIGURATION
# =============================================
def setup_monitoring(app: FastAPI) -> FastAPI:
    """Configure le monitoring pour l'application"""
    
    # Ajouter le middleware
    app.add_middleware(MonitoringMiddleware)
    
    # Endpoint pour Prometheus
    @app.get("/metrics", include_in_schema=False)
    async def get_metrics():
        try:
            CPU_USAGE.set(psutil.cpu_percent())
            MEMORY_USAGE.set(psutil.virtual_memory().percent)
            API_HEALTH.set(1)
        except:
            pass
        return Response(content=generate_latest(REGISTRY), media_type="text/plain")
    
    return app