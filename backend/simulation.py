# simulation.py
import requests
import time
import random
import concurrent.futures
import json

BASE_URL = "http://localhost:8000"

# ================================================================
# DONNÉES DE TEST RÉALISTES
# ================================================================
PURCHASE_CLASSIF_PAYLOAD = {
    "Montant_HT": 1500.0, "Taux_TVA": 19.0, "Marge_TVA": 285.0,
    "Mois": 6, "Annee": 2024, "Semaine": 24, "Est_weekend": 0,
    "fournisseur": "FOURNISSEUR_A", "categorie": "INFORMATIQUE"
}

PURCHASE_REG_PAYLOAD = {
    "Mois": 6, "Annee": 2024, "Semaine": 24, "Est_weekend": 0,
    "fournisseur": "FOURNISSEUR_A", "categorie": "INFORMATIQUE",
    "methode": "VIREMENT", "Taux_TVA": 19.0
}

PURCHASE_CLUST_PAYLOAD = {
    "Nb_Factures": 12.0, "Montant_Total": 45000.0,
    "Montant_Moyen": 3750.0, "Montant_Max": 8000.0, "TVA_Moy": 19.0
}

COMMERCIAL_REG_PAYLOAD = {
    "feat_avg_price": 45.5, "feat_max_price": 120.0,
    "feat_free_shipping": 1.0, "feat_shipping_pct": 0.05,
    "feat_payment_encoded": 2.0, "feat_is_tunis": 1.0,
    "feat_nb_products": 3.0, "feat_total_qty": 5.0,
    "feat_has_promo": 1.0, "feat_is_gift": 0.0
}

COMMERCIAL_CLASSIF_PAYLOAD = {
    "feat_is_peak_season": 1.0, "feat_payment_encoded": 2.0,
    "feat_shipping_pct": 0.05, "feat_is_tunis": 1.0,
    "feat_avg_price": 45.5, "feat_max_price": 120.0,
    "feat_has_note": 1.0, "feat_has_promo": 1.0,
    "feat_free_shipping": 1.0, "feat_discount": 0.1
}

MARKETING_CLUST_PAYLOAD = {
    "price_current": 29.99, "discount_depth": 0.15,
    "rating_value": 4.2, "reviews_count": 145.0,
    "sales_qty": 230.0, "sales_revenue": 6897.7,
    "order_lines": 89.0, "sales_velocity": 7.6, "review_signal": 0.87
}

GM_CLASSIF_PAYLOAD = {
    "Recency": 15.0, "Customer_Age_Days": 365.0, "Pct_Weekend": 0.3,
    "Frequency": 12.0, "Nb_Categories": 4.0,
    "Monetary": 2500.0, "Avg_Price": 208.3
}

FIN_CLUST_PAYLOAD = {
    "nb_commandes": 25.0, "ca_total": 12500.0, "panier_moyen": 500.0,
    "total_remise": 800.0, "total_remboursement": 200.0,
    "livraison_moyenne": 8.5, "total_articles": 87.0,
    "nb_categories": 6.0, "nb_produits": 34.0,
    "taux_finalisation": 0.88, "recence": 7.0
}

ALL_ENDPOINTS = [
    ("/decideur-purchase/classification/predict",  PURCHASE_CLASSIF_PAYLOAD),
    ("/decideur-purchase/regression/predict",      PURCHASE_REG_PAYLOAD),
    ("/decideur-purchase/clustering/predict",      PURCHASE_CLUST_PAYLOAD),
    ("/decideur-commercial/regression/predict",    COMMERCIAL_REG_PAYLOAD),
    ("/decideur-commercial/classification/predict",COMMERCIAL_CLASSIF_PAYLOAD),
    ("/decideur-marketing/clustering/predict",     MARKETING_CLUST_PAYLOAD),
    ("/decideur-gm/classification/predict",        GM_CLASSIF_PAYLOAD),
    ("/decideur-fin/clustering/predict",           FIN_CLUST_PAYLOAD),
]

# ================================================================
# SCÉNARIO 1 — TRAFIC ÉLEVÉ
# ================================================================
def scenario_high_traffic(n_requests=100, workers=10):
    """
    Envoie 100 requêtes en parallèle pour simuler un pic de trafic.
    Observer dans Grafana : latence P95 qui monte, requêtes/sec qui spike.
    """
    print("\n" + "="*60)
    print("🚀 SCÉNARIO 1 : TRAFIC ÉLEVÉ")
    print(f"   {n_requests} requêtes avec {workers} workers parallèles")
    print("="*60)

    results = {"success": 0, "error": 0, "total_time": 0}

    def send_request(_):
        endpoint, payload = random.choice(ALL_ENDPOINTS)
        try:
            start = time.time()
            r = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=10)
            elapsed = time.time() - start
            return {"status": r.status_code, "time": elapsed}
        except Exception as e:
            return {"status": 0, "time": 0, "error": str(e)}

    start_global = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = list(executor.map(send_request, range(n_requests)))

    for r in futures:
        if r["status"] == 200:
            results["success"] += 1
        else:
            results["error"] += 1
        results["total_time"] += r["time"]

    duration = time.time() - start_global
    avg_time = results["total_time"] / n_requests if n_requests > 0 else 0

    print(f"   ✅ Succès    : {results['success']}/{n_requests}")
    print(f"   ❌ Erreurs   : {results['error']}/{n_requests}")
    print(f"   ⏱  Durée     : {duration:.2f}s")
    print(f"   ⚡ Req/sec   : {n_requests/duration:.1f}")
    print(f"   📊 Latence moy: {avg_time*1000:.1f}ms")
    print("\n   👉 Vérifiez dans Grafana : latence P95 et requêtes/sec")


# ================================================================
# SCÉNARIO 2 — ERREURS API (400/503)
# ================================================================
def scenario_api_errors(n_requests=30):
    """
    Envoie des requêtes malformées pour générer des erreurs 400/422.
    Observer dans Grafana : spike sur error_total, taux d'erreur.
    """
    print("\n" + "="*60)
    print("💥 SCÉNARIO 2 : ERREURS API")
    print(f"   {n_requests} requêtes invalides")
    print("="*60)

    bad_payloads = [
        # Mauvais nombre de features B2B (attendu: variable, envoyé: 3)
        ("/decideur-b2b/classification/predict",
         {"features": [1.0, 2.0, 3.0]}),
        # Champs manquants
        ("/decideur-purchase/classification/predict",
         {"Montant_HT": 100.0}),
        # Valeur fournisseur inconnue
        ("/decideur-purchase/classification/predict",
         {**PURCHASE_CLASSIF_PAYLOAD, "fournisseur": "FOURNISSEUR_INCONNU_XYZ"}),
        # B2B anomaly avec mauvais nb features
        ("/decideur-b2b/anomaly/detect",
         {"features": [1.0, 2.0]}),
    ]

    error_counts = {}
    for i in range(n_requests):
        endpoint, payload = random.choice(bad_payloads)
        try:
            r = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=5)
            status = r.status_code
            error_counts[status] = error_counts.get(status, 0) + 1
            print(f"   [{i+1:02d}] {endpoint[-30:]:30s} → {status}", end="\r")
        except Exception as e:
            error_counts["timeout"] = error_counts.get("timeout", 0) + 1
        time.sleep(0.1)

    print("\n")
    print("   Résumé des erreurs :")
    for status, count in error_counts.items():
        print(f"   HTTP {status} : {count} fois")
    print("\n   👉 Vérifiez dans Grafana : error_total et taux d'erreur")


# ================================================================
# SCÉNARIO 3 — DRIFT DU MODÈLE (confiance qui baisse)
# ================================================================
def scenario_model_drift(n_requests=50):
    """
    Envoie des données avec des valeurs extrêmes/out-of-distribution
    pour simuler un drift — la confiance des modèles devrait baisser.
    Observer dans Grafana : model_confidence qui descend sous le baseline.
    """
    print("\n" + "="*60)
    print("📉 SCÉNARIO 3 : DRIFT DU MODÈLE")
    print(f"   {n_requests} requêtes avec données out-of-distribution")
    print("="*60)

    # Données avec valeurs extrêmes (hors distribution d'entraînement)
    drifted_payloads = [
        ("/decideur-purchase/classification/predict", {
            "Montant_HT": 9999999.0,   # valeur extrême
            "Taux_TVA": 99.0,
            "Marge_TVA": 9999.0,
            "Mois": 12, "Annee": 2030, "Semaine": 52,
            "Est_weekend": 1,
            "fournisseur": "FOURNISSEUR_A",
            "categorie": "INFORMATIQUE"
        }),
        ("/decideur-commercial/classification/predict", {
            "feat_is_peak_season": 1.0,
            "feat_payment_encoded": 999.0,   # hors distribution
            "feat_shipping_pct": 99.0,        # extrême
            "feat_is_tunis": 1.0,
            "feat_avg_price": 99999.0,        # extrême
            "feat_max_price": 999999.0,
            "feat_has_note": 1.0,
            "feat_has_promo": 1.0,
            "feat_free_shipping": 0.0,
            "feat_discount": 0.99
        }),
        ("/decideur-gm/classification/predict", {
            "Recency": 9999.0,         # très ancien
            "Customer_Age_Days": 0.0,   # client tout neuf
            "Pct_Weekend": 1.0,
            "Frequency": 0.0,          # jamais acheté
            "Nb_Categories": 100.0,    # extrême
            "Monetary": 0.01,          # presque rien
            "Avg_Price": 0.001
        }),
        ("/decideur-marketing/classification/predict", {
            "price_current": 999999.0,
            "discount_depth": 0.99,
            "rating_value": 0.1,       # très bas
            "reviews_count": 0.0,
            "name_len": 1000.0,
            "desc_len": 10000.0,
            "sales_qty": 0.0,
            "sales_revenue": 0.0,
            "order_lines": 0.0,
            "avg_qty": 0.0,
            "avg_unit_price": 0.0,
            "days_on_sale": 9999.0,
            "sales_velocity": 0.0,
            "revenue_per_orderline": 0.0,
            "review_signal": 0.0,
            "broad_category": "OTHER",
            "stock_status": "out_of_stock",
            "main_payment": "unknown",
            "main_delivery": "unknown"
        }),
    ]

    confidence_values = []
    for i in range(n_requests):
        endpoint, payload = random.choice(drifted_payloads)
        try:
            r = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("probabilities"):
                    confidence = max(data["probabilities"])
                    confidence_values.append(confidence)
                    print(f"   [{i+1:02d}] Confiance: {confidence:.3f}", end="\r")
        except Exception as e:
            pass
        time.sleep(0.2)

    print("\n")
    if confidence_values:
        avg_conf = sum(confidence_values) / len(confidence_values)
        min_conf = min(confidence_values)
        print(f"   Confiance moyenne : {avg_conf:.3f} (baseline: 0.70)")
        print(f"   Confiance minimum : {min_conf:.3f}")
        if avg_conf < 0.70:
            print(f"   ⚠️  DRIFT DÉTECTÉ — confiance sous le baseline !")
        else:
            print(f"   ✅ Confiance acceptable")
    print("\n   👉 Vérifiez dans Grafana : model_confidence")


# ================================================================
# MAIN — Menu interactif
# ================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🧪 SOUGUI MLOps — Simulation de Scénarios")
    print("="*60)
    print("  1. Trafic élevé (100 requêtes parallèles)")
    print("  2. Erreurs API  (requêtes malformées)")
    print("  3. Drift modèle (données hors distribution)")
    print("  4. Tous les scénarios (complet)")
    print("  0. Quitter")
    print("="*60)

    choice = input("\nChoix : ").strip()

    if choice == "1":
        scenario_high_traffic(n_requests=100, workers=10)
    elif choice == "2":
        scenario_api_errors(n_requests=30)
    elif choice == "3":
        scenario_model_drift(n_requests=50)
    elif choice == "4":
        print("\n🎬 Lancement de tous les scénarios...")
        print("   ⏳ Gardez Grafana ouvert sur http://localhost:3001")
        input("   Appuyez sur Entrée pour commencer...")

        scenario_high_traffic(n_requests=80, workers=8)
        print("\n   ⏸  Pause 15s — observez le spike de trafic dans Grafana...")
        time.sleep(15)

        scenario_api_errors(n_requests=25)
        print("\n   ⏸  Pause 15s — observez les erreurs dans Grafana...")
        time.sleep(15)

        scenario_model_drift(n_requests=40)
        print("\n   ⏸  Pause 15s — observez la confiance dans Grafana...")
        time.sleep(15)

        print("\n✅ Tous les scénarios terminés !")
        print("   📸 Prenez des screenshots de Grafana maintenant.")
    else:
        print("Au revoir !")