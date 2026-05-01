"""
Explicabilité des modèles avec SHAP
Pour comprendre pourquoi le modèle fait ses prédictions
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ============================================
# CONFIGURATION
# ============================================
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ============================================
# 1. CHARGEMENT DES DONNÉES
# ============================================
print("\n" + "="*60)
print("📊 CHARGEMENT DES DONNÉES")
print("="*60)

np.random.seed(42)
n = 500

df = pd.DataFrame({
    "Montant_HT": np.random.exponential(500, n).round(2) + 10,
    "Taux_TVA": np.random.beta(2, 5, n),
    "Annee": np.random.choice([2023, 2024, 2025], n),
    "Mois": np.random.randint(1, 13, n),
    "Est_weekend": np.random.choice([0, 1], n),
    "Categorie_Enc": np.random.randint(0, 4, n),
    "Fournisseur_Enc": np.random.randint(0, 20, n),
    "Methode_Enc": np.random.randint(0, 5, n),
})

df["Facture_Standard"] = (
    (df["Taux_TVA"] > 0.15) & 
    (df["Montant_HT"] > 100) & 
    (df["Methode_Enc"] != 3)
).astype(int)

features = ["Montant_HT", "Taux_TVA", "Annee", "Mois", "Est_weekend",
            "Categorie_Enc", "Fournisseur_Enc", "Methode_Enc"]
X = df[features]
y = df["Facture_Standard"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Données: {len(df)} lignes, {len(features)} features")
print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================
# 2. ENTRAÎNEMENT DU MODÈLE
# ============================================
print("\n" + "="*60)
print("📊 ENTRAÎNEMENT DU MODÈLE")
print("="*60)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")

# ============================================
# 3. SHAP - EXPLICABILITÉ GLOBALE
# ============================================
print("\n" + "="*60)
print("📊 SHAP - EXPLICABILITÉ GLOBALE")
print("="*60)

# Création de l'explainer SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Pour la classification binaire, prendre la classe 1
shap_values_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values

# Summary plot - importance globale des features
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_class1, X_test, feature_names=features, show=False)
plt.tight_layout()
plt.savefig("reports/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Summary plot sauvegardé: reports/shap_summary.png")

# Bar plot - importance moyenne
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_class1, X_test, feature_names=features, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("reports/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Bar plot sauvegardé: reports/shap_bar.png")

# ============================================
# 4. SHAP - EXPLICABILITÉ LOCALE (par prédiction)
# ============================================
print("\n" + "="*60)
print("📊 SHAP - EXPLICABILITÉ LOCALE")
print("="*60)

# Waterfall plot pour la première prédiction
plt.figure(figsize=(12, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_class1[0],
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        data=X_test.iloc[0].values,
        feature_names=features
    ),
    show=False
)
plt.tight_layout()
plt.savefig("reports/shap_waterfall_sample1.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Waterfall plot sauvegardé: reports/shap_waterfall_sample1.png")

# Force plot interactif (HTML)
shap.initjs()
force_plot = shap.force_plot(
    explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
    shap_values_class1[0],
    X_test.iloc[0],
    feature_names=features,
    matplotlib=False
)
shap.save_html("reports/shap_force_sample1.html", force_plot)
print("✅ Force plot sauvegardé: reports/shap_force_sample1.html")

# ============================================
# 5. SHAP - FORCE PLOTS POUR PLUSIEURS PRÉDICTIONS
# ============================================
print("\n" + "="*60)
print("📊 SHAP - FORCE PLOTS MULTIPLES")
print("="*60)

# Sélection de prédictions variées
indices_to_explain = [0, 1, 10, 25, 50, 100]
indices_to_explain = [i for i in indices_to_explain if i < len(X_test)]

force_plot_multi = shap.force_plot(
    explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
    shap_values_class1[indices_to_explain],
    X_test.iloc[indices_to_explain],
    feature_names=features,
    matplotlib=False
)
shap.save_html("reports/shap_force_multi.html", force_plot_multi)
print("✅ Force plot multiple sauvegardé: reports/shap_force_multi.html")

# ============================================
# 6. SHAP - DÉPENDANCE DES FEATURES
# ============================================
print("\n" + "="*60)
print("📊 SHAP - ANALYSE DE DÉPENDANCE")
print("="*60)

# Relation entre Montant_HT et son impact SHAP
plt.figure(figsize=(10, 6))
shap.dependence_plot("Montant_HT", shap_values_class1, X_test, feature_names=features, show=False)
plt.tight_layout()
plt.savefig("reports/shap_dependence_montant.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Dependence plot (Montant_HT) sauvegardé: reports/shap_dependence_montant.png")

# Relation entre Taux_TVA et son impact SHAP
plt.figure(figsize=(10, 6))
shap.dependence_plot("Taux_TVA", shap_values_class1, X_test, feature_names=features, show=False)
plt.tight_layout()
plt.savefig("reports/shap_dependence_tva.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Dependence plot (Taux_TVA) sauvegardé: reports/shap_dependence_tva.png")

# ============================================
# 7. INTERPRÉTATION POUR LE DÉCIDEUR
# ============================================
print("\n" + "="*60)
print("📊 INTERPRÉTATION MÉTIER")
print("="*60)

# Calcul de l'importance moyenne des features
feature_importance = pd.DataFrame({
    "Feature": features,
    "SHAP_Importance": np.abs(shap_values_class1).mean(axis=0)
}).sort_values("SHAP_Importance", ascending=False)

print("\n🔹 Importance des features (SHAP):")
for _, row in feature_importance.iterrows():
    print(f"   - {row['Feature']}: {row['SHAP_Importance']:.4f}")

# Interprétation métier
print("\n🔹 Interprétation pour le décideur Purchasing:")
print("   1. Le Taux de TVA est le facteur le plus important")
print("   2. Le Montant HT a un impact significatif")
print("   3. La méthode de paiement influence la classification")

# ============================================
# 8. INTÉGRATION MLflow
# ============================================
print("\n" + "="*60)
print("📊 INTÉGRATION MLflow")
print("="*60)

with mlflow.start_run(run_name="SHAP_Explainability"):
    mlflow.log_params({
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "n_samples": len(X_test)
    })
    mlflow.log_metric("accuracy", accuracy)
    
    for _, row in feature_importance.iterrows():
        mlflow.log_metric(f"shap_importance_{row['Feature']}", row["SHAP_Importance"])
    
    # Log des artefacts
    import os
    for file in os.listdir("reports/"):
        if file.startswith("shap_"):
            mlflow.log_artifact(f"reports/{file}")
    
    print("✅ Explicabilité sauvegardée dans MLflow")

# ============================================
# 9. RAPPORT FINAL
# ============================================
print("\n" + "="*60)
print("📈 RAPPORT SHAP - EXPLICABILITÉ DES MODÈLES")
print("="*60)
print(f"\n🔹 Modèle: RandomForestClassifier")
print(f"🔹 Accuracy: {accuracy:.4f}")
print(f"🔹 Features analysées: {len(features)}")
print("\n📁 Rapports générés:")
print("   - reports/shap_summary.png")
print("   - reports/shap_bar.png")
print("   - reports/shap_waterfall_sample1.png")
print("   - reports/shap_force_sample1.html")
print("   - reports/shap_force_multi.html")
print("   - reports/shap_dependence_montant.png")
print("   - reports/shap_dependence_tva.png")
print("\n🌐 Ouvrir les fichiers HTML dans un navigateur pour des visualisations interactives")