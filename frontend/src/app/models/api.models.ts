// ═══════════════════════════════════════════════════════════════════
// PURCHASE
// ═══════════════════════════════════════════════════════════════════
export interface PurchaseClassificationRequest {
  Montant_HT: number;
  Taux_TVA: number;
  Marge_TVA: number;
  Mois: number;
  Annee: number;
  Semaine: number;
  Est_weekend: number;
  fournisseur: string;
  categorie: string;
}
export interface PurchaseClassificationResponse {
  decideur: string;
  task: string;
  prediction: number;
  probabilities: number[] | null;
}

export interface PurchaseRegressionRequest {
  Mois: number;
  Annee: number;
  Semaine: number;
  Est_weekend: number;
  fournisseur: string;
  categorie: string;
  methode: string;
  Taux_TVA: number;
}
export interface PurchaseRegressionResponse {
  decideur: string;
  task: string;
  predicted_value: number;
}

export interface PurchaseClusteringRequest {
  Nb_Factures: number;
  Montant_Total: number;
  Montant_Moyen: number;
  Montant_Max: number;
  TVA_Moy: number;
}
export interface PurchaseClusteringResponse {
  decideur: string;
  task: string;
  cluster: number;
}

// ═══════════════════════════════════════════════════════════════════
// COMMERCIAL B2C
// ═══════════════════════════════════════════════════════════════════
export interface CommercialRegressionRequest {
  feat_avg_price: number;
  feat_max_price: number;
  feat_free_shipping: number;
  feat_shipping_pct: number;
  feat_payment_encoded: number;
  feat_is_tunis: number;
  feat_nb_products: number;
  feat_total_qty: number;
  feat_has_promo: number;
  feat_is_gift: number;
}
export interface CommercialRegressionResponse {
  decideur: string;
  task: string;
  predicted_value_log: number;
  predicted_value_tnd: number;
  note: string;
}

export interface CommercialClassificationRequest {
  feat_is_peak_season: number;
  feat_payment_encoded: number;
  feat_shipping_pct: number;
  feat_is_tunis: number;
  feat_avg_price: number;
  feat_max_price: number;
  feat_has_note: number;
  feat_has_promo: number;
  feat_free_shipping: number;
  feat_discount: number;
}
export interface CommercialClassificationResponse {
  decideur: string;
  task: string;
  prediction: number;
  label: string;
  probabilities: number[] | null;
}

export interface CommercialAnomalyRequest {
  feat_nb_products: number;
  feat_avg_price: number;
  feat_max_price: number;
  feat_discount: number;
  feat_shipping_pct: number;
  target_value: number;
}
export interface AnomalyResponse {
  decideur: string;
  task: string;
  anomaly_score: number;
  is_anomaly: boolean;
  prediction: number;
  note: string;
}

export interface ForecastPeriodRequest {
  periods: number;
}
export interface CommercialForecastResponse {
  decideur: string;
  task: string;
  periods: number;
  predictions: { ds: string; yhat: number; yhat_lower: number; yhat_upper: number }[];
}

// ═══════════════════════════════════════════════════════════════════
// MARKETING
// ═══════════════════════════════════════════════════════════════════
export interface MarketingClusteringRequest {
  price_current: number;
  discount_depth: number;
  rating_value: number;
  reviews_count: number;
  sales_qty: number;
  sales_revenue: number;
  order_lines: number;
  sales_velocity: number;
  review_signal: number;
}
export interface MarketingClusteringResponse {
  decideur: string;
  task: string;
  cluster: number;
  n_clusters: number;
}

export interface MarketingTimeseriesResponse {
  decideur: string;
  task: string;
  periods: number;
  forecast: number[];
  order: number[] | null;
  seasonal_order: number[] | null;
  last_train_date: string | null;
}

export interface MarketingRegressionRequest {
  discount_depth: number;
  rating_value: number;
  reviews_count: number;
  name_len: number;
  desc_len: number;
  sales_qty: number;
  sales_revenue: number;
  order_lines: number;
  avg_qty: number;
  avg_unit_price: number;
  days_on_sale: number;
  sales_velocity: number;
  revenue_per_orderline: number;
  review_signal: number;
  broad_category: string;
  stock_status: string;
  main_payment: string;
  main_delivery: string;
}
export interface MarketingRegressionResponse {
  decideur: string;
  task: string;
  predicted_value: number;
}

export interface MarketingClassificationRequest {
  price_current: number;
  discount_depth: number;
  rating_value: number;
  reviews_count: number;
  name_len: number;
  desc_len: number;
  sales_qty: number;
  sales_revenue: number;
  order_lines: number;
  avg_qty: number;
  avg_unit_price: number;
  days_on_sale: number;
  sales_velocity: number;
  revenue_per_orderline: number;
  review_signal: number;
  broad_category: string;
  stock_status: string;
  main_payment: string;
  main_delivery: string;
}
export interface MarketingClassificationResponse {
  decideur: string;
  task: string;
  prediction: number;
  probabilities: number[] | null;
}

// ═══════════════════════════════════════════════════════════════════
// GM
// ═══════════════════════════════════════════════════════════════════
export interface GmClassificationRequest {
  Recency: number;
  Customer_Age_Days: number;
  Pct_Weekend: number;
  Frequency: number;
  Nb_Categories: number;
  Monetary: number;
  Avg_Price: number;
}
export interface GmClassificationResponse {
  decideur: string;
  task: string;
  prediction: number;
  probabilities: number[] | null;
}

export interface GmRegressionRequest {
  est_weekend: number;
  mois: number;
  trimestre: number;
  quantite: number;
  categorie_id_2: number;
  categorie_id_3: number;
  categorie_id_4: number;
  categorie_id_5: number;
  categorie_id_6: number;
  categorie_id_7: number;
  categorie_id_8: number;
  categorie_id_9: number;
  categorie_id_10: number;
  categorie_id_11: number;
  categorie_id_12: number;
  canal_id_3: number;
  canal_id_4: number;
  gouvernorat_Ben_Arous: number;
  gouvernorat_Bizerte: number;
  gouvernorat_INCONNU: number;
  gouvernorat_Monastir: number;
  gouvernorat_Nabeul: number;
  gouvernorat_Sfax: number;
  gouvernorat_Sousse: number;
  gouvernorat_Tunis: number;
}
export interface GmRegressionResponse {
  decideur: string;
  task: string;
  predicted_value: number;
}

export interface GmClusteringRequest {
  Recency: number;
  Frequency: number;
  Monetary: number;
  Avg_Order_Value: number;
  Nb_Categories: number;
  Pct_Weekend: number;
  Is_Online_Buyer: number;
}
export interface GmClusteringResponse {
  decideur: string;
  task: string;
  cluster: number;
}

export interface GmAnomalyRequest {
  prix_unitaire: number;
  quantite: number;
  montant_total: number;
  mois: number;
  est_weekend: number;
}

// ═══════════════════════════════════════════════════════════════════
// B2B
// ═══════════════════════════════════════════════════════════════════
export interface B2bFeaturesRequest {
  features: number[];
}
export interface B2bClassificationResponse {
  decideur: string;
  task: string;
  prediction: number;
  probabilities: number[] | null;
}
export interface B2bClusteringResponse {
  decideur: string;
  task: string;
  cluster: number;
}
export interface B2bRegressionResponse {
  decideur: string;
  task: string;
  predicted_value: number;
}
export interface B2bAnomalyResponse {
  decideur: string;
  task: string;
  prediction: number;
  is_anomaly: boolean;
  anomaly_score: number;
}

// ═══════════════════════════════════════════════════════════════════
// FINANCIER
// ═══════════════════════════════════════════════════════════════════
export interface FinClusteringRequest {
  nb_commandes: number;
  ca_total: number;
  panier_moyen: number;
  total_remise: number;
  total_remboursement: number;
  livraison_moyenne: number;
  total_articles: number;
  nb_categories: number;
  nb_produits: number;
  taux_finalisation: number;
  recence: number;
}
export interface FinClusteringResponse {
  decideur: string;
  task: string;
  cluster: number;
}
export interface FinForecastResponse {
  decideur: string;
  task: string;
  periods: number;
  forecast: number[];
}
