import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import {
  MarketingClusteringRequest,
  MarketingClusteringResponse,
  MarketingRegressionRequest,
  MarketingRegressionResponse,
  MarketingClassificationRequest,
  MarketingClassificationResponse,
  MarketingTimeseriesResponse,
} from '../../models/api.models';

export type MarketingTab = 'clustering' | 'timeseries' | 'regression' | 'classification';

// ── Données réelles extraites des fichiers joblib ────────────────
// Clustering KMeans — 9 features, StandardScaler
// Moyennes des features (scaler.mean_) :
//   price_current=133.6, discount=6.76, sales_qty=35.9, sales_revenue=1391.2
//   order_lines=4.88, sales_velocity=0.187
// Segments déduits des centres de clusters :
const CLUSTER_META: Record<number, { label: string; desc: string; icon: string; color: string }> = {
  0: {
    label: 'Produit standard',
    desc: 'Volume et prix dans la moyenne. Produit courant du catalogue artisanal.',
    icon: '🛍️',
    color: 'cluster--blue',
  },
  1: {
    label: 'Best-seller',
    desc: 'Forte vélocité, nombreux avis positifs. Produit phare à mettre en avant.',
    icon: '⭐',
    color: 'cluster--green',
  },
  2: {
    label: 'Produit dormant',
    desc: 'Faible rotation, peu d\'avis. Nécessite une action marketing (promo ou retrait).',
    icon: '💤',
    color: 'cluster--amber',
  },
};

// Régression : RandomForest(200 arbres) prédit price_current
// Top importances : price_current 37.7%, reviews_count 12.8%, sales_qty 3.7%

// SARIMA(0,1,1)(0,1,1,12) — entraîné 24 mois jusqu'au 2025-06-01
// Prévision : ~2700 TND/mois

// Classification : classes produit selon performance globale
const CLASSIF_META: Record<number, { label: string; desc: string; icon: string }> = {
  0: {
    label: 'Produit faible performance',
    desc: 'Ce produit génère peu de revenus et de visibilité. Action requise.',
    icon: '📉',
  },
  1: {
    label: 'Produit performant',
    desc: 'Ce produit affiche de bonnes métriques de vente et d\'engagement client.',
    icon: '📈',
  },
};

@Component({
  selector: 'app-marketing',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './marketing.component.html',
  styleUrl: './marketing.component.scss',
})
export class MarketingComponent {
  activeTab = signal<MarketingTab>('clustering');
  loading   = signal(false);
  errorMsg  = signal<string | null>(null);

  clusterResult    = signal<MarketingClusteringResponse | null>(null);
  tsResult         = signal<MarketingTimeseriesResponse | null>(null);
  regrResult       = signal<MarketingRegressionResponse | null>(null);
  classifResult    = signal<MarketingClassificationResponse | null>(null);

  // ── SARIMA config (lu depuis sarima_config.joblib) ───────────
  readonly sarimaConfig = {
    order: '(0,1,1)',
    seasonalOrder: '(0,1,1,12)',
    lastTrainDate: '2025-06-01',
    trainLength: 24,
  };

  // ── Clustering (9 features — ordre exact de feature_names.joblib) ─
  // ['price_current','discount_depth','rating_value','reviews_count',
  //  'sales_qty','sales_revenue','order_lines','sales_velocity','review_signal']
  clusterForm: MarketingClusteringRequest = {
    price_current:  49.99,
    discount_depth: 0.15,
    rating_value:   4.2,
    reviews_count:  120,
    sales_qty:      450,
    sales_revenue:  13495.5,
    order_lines:    3,
    sales_velocity: 15.2,
    review_signal:  0.85,
  };

  tsPeriods = 12;

  // ── Régression (18 features, SANS price_current) ─────────────
  // top feature importance : reviews_count (12.8%), sales_qty (3.7%)
  regForm: MarketingRegressionRequest = {
    discount_depth:          0.12,
    rating_value:            4.5,
    reviews_count:           85,
    name_len:                24,
    desc_len:                180,
    sales_qty:               320,
    sales_revenue:           11200.0,
    order_lines:             2,
    avg_qty:                 160,
    avg_unit_price:          35.0,
    days_on_sale:            45,
    sales_velocity:          7.11,
    revenue_per_orderline:   5600.0,
    review_signal:           0.78,
    broad_category:          'Electronics',
    stock_status:            'in_stock',
    main_payment:            'credit_card',
    main_delivery:           'express',
  };

  // ── Classification (19 features, AVEC price_current) ─────────
  classifForm: MarketingClassificationRequest = {
    price_current:           59.99,
    discount_depth:          0.10,
    rating_value:            4.8,
    reviews_count:           200,
    name_len:                30,
    desc_len:                250,
    sales_qty:               500,
    sales_revenue:           25000.0,
    order_lines:             5,
    avg_qty:                 100,
    avg_unit_price:          50.0,
    days_on_sale:            60,
    sales_velocity:          8.33,
    revenue_per_orderline:   5000.0,
    review_signal:           0.92,
    broad_category:          'Electronics',
    stock_status:            'in_stock',
    main_payment:            'paypal',
    main_delivery:           'standard',
  };

  readonly categories    = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports', 'Other'];
  readonly stockStatuses = ['in_stock', 'out_of_stock', 'limited'];
  readonly paymentMethods = ['credit_card', 'paypal', 'cash', 'virement'];
  readonly deliveryMethods = ['express', 'standard', 'pickup'];

  // Feature importances réelles (régression)
  readonly topFeatures = [
    { name: 'price_current',  pct: 37.7 },
    { name: 'reviews_count',  pct: 12.8 },
    { name: 'sales_qty',      pct: 3.7  },
    { name: 'sales_revenue',  pct: 1.2  },
  ];

  constructor(private api: ApiService) {}

  setTab(tab: MarketingTab): void {
    this.activeTab.set(tab);
    this.errorMsg.set(null);
  }

  private clearResults(): void {
    this.clusterResult.set(null);
    this.tsResult.set(null);
    this.regrResult.set(null);
    this.classifResult.set(null);
    this.errorMsg.set(null);
  }

  submit(): void {
    this.loading.set(true);
    this.clearResults();

    const handleError = (err: Error) => {
      this.errorMsg.set(err.message);
      this.loading.set(false);
    };

    switch (this.activeTab()) {
      case 'clustering':
        this.api.marketingCluster(this.clusterForm).subscribe({
          next: (r) => { this.clusterResult.set(r); this.loading.set(false); },
          error: handleError,
        });
        break;
      case 'timeseries':
        this.api.marketingTimeseries({ periods: this.tsPeriods }).subscribe({
          next: (r) => { this.tsResult.set(r); this.loading.set(false); },
          error: handleError,
        });
        break;
      case 'regression':
        this.api.marketingRegress(this.regForm).subscribe({
          next: (r) => { this.regrResult.set(r); this.loading.set(false); },
          error: handleError,
        });
        break;
      case 'classification':
        this.api.marketingClassify(this.classifForm).subscribe({
          next: (r) => { this.classifResult.set(r); this.loading.set(false); },
          error: handleError,
        });
        break;
    }
  }

  getClusterMeta(c: number) {
    return CLUSTER_META[c] ?? { label: `Cluster ${c}`, desc: '', icon: '📦', color: 'cluster--blue' };
  }

  getClassifMeta(c: number) {
    return CLASSIF_META[c] ?? { label: `Classe ${c}`, desc: '', icon: '❓' };
  }

  probPercent(v: number): string {
    return (v * 100).toFixed(1) + '%';
  }

  tsAvg(forecast: number[]): number {
    return forecast.reduce((a, b) => a + b, 0) / (forecast.length || 1);
  }

  tsMax(forecast: number[]): number {
    return Math.max(...forecast);
  }
}