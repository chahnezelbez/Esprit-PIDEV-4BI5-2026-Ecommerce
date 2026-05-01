import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { MaxYhatPipe } from './max-yhat.pipe';
import {
  CommercialRegressionRequest,
  CommercialClassificationRequest,
  CommercialAnomalyRequest,
  CommercialRegressionResponse,
  CommercialClassificationResponse,
  AnomalyResponse,
  CommercialForecastResponse,
} from '../../models/api.models';

export type CommercialTab = 'regression' | 'classification' | 'anomaly' | 'forecast';

// ── Métadonnées issues des fichiers metadata.json ────────────────
// IsolationForest : contamination=0.05 → 5% des commandes marquées anomalie
// Prophet : entraîné sur 2023-07-05 → 2026-01-02, target = daily_orders
// Classification : Classe 0 = En cours | Classe 1 = Terminée

const CLASSIF_META: Record<number, { label: string; desc: string; icon: string }> = {
  0: {
    label: 'Commande en cours',
    desc: 'La commande n\'est pas encore finalisée. Un suivi commercial est recommandé.',
    icon: '🔄',
  },
  1: {
    label: 'Commande terminée',
    desc: 'La commande est finalisée et livrée. Le cycle commercial est complet.',
    icon: '✅',
  },
};

@Component({
  selector: 'app-commercial',
  standalone: true,
  imports: [CommonModule, FormsModule, MaxYhatPipe],
  templateUrl: './commercial.component.html',
  styleUrl: './commercial.component.scss',
})
export class CommercialComponent {
  activeTab = signal<CommercialTab>('regression');
  loading   = signal(false);
  errorMsg  = signal<string | null>(null);

  regrResult    = signal<CommercialRegressionResponse | null>(null);
  classifResult = signal<CommercialClassificationResponse | null>(null);
  anomalyResult = signal<AnomalyResponse | null>(null);
  forecastResult = signal<CommercialForecastResponse | null>(null);

  // ── Formulaires pré-remplis avec données de test ─────────────
  regForm: CommercialRegressionRequest = {
    feat_avg_price: 29.99,
    feat_max_price: 49.99,
    feat_free_shipping: 1,
    feat_shipping_pct: 0.05,
    feat_payment_encoded: 1,
    feat_is_tunis: 1,
    feat_nb_products: 3,
    feat_total_qty: 5,
    feat_has_promo: 0,
    feat_is_gift: 0,
  };

  classifForm: CommercialClassificationRequest = {
    feat_is_peak_season: 0,
    feat_payment_encoded: 2,
    feat_shipping_pct: 0.03,
    feat_is_tunis: 1,
    feat_avg_price: 35.5,
    feat_max_price: 55.0,
    feat_has_note: 1,
    feat_has_promo: 0,
    feat_free_shipping: 1,
    feat_discount: 0.1,
  };

  // Features issues de features.json :
  // feat_nb_products, feat_avg_price, feat_max_price,
  // feat_discount, feat_shipping_pct, target_value
  anomalyForm: CommercialAnomalyRequest = {
    feat_nb_products: 4,
    feat_avg_price: 120.0,
    feat_max_price: 180.0,
    feat_discount: 0.15,
    feat_shipping_pct: 0.02,
    target_value: 500.0,
  };

  forecastPeriods = 30;

  constructor(private api: ApiService) {}

  setTab(tab: CommercialTab): void {
    this.activeTab.set(tab);
    this.errorMsg.set(null);
  }

  private clearResults(): void {
    this.regrResult.set(null);
    this.classifResult.set(null);
    this.anomalyResult.set(null);
    this.forecastResult.set(null);
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
      case 'regression':
        this.api.commercialRegress(this.regForm).subscribe({
          next: (r) => { this.regrResult.set(r); this.loading.set(false); },
          error: handleError,
        });
        break;
      case 'classification':
        this.api.commercialClassify(this.classifForm).subscribe({
          next: (r) => { this.classifResult.set(r); this.loading.set(false); },
          error: handleError,
        });
        break;
      case 'anomaly':
        this.api.commercialAnomaly(this.anomalyForm).subscribe({
          next: (r) => { this.anomalyResult.set(r); this.loading.set(false); },
          error: handleError,
        });
        break;
      case 'forecast':
        this.api.commercialForecast({ periods: this.forecastPeriods }).subscribe({
          next: (r) => { this.forecastResult.set(r); this.loading.set(false); },
          error: handleError,
        });
        break;
    }
  }

  getClassifMeta(pred: number) {
    return CLASSIF_META[pred] ?? { label: `Classe ${pred}`, desc: '', icon: '❓' };
  }

  probPercent(val: number): string {
    return (val * 100).toFixed(1) + '%';
  }

  /** Score anomalie : -1 = très anormal, 0 = seuil, positif = normal */
  anomalyScoreColor(score: number): string {
    if (score < -0.1) return 'score--red';
    if (score < 0)    return 'score--orange';
    return 'score--green';
  }

  anomalyScoreBar(score: number): number {
    // Normalise score [-0.5, 0.2] → [0, 100]
    const pct = ((score + 0.5) / 0.7) * 100;
    return Math.min(100, Math.max(0, pct));
  }
}