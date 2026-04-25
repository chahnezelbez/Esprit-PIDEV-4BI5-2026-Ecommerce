import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import {
  CommercialRegressionRequest,
  CommercialClassificationRequest,
  CommercialAnomalyRequest,
} from '../../models/api.models';

export type CommercialTab = 'regression' | 'classification' | 'anomaly' | 'forecast';

@Component({
  selector: 'app-commercial',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './commercial.component.html',
  styleUrl: './commercial.component.scss',
})
export class CommercialComponent {
  activeTab: CommercialTab = 'regression';
  loading = false;
  error = '';
  result: any = null;

  regForm: CommercialRegressionRequest = {
    feat_avg_price: 29.99, feat_max_price: 49.99, feat_free_shipping: 1,
    feat_shipping_pct: 0.05, feat_payment_encoded: 1, feat_is_tunis: 1,
    feat_nb_products: 3, feat_total_qty: 5, feat_has_promo: 0, feat_is_gift: 0,
  };

  classifForm: CommercialClassificationRequest = {
    feat_is_peak_season: 0, feat_payment_encoded: 2, feat_shipping_pct: 0.03,
    feat_is_tunis: 1, feat_avg_price: 35.5, feat_max_price: 55.0,
    feat_has_note: 1, feat_has_promo: 0, feat_free_shipping: 1, feat_discount: 0.1,
  };

  anomalyForm: CommercialAnomalyRequest = {
    feat_nb_products: 4, feat_avg_price: 120.0, feat_max_price: 180.0,
    feat_discount: 0.15, feat_shipping_pct: 0.02, target_value: 500.0,
  };

  forecastPeriods = 30;

  constructor(private api: ApiService) {}

  setTab(tab: CommercialTab): void {
    this.activeTab = tab;
    this.result = null;
    this.error = '';
  }

  submit(): void {
    this.loading = true;
    this.error = '';
    this.result = null;
    let obs$: any;
    switch (this.activeTab) {
      case 'regression':     obs$ = this.api.commercialRegress(this.regForm); break;
      case 'classification': obs$ = this.api.commercialClassify(this.classifForm); break;
      case 'anomaly':        obs$ = this.api.commercialAnomaly(this.anomalyForm); break;
      case 'forecast':       obs$ = this.api.commercialForecast({ periods: this.forecastPeriods }); break;
    }
    obs$.subscribe({
      next: (res: any) => { this.result = res; this.loading = false; },
      error: (err: Error) => { this.error = err.message; this.loading = false; },
    });
  }
}