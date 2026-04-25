import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import {
  MarketingClusteringRequest,
  MarketingRegressionRequest,
  MarketingClassificationRequest,
} from '../../models/api.models';

export type MarketingTab = 'clustering' | 'timeseries' | 'regression' | 'classification';

@Component({
  selector: 'app-marketing',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './marketing.component.html',
  styleUrl: './marketing.component.scss',
})
export class MarketingComponent {
  activeTab: MarketingTab = 'clustering';
  loading = false;
  error = '';
  result: any = null;

  // ── Clustering (9 features) ──────────────────────────────────
  clusterForm: MarketingClusteringRequest = {
    price_current: 49.99,
    discount_depth: 0.15,
    rating_value: 4.2,
    reviews_count: 120,
    sales_qty: 450,
    sales_revenue: 13495.5,
    order_lines: 3,
    sales_velocity: 15.2,
    review_signal: 0.85,
  };

  // ── Timeseries ───────────────────────────────────────────────
  tsPeriods = 12;

  // ── Régression (18 features) ─────────────────────────────────
  regForm: MarketingRegressionRequest = {
    discount_depth: 0.12,
    rating_value: 4.5,
    reviews_count: 85,
    name_len: 24,
    desc_len: 180,
    sales_qty: 320,
    sales_revenue: 11200.0,
    order_lines: 2,
    avg_qty: 160,
    avg_unit_price: 35.0,
    days_on_sale: 45,
    sales_velocity: 7.11,
    revenue_per_orderline: 5600.0,
    review_signal: 0.78,
    broad_category: 'Electronics',
    stock_status: 'in_stock',
    main_payment: 'credit_card',
    main_delivery: 'express',
  };

  // ── Classification (19 features) ─────────────────────────────
  classifForm: MarketingClassificationRequest = {
    price_current: 59.99,
    discount_depth: 0.10,
    rating_value: 4.8,
    reviews_count: 200,
    name_len: 30,
    desc_len: 250,
    sales_qty: 500,
    sales_revenue: 25000.0,
    order_lines: 5,
    avg_qty: 100,
    avg_unit_price: 50.0,
    days_on_sale: 60,
    sales_velocity: 8.33,
    revenue_per_orderline: 5000.0,
    review_signal: 0.92,
    broad_category: 'Electronics',
    stock_status: 'in_stock',
    main_payment: 'paypal',
    main_delivery: 'standard',
  };

  readonly categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports', 'Other'];
  readonly stockStatuses = ['in_stock', 'out_of_stock', 'limited'];
  readonly paymentMethods = ['credit_card', 'paypal', 'cash', 'virement'];
  readonly deliveryMethods = ['express', 'standard', 'pickup'];

  constructor(private api: ApiService) {}

  setTab(tab: MarketingTab): void {
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
      case 'clustering':      obs$ = this.api.marketingCluster(this.clusterForm);            break;
      case 'timeseries':      obs$ = this.api.marketingTimeseries({ periods: this.tsPeriods }); break;
      case 'regression':      obs$ = this.api.marketingRegress(this.regForm);                break;
      case 'classification':  obs$ = this.api.marketingClassify(this.classifForm);           break;
    }
    obs$.subscribe({
      next: (res: any) => { this.result = res; this.loading = false; },
      error: (err: Error) => { this.error = err.message; this.loading = false; },
    });
  }
}