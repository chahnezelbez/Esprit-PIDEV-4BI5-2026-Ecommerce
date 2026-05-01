import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import {
  B2bFeaturesRequest,
  B2bClassificationResponse,
  B2bClusteringResponse,
  B2bForecastResponse,
  B2bRegressionResponse,
  B2bAnomalyResponse,
} from '../../models/api.models';
import { ResultCardComponent } from '../../shared/result-card/result-card.component';

type B2bTask = 'classification' | 'classification-risks' | 'clustering' | 'forecast' | 'regression' | 'anomaly';

type B2bResponse = 
  | B2bClassificationResponse 
  | B2bClusteringResponse 
  | B2bForecastResponse 
  | B2bRegressionResponse 
  | B2bAnomalyResponse;

@Component({
  selector: 'app-b2b',
  standalone: true,
  imports: [CommonModule, FormsModule, ResultCardComponent],
  templateUrl: './b2b.component.html',
  styleUrls: ['./b2b.component.scss'],
})
export class B2bComponent {
  // Signals
  loading = signal(false);
  errorMsg = signal<string | null>(null);
  selectedTask = signal<B2bTask>('classification');
  
  // Typed result signals
  classifResult = signal<B2bClassificationResponse | null>(null);
  clusterResult = signal<B2bClusteringResponse | null>(null);
  forecastResult = signal<B2bForecastResponse | null>(null);
  regressResult = signal<B2bRegressionResponse | null>(null);
  anomalyResult = signal<B2bAnomalyResponse | null>(null);
  
  // Form Values
  featuresValue = '0.15,1,0.4,0.07,0.9,12,220,18,0.02,0.58';
  forecastPeriods = 12;

  availableTasks: B2bTask[] = [
    'classification',
    'classification-risks',
    'clustering',
    'forecast',
    'regression',
    'anomaly',
  ];

  // ✅ IMPORTANT: Inject ApiService in constructor
  constructor(private api: ApiService) {}

  setTask(task: B2bTask): void {
    this.selectedTask.set(task);
    this.clearResults();
    this.errorMsg.set(null);
  }

  /**
   * Clear all result signals
   */
  private clearResults(): void {
    this.classifResult.set(null);
    this.clusterResult.set(null);
    this.forecastResult.set(null);
    this.regressResult.set(null);
    this.anomalyResult.set(null);
  }

  /**
   * Parse features from input string
   */
  get features(): number[] {
    return this.featuresValue
      .split(',')
      .map((value) => Number(value.trim()))
      .filter((value) => !Number.isNaN(value));
  }

  /**
   * Get user-friendly label for task
   */
  getTaskLabel(task: B2bTask): string {
    const labels: Record<B2bTask, string> = {
      classification: '📊 Classification',
      'classification-risks': '⚠️ Risques',
      clustering: '🎯 Clustering',
      forecast: '📈 Prévision',
      regression: '📉 Régression',
      anomaly: '🔍 Anomalie',
    };
    return labels[task];
  }

  /**
   * Get result title based on selected task
   */
  getResultTitle(): string {
    const titles: Record<B2bTask, string> = {
      classification: 'Résultat Classification B2B',
      'classification-risks': 'Analyse des Risques B2B',
      clustering: 'Résultat Clustering B2B',
      forecast: 'Prévision B2B',
      regression: 'Résultat Régression B2B',
      anomaly: 'Détection d\'Anomalies B2B',
    };
    return titles[this.selectedTask()];
  }

  /**
   * Format number as percentage
   */
  formatPct(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }

  /**
   * Check if any result is available
   */
  hasResult(): boolean {
    return !!(
      this.classifResult() ||
      this.clusterResult() ||
      this.forecastResult() ||
      this.regressResult() ||
      this.anomalyResult()
    );
  }

  /**
   * Main submit function - routes to appropriate API call
   */
  submit(): void {
    // Validation
    if (this.features.length === 0) {
      this.errorMsg.set('❌ Veuillez entrer au moins une feature');
      return;
    }

    // Additional validation for risk classification (27 features)
    if (this.selectedTask() === 'classification-risks' && this.features.length !== 27) {
      this.errorMsg.set(
        `❌ La classification des risques nécessite exactement 27 features (vous en avez ${this.features.length})`
      );
      return;
    }

    this.loading.set(true);
    this.errorMsg.set(null);
    this.clearResults();

    const task = this.selectedTask();

    switch (task) {
      case 'classification':
        this.submitClassification();
        break;
      case 'classification-risks':
        this.submitRisks();
        break;
      case 'clustering':
        this.submitClustering();
        break;
      case 'forecast':
        this.submitForecast();
        break;
      case 'regression':
        this.submitRegression();
        break;
      case 'anomaly':
        this.submitAnomaly();
        break;
    }
  }

  /**
   * Classification B2B
   */
  private submitClassification(): void {
    const payload: B2bFeaturesRequest = { features: this.features };

    this.api.b2bClassify(payload).subscribe({
      next: (res: B2bClassificationResponse) => {
        this.classifResult.set(res);
        this.loading.set(false);
      },
      error: (err: Error) => {
        this.handleError(err);
      },
    });
  }

  /**
   * Classification Risks B2B (27 features)
   */
  private submitRisks(): void {
    const payload: B2bFeaturesRequest = { features: this.features };

    this.api.b2bClassifyRisks(payload).subscribe({
      next: (res: B2bClassificationResponse) => {
        this.classifResult.set(res);
        this.loading.set(false);
      },
      error: (err: Error) => {
        this.handleError(err);
      },
    });
  }

  /**
   * Clustering B2B
   */
  private submitClustering(): void {
    const payload: B2bFeaturesRequest = { features: this.features };

    this.api.b2bCluster(payload).subscribe({
      next: (res: B2bClusteringResponse) => {
        this.clusterResult.set(res);
        this.loading.set(false);
      },
      error: (err: Error) => {
        this.handleError(err);
      },
    });
  }

  /**
   * Forecast B2B
   */
  private submitForecast(): void {
    const payload: B2bFeaturesRequest = { features: this.features };

    this.api.b2bForecast(payload).subscribe({
      next: (res: B2bRegressionResponse) => {
        this.forecastResult.set(res);
        this.loading.set(false);
      },
      error: (err: Error) => {
        this.handleError(err);
      },
    });
  }

  /**
   * Regression B2B
   */
  private submitRegression(): void {
    const payload: B2bFeaturesRequest = { features: this.features };

    this.api.b2bRegress(payload).subscribe({
      next: (res: B2bRegressionResponse) => {
        this.regressResult.set(res);
        this.loading.set(false);
      },
      error: (err: Error) => {
        this.handleError(err);
      },
    });
  }

  /**
   * Anomaly Detection B2B
   */
  private submitAnomaly(): void {
    const payload: B2bFeaturesRequest = { features: this.features };

    this.api.b2bAnomaly(payload).subscribe({
      next: (res: B2bAnomalyResponse) => {
        this.anomalyResult.set(res);
        this.loading.set(false);
      },
      error: (err: Error) => {
        this.handleError(err);
      },
    });
  }

  /**
   * Handle API errors
   */
  private handleError(err: Error): void {
    let message = err.message;

    // Extract meaningful error messages
    if (message.includes('400')) {
      message = '❌ Erreur de validation des données';
    } else if (message.includes('503')) {
      message = '⚠️ Le modèle n\'est pas chargé';
    } else if (message.includes('500')) {
      message = '❌ Erreur serveur';
    } else if (message.includes('Network')) {
      message = '❌ Impossible de se connecter à l\'API';
    }

    this.errorMsg.set(message);
    this.loading.set(false);
  }
}