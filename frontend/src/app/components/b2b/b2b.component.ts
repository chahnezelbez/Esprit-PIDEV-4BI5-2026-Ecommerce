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

type B2bTask = 'classification' | 'classification-risks' | 'clustering' | 'forecast' | 'regression' | 'anomaly';

@Component({
  selector: 'app-b2b',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './b2b.component.html',
  styleUrls: ['./b2b.component.scss'],
})
export class B2bComponent {
  loading = signal(false);
  errorMsg = signal<string | null>(null);
  selectedTask = signal<B2bTask>('classification');
  
  classifResult = signal<B2bClassificationResponse | null>(null);
  clusterResult = signal<B2bClusteringResponse | null>(null);
  forecastResult = signal<B2bForecastResponse | null>(null);
  regressResult = signal<B2bRegressionResponse | null>(null);
  anomalyResult = signal<B2bAnomalyResponse | null>(null);
  
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

  constructor(private api: ApiService) {}

  setTask(task: B2bTask): void {
    this.selectedTask.set(task);
    this.clearResults();
    this.errorMsg.set(null);
  }

  private clearResults(): void {
    this.classifResult.set(null);
    this.clusterResult.set(null);
    this.forecastResult.set(null);
    this.regressResult.set(null);
    this.anomalyResult.set(null);
  }

  get features(): number[] {
    return this.featuresValue
      .split(',')
      .map((v) => Number(v.trim()))
      .filter((v) => !isNaN(v));
  }

  getTaskLabel(task: B2bTask): string {
    const labels: Record<B2bTask, string> = {
      classification: 'Classification',
      'classification-risks': 'Risques',
      clustering: 'Clustering',
      forecast: 'Prévision',
      regression: 'Régression',
      anomaly: 'Anomalie',
    };
    return labels[task];
  }

  getTaskIcon(task: B2bTask): string {
    const icons: Record<B2bTask, string> = {
      classification: '📊',
      'classification-risks': '⚠️',
      clustering: '🎯',
      forecast: '📈',
      regression: '📉',
      anomaly: '🔍',
    };
    return icons[task];
  }

  hasResult(): boolean {
    return !!(
      this.classifResult() ||
      this.clusterResult() ||
      this.forecastResult() ||
      this.regressResult() ||
      this.anomalyResult()
    );
  }

  submit(): void {
    if (this.features.length === 0) {
      this.errorMsg.set('Veuillez entrer au moins une feature');
      return;
    }
    if (this.selectedTask() === 'classification-risks' && this.features.length !== 27) {
      this.errorMsg.set(`La classification des risques nécessite 27 features (${this.features.length} fournies)`);
      return;
    }

    this.loading.set(true);
    this.errorMsg.set(null);
    this.clearResults();

    const task = this.selectedTask();
    const payload: B2bFeaturesRequest = { features: this.features };

    const handleError = (err: Error) => {
      let msg = err.message;
      if (msg.includes('400')) msg = 'Erreur de validation des données';
      else if (msg.includes('503')) msg = 'Le modèle n\'est pas chargé';
      else if (msg.includes('500')) msg = 'Erreur serveur';
      else if (msg.includes('Network')) msg = 'Impossible de se connecter à l\'API';
      this.errorMsg.set(msg);
      this.loading.set(false);
    };

    switch (task) {
      case 'classification':
        this.api.b2bClassify(payload).subscribe({
          next: (res) => { this.classifResult.set(res); this.loading.set(false); },
          error: handleError,
        });
        break;
      case 'classification-risks':
        this.api.b2bClassifyRisks(payload).subscribe({
          next: (res) => { this.classifResult.set(res); this.loading.set(false); },
          error: handleError,
        });
        break;
      case 'clustering':
        this.api.b2bCluster(payload).subscribe({
          next: (res) => { this.clusterResult.set(res); this.loading.set(false); },
          error: handleError,
        });
        break;
      case 'forecast':
        // Attention: forecast utilise B2bFeaturesRequest? Vérifiez votre API.
        // Si l'API attend { periods }, adapter. Ici on utilise les features.
        this.api.b2bForecast(payload).subscribe({
          next: (res) => { this.forecastResult.set(res); this.loading.set(false); },
          error: handleError,
        });
        break;
      case 'regression':
        this.api.b2bRegress(payload).subscribe({
          next: (res) => { this.regressResult.set(res); this.loading.set(false); },
          error: handleError,
        });
        break;
      case 'anomaly':
        this.api.b2bAnomaly(payload).subscribe({
          next: (res) => { this.anomalyResult.set(res); this.loading.set(false); },
          error: handleError,
        });
        break;
    }
  }
}