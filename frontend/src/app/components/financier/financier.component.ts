import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import {
  FinClusteringRequest,
  FinClusteringResponse,
  FinForecastResponse,
} from '../../models/api.models';
import { ResultCardComponent } from '../../shared/result-card/result-card.component';

type FinanceTask = 'clustering' | 'forecast';

/**
 * Segment metadata for customer classification
 */
const SEGMENT_META: Record<number, {
  label: string;
  icon: string;
  description: string;
  recommendation: string;
  color: string;
}> = {
  0: {
    label: 'Clients Réguliers',
    icon: '👤',
    description: 'Clients stables avec activité modérée. Bonne rétention et engagement faible à modéré.',
    recommendation: 'Proposez des offres fidélité pour augmenter la fréquence d\'achat.',
    color: 'segment-0',
  },
  1: {
    label: 'Clients VIP',
    icon: '👑',
    description: 'Meilleurs clients avec activité élevée et paniers importants. Très engagés.',
    recommendation: 'Proposez un programme premium exclusif et un service personnalisé.',
    color: 'segment-1',
  },
  2: {
    label: 'Clients Occasionnels',
    icon: '🎯',
    description: 'Achètent peu fréquemment. Montants variables, engagement faible.',
    recommendation: 'Relancez-les avec des promotions ciblées et offres spéciales.',
    color: 'segment-2',
  },
  3: {
    label: 'Clients à Risque',
    icon: '⚠️',
    description: 'Activité faible et rétention difficile. Peu d\'engagement récent.',
    recommendation: 'Lancez une campagne de réactivation avec offres attrayantes.',
    color: 'segment-3',
  },
  4: {
    label: 'Clients Croissance',
    icon: '📈',
    description: 'Clients en développement avec potentiel d\'augmentation. Activité en hausse.',
    recommendation: 'Investissez dans la relation pour transformer en VIP.',
    color: 'segment-4',
  },
};

@Component({
  selector: 'app-financier',
  standalone: true,
  imports: [CommonModule, FormsModule, ResultCardComponent],
  templateUrl: './financier.component.html',
  styleUrls: ['./financier.component.scss'],
})
export class FinancierComponent {
  // Signals
  loading = signal(false);
  errorMsg = signal<string | null>(null);
  selectedTask = signal<FinanceTask>('clustering');

  // Result signals
  clusteringResult = signal<FinClusteringResponse | null>(null);
  forecastResult = signal<FinForecastResponse | null>(null);

  // Form Values - Clustering (Customer Profile)
  clusteringForm: FinClusteringRequest = {
    nb_commandes: 25,
    ca_total: 5000,
    panier_moyen: 200,
    total_remise: 500,
    total_remboursement: 200,
    livraison_moyenne: 7.5,
    total_articles: 150,
    nb_categories: 8,
    nb_produits: 45,
    taux_finalisation: 85,
    recence: 30,
  };

  // Form Values - Forecast
  forecastPeriods = 12;

  /**
   * Constructor - Inject ApiService
   */
  constructor(private api: ApiService) {}

  /**
   * Switch between tasks
   */
  setTask(task: FinanceTask): void {
    this.selectedTask.set(task);
    this.errorMsg.set(null);
  }

  /**
   * Get segment label from cluster number
   */
  getSegmentLabel(cluster: number): string {
    return SEGMENT_META[cluster]?.label ?? `Segment ${cluster}`;
  }

  /**
   * Get segment icon from cluster number
   */
  getSegmentIcon(cluster: number): string {
    return SEGMENT_META[cluster]?.icon ?? '📦';
  }

  /**
   * Get segment description
   */
  getSegmentDescription(cluster: number): string {
    return SEGMENT_META[cluster]?.description ?? 'Segment non identifié';
  }

  /**
   * Get segment recommendation
   */
  getSegmentRecommendation(cluster: number): string {
    return SEGMENT_META[cluster]?.recommendation ?? 'Analysez ce segment davantage';
  }

  /**
   * Get CSS class for segment styling
   */
  getSegmentClass(cluster: number): string {
    return SEGMENT_META[cluster]?.color ?? 'segment-0';
  }

  /**
   * Calculate average forecast value
   */
  getForecastAvg(forecast: number[]): number {
    if (forecast.length === 0) return 0;
    return forecast.reduce((a, b) => a + b, 0) / forecast.length;
  }

  /**
   * Get maximum forecast value
   */
  getForecastMax(forecast: number[]): number {
    return Math.max(...forecast);
  }

  /**
   * Get minimum forecast value
   */
  getForecastMin(forecast: number[]): number {
    return Math.min(...forecast);
  }

  /**
   * Submit clustering request
   */
  submitClustering(): void {
    this.loading.set(true);
    this.errorMsg.set(null);
    this.clusteringResult.set(null);

    this.api.finCluster(this.clusteringForm).subscribe({
      next: (res: FinClusteringResponse) => {
        this.clusteringResult.set(res);
        this.loading.set(false);
      },
      error: (err: Error) => {
        this.handleError(err);
      },
    });
  }

  /**
   * Submit forecast request
   */
  submitForecast(): void {
    this.loading.set(true);
    this.errorMsg.set(null);
    this.forecastResult.set(null);

    const payload = { periods: this.forecastPeriods };

    this.api.finForecast(payload).subscribe({
      next: (res: FinForecastResponse) => {
        this.forecastResult.set(res);
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
      message = '❌ Données invalides. Vérifiez les champs requis.';
    } else if (message.includes('503')) {
      message = '⚠️ Le modèle financier n\'est pas disponible.';
    } else if (message.includes('500')) {
      message = '❌ Erreur serveur. Veuillez réessayer.';
    } else if (message.includes('Network')) {
      message = '❌ Impossible de joindre le serveur. Vérifiez votre connexion.';
    }

    this.errorMsg.set(message);
    this.loading.set(false);
  }
}