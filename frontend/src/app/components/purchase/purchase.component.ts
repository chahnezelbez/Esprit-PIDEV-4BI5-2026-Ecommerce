import { Component, Inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import {
  PurchaseClassificationRequest,
  PurchaseRegressionRequest,
  PurchaseClusteringRequest,
  PurchaseClassificationResponse,
  PurchaseRegressionResponse,
  PurchaseClusteringResponse,
} from '../../models/api.models';

type ActiveTask = 'classification' | 'regression' | 'clustering';

@Component({
  selector: 'app-purchase',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './purchase.component.html',
  styleUrls: ['./purchase.component.scss'],
})
export class PurchaseComponent {
  activeTask = signal<ActiveTask>('classification');
  loading = signal(false);
  errorMsg = signal<string | null>(null);

  // ─── Résultats ───────────────────────────────────────────────────
  classifResult = signal<PurchaseClassificationResponse | null>(null);
  regrResult    = signal<PurchaseRegressionResponse | null>(null);
  clustResult   = signal<PurchaseClusteringResponse | null>(null);

  // ─── Valeurs des formulaires ─────────────────────────────────────

  classifForm: PurchaseClassificationRequest = {
    Montant_HT: 1500,
    Taux_TVA: 0.18,
    Marge_TVA: 0.05,
    Mois: 5,
    Annee: 2025,
    Semaine: 20,
    Est_weekend: 0,
    fournisseur: 'ASCOM',
    categorie: 'MATERIAUX',
  };

  regrForm: PurchaseRegressionRequest = {
    Mois: 6,
    Annee: 2025,
    Semaine: 24,
    Est_weekend: 1,
    fournisseur: 'STEG',
    categorie: 'SERVICE',
    methode: 'Inconnu',
    Taux_TVA: 0.19,
  };

  clustForm: PurchaseClusteringRequest = {
    Nb_Factures: 12,
    Montant_Total: 12500,
    Montant_Moyen: 1041.67,
    Montant_Max: 3500,
    TVA_Moy: 0.18,
  };

  constructor(@Inject(ApiService) private api: ApiService) {}

  setTask(task: ActiveTask): void {
    this.activeTask.set(task);
    this.errorMsg.set(null);
  }

  submitClassification(): void {
    this.loading.set(true);
    this.errorMsg.set(null);
    this.classifResult.set(null);

    this.api.purchaseClassify(this.classifForm).subscribe({
      next: (res) => {
        this.classifResult.set(res);
        this.loading.set(false);
      },
      error: (err: Error) => {
        this.errorMsg.set(err.message);
        this.loading.set(false);
      },
    });
  }

  submitRegression(): void {
    this.loading.set(true);
    this.errorMsg.set(null);
    this.regrResult.set(null);

    this.api.purchaseRegress(this.regrForm).subscribe({
      next: (res) => {
        this.regrResult.set(res);
        this.loading.set(false);
      },
      error: (err: Error) => {
        this.errorMsg.set(err.message);
        this.loading.set(false);
      },
    });
  }

  submitClustering(): void {
    this.loading.set(true);
    this.errorMsg.set(null);
    this.clustResult.set(null);

    this.api.purchaseCluster(this.clustForm).subscribe({
      next: (res) => {
        this.clustResult.set(res);
        this.loading.set(false);
      },
      error: (err: Error) => {
        this.errorMsg.set(err.message);
        this.loading.set(false);
      },
    });
  }

  // Helper pour afficher le % de probabilité
  probPercent(val: number): string {
    return (val * 100).toFixed(1) + '%';
  }

  // Label lisible pour la classification
  classifLabel(pred: number): string {
    const labels: Record<number, string> = {
      0: 'Classe 0',
      1: 'Classe 1',
      2: 'Classe 2',
    };
    return labels[pred] ?? `Classe ${pred}`;
  }

  // Label lisible pour le cluster
  clusterLabel(cluster: number): string {
    const labels: Record<number, string> = {
      0: 'Petit fournisseur',
      1: 'Fournisseur moyen',
      2: 'Grand fournisseur',
    };
    return labels[cluster] ?? `Segment ${cluster}`;
  }
}