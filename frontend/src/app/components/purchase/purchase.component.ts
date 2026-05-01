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

// ─── Labels métier basés sur l'analyse du modèle ──────────────────
// Le RandomForest est piloté à 96% par Taux_TVA (52%) + Marge_TVA (44%)
// Classe 0 = profil TVA faible → achat exonéré ou à régime réduit
// Classe 1 = profil TVA standard → achat courant validé
const CLASSIF_LABELS: Record<number, { label: string; desc: string; color: string }> = {
  0: {
    label: 'Achat exonéré / TVA réduite',
    desc: 'Cet achat présente un profil TVA faible. Il est classé hors TVA standard ou à régime réduit.',
    color: 'result-box--amber',
  },
  1: {
    label: 'Achat soumis TVA standard',
    desc: 'Cet achat suit le régime TVA courant. Il peut être validé normalement.',
    color: 'result-box--green',
  },
};

// ─── Segments de clustering fournisseurs ──────────────────────────
// KMeans sur : Nb_Factures, Montant_Total, Montant_Moyen, Montant_Max, TVA_Moy
const CLUSTER_LABELS: Record<number, { label: string; desc: string; icon: string }> = {
  0: {
    label: 'Petit fournisseur',
    desc: 'Volume faible, faible nombre de factures. Relation occasionnelle.',
    icon: '🏪',
  },
  1: {
    label: 'Fournisseur régulier',
    desc: 'Volume et fréquence modérés. Partenaire de confiance établi.',
    icon: '🤝',
  },
  2: {
    label: 'Fournisseur stratégique',
    desc: 'Grand volume, montants élevés. Partenaire clé à fidéliser.',
    icon: '⭐',
  },
};

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

  classifResult = signal<PurchaseClassificationResponse | null>(null);
  regrResult    = signal<PurchaseRegressionResponse | null>(null);
  clustResult   = signal<PurchaseClusteringResponse | null>(null);

  // ─── Fournisseurs connus (extraits de l'encodeur) ─────────────
  readonly fournisseurs = [
    'ASCOM','STEG','SONEDE','ORANGE','Tunisie Telecom',
    'KERAMOS','HAMILA CERAMIQUE','ENNAIM Céramique','Zouba Ceramic',
    'Zitoun Artisanat','Yassin Calligraphie','Fabripierre',
    'DJELASSI','KAMTRADE','KACO SA','PROSCOM','SMPA',
    'Quincaillerie Générale','Quincaillerie Zormani',
    'La Quincaillerie du Sahel','Delta Distribution',
    'META (FACEBOOK)','Elementor Ltd.','Oxahost',
    'Vivo Energy','NATBAG','TRANSAF','Transport Fehri',
    'My Print Tunisia','SNAPRINT','My Print Tunisia',
    'INCONNU'
  ];
  readonly categories = ['MATERIAUX', 'SERVICE'];

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
      next: (res) => { this.classifResult.set(res); this.loading.set(false); },
      error: (err: Error) => { this.errorMsg.set(err.message); this.loading.set(false); },
    });
  }

  submitRegression(): void {
    this.loading.set(true);
    this.errorMsg.set(null);
    this.regrResult.set(null);
    this.api.purchaseRegress(this.regrForm).subscribe({
      next: (res) => { this.regrResult.set(res); this.loading.set(false); },
      error: (err: Error) => { this.errorMsg.set(err.message); this.loading.set(false); },
    });
  }

  submitClustering(): void {
    this.loading.set(true);
    this.errorMsg.set(null);
    this.clustResult.set(null);
    this.api.purchaseCluster(this.clustForm).subscribe({
      next: (res) => { this.clustResult.set(res); this.loading.set(false); },
      error: (err: Error) => { this.errorMsg.set(err.message); this.loading.set(false); },
    });
  }

  probPercent(val: number): string {
    return (val * 100).toFixed(1) + '%';
  }

  getClassifMeta(pred: number) {
    return CLASSIF_LABELS[pred] ?? {
      label: `Classe ${pred}`,
      desc: 'Classe non documentée.',
      color: 'result-box--blue',
    };
  }

  getClusterMeta(cluster: number) {
    return CLUSTER_LABELS[cluster] ?? {
      label: `Segment ${cluster}`,
      desc: 'Segment non documenté.',
      icon: '📦',
    };
  }
}