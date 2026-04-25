import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import {
  GmClassificationRequest,
  GmRegressionRequest,
  GmClusteringRequest,
  GmAnomalyRequest,
} from '../../models/api.models';

export type GmTab = 'classification' | 'regression' | 'clustering' | 'anomaly';

@Component({
  selector: 'app-gm',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './gm.component.html',
  styleUrls: ['./gm.component.scss'],
})
export class GmComponent {
  activeTab: GmTab = 'classification';
  loading = false;
  error = '';
  result: any = null;

  // ── Classification (7 features) ──────────────────────────────
  classifForm: GmClassificationRequest = {
    Recency: 12,
    Customer_Age_Days: 365,
    Pct_Weekend: 0.25,
    Frequency: 5,
    Nb_Categories: 3,
    Monetary: 1250.75,
    Avg_Price: 250.15,
  };

  // ── Régression (25 features) ─────────────────────────────────
  regForm: GmRegressionRequest = {
    est_weekend: 0, mois: 6, trimestre: 2, quantite: 10,
    categorie_id_2: 1, categorie_id_3: 0, categorie_id_4: 0,
    categorie_id_5: 0, categorie_id_6: 0, categorie_id_7: 0,
    categorie_id_8: 0, categorie_id_9: 0, categorie_id_10: 0,
    categorie_id_11: 0, categorie_id_12: 0,
    canal_id_3: 1, canal_id_4: 0,
    gouvernorat_Ben_Arous: 1, gouvernorat_Bizerte: 0,
    gouvernorat_INCONNU: 0, gouvernorat_Monastir: 0,
    gouvernorat_Nabeul: 0, gouvernorat_Sfax: 0,
    gouvernorat_Sousse: 0, gouvernorat_Tunis: 0,
  };

  // ── Clustering (7 features) ──────────────────────────────────
  clusterForm: GmClusteringRequest = {
    Recency: 8,
    Frequency: 12,
    Monetary: 2450.0,
    Avg_Order_Value: 204.17,
    Nb_Categories: 4,
    Pct_Weekend: 0.33,
    Is_Online_Buyer: 1,
  };

  // ── Anomalie (5 features) ────────────────────────────────────
  anomalyForm: GmAnomalyRequest = {
    prix_unitaire: 49.99,
    quantite: 2,
    montant_total: 99.98,
    mois: 12,
    est_weekend: 1,
  };

  // one-hot helpers for gouvernorat & categorie selection
  gouvernorats = ['Ben_Arous','Bizerte','INCONNU','Monastir','Nabeul','Sfax','Sousse','Tunis'];
  categories   = [2,3,4,5,6,7,8,9,10,11,12];
  canaux       = [3,4];

  selectedGouv    = 'Ben_Arous';
  selectedCat     = 2;
  selectedCanal   = 3;

  constructor(private api: ApiService) {}

  setTab(tab: GmTab): void {
    this.activeTab = tab;
    this.result = null;
    this.error = '';
  }

  /** Sync one-hot fields from dropdowns before submit */
  syncRegForm(): void {
    // Reset all gouvernorats
    this.gouvernorats.forEach(g => {
      (this.regForm as any)[`gouvernorat_${g}`] = this.selectedGouv === g ? 1 : 0;
    });
    // Reset all categories
    this.categories.forEach(c => {
      (this.regForm as any)[`categorie_id_${c}`] = this.selectedCat === c ? 1 : 0;
    });
    // Reset canaux
    this.canaux.forEach(c => {
      (this.regForm as any)[`canal_id_${c}`] = this.selectedCanal === c ? 1 : 0;
    });
  }

  submit(): void {
    if (this.activeTab === 'regression') this.syncRegForm();
    this.loading = true;
    this.error = '';
    this.result = null;
    let obs$: any;
    switch (this.activeTab) {
      case 'classification': obs$ = this.api.gmClassify(this.classifForm);  break;
      case 'regression':     obs$ = this.api.gmRegress(this.regForm);       break;
      case 'clustering':     obs$ = this.api.gmCluster(this.clusterForm);   break;
      case 'anomaly':        obs$ = this.api.gmAnomaly(this.anomalyForm);   break;
    }
    obs$.subscribe({
      next: (res: any) => { this.result = res; this.loading = false; },
      error: (err: Error) => { this.error = err.message; this.loading = false; },
    });
  }
}