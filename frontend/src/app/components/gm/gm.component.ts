import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import {
  GmClassificationRequest, GmClassificationResponse,
  GmRegressionRequest,     GmRegressionResponse,
  GmClusteringRequest,     GmClusteringResponse,
  GmAnomalyRequest,        AnomalyResponse,
} from '../../models/api.models';

export type GmTab = 'classification' | 'regression' | 'clustering' | 'anomaly';

const PROFIL_CLIENT: Record<number, { label: string; desc: string; action: string; icon: string; color: string }> = {
  0: {
    label: 'Client à risque de départ',
    desc: 'Ce client n\'a pas commandé depuis longtemps et ses achats se raréfient.',
    action: 'Recommandé : offre de réactivation ou bon de fidélité personnalisé.',
    icon: '⚠️', color: 'result-reveal--amber',
  },
  1: {
    label: 'Client régulier',
    desc: 'Ce client commande régulièrement avec un panier dans la moyenne de Sougui.',
    action: 'Maintenir la relation et lui proposer les nouveautés du catalogue artisanal.',
    icon: '🤝', color: 'result-reveal--blue',
  },
  2: {
    label: 'Client fidèle — Prioritaire',
    desc: 'Excellent profil : commandes fréquentes, montants élevés, très engagé envers Sougui.',
    action: 'Traitement prioritaire — accès aux pièces exclusives et programme partenaire.',
    icon: '⭐', color: 'result-reveal--green',
  },
};

const SEGMENT_CLIENT: Record<number, { label: string; desc: string; action: string; icon: string; color: string }> = {
  0: {
    label: 'Acheteur occasionnel',
    desc: 'Achats ponctuels, panier modeste. Client en phase de découverte.',
    action: 'Proposer une sélection d\'entrée de gamme et un guide artisanat.',
    icon: '🛒', color: 'result-reveal--blue',
  },
  1: {
    label: 'Acheteur fidèle',
    desc: 'Commandes régulières couvrant plusieurs catégories artisanales Sougui.',
    action: 'Invitation aux événements et campagne multi-catégories.',
    icon: '💙', color: 'result-reveal--green',
  },
  2: {
    label: 'Grand compte',
    desc: 'Volumes et montants élevés. Client stratégique de la maison Sougui.',
    action: 'Suivi personnalisé, offres exclusives et partenariat artisan dédié.',
    icon: '🏆', color: 'result-reveal--amber',
  },
};

const CAT_NAMES: Record<number, string> = {
  2: 'Bijou artisanal', 3: 'Couffin tressé', 4: 'Poterie fine',
  5: 'Tableau décoratif', 6: 'Couffin tissé', 7: 'Couffin palmier',
  8: 'Poterie décorée', 9: 'Textile artisanal', 10: 'Tableau sur bois',
  11: 'Poterie émaillée', 12: 'Calligraphie',
};

const CAT_IMPACT: Record<number, number> = {
  10: 95, 2: 72, 11: 54, 7: 52, 12: 38, 8: 34, 9: 20, 3: 15, 4: 12, 5: 10, 6: 8,
};

@Component({
  selector: 'app-gm',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './gm.component.html',
  styleUrls: ['./gm.component.scss'],
})
export class GmComponent {
  activeTab = signal<GmTab>('classification');
  loading   = signal(false);
  errorMsg  = signal<string | null>(null);

  classifResult  = signal<GmClassificationResponse | null>(null);
  regrResult     = signal<GmRegressionResponse | null>(null);
  clusterResult  = signal<GmClusteringResponse | null>(null);
  anomalyResult  = signal<AnomalyResponse | null>(null);

  readonly catNames   = CAT_NAMES;
  readonly catImpact  = CAT_IMPACT;
  readonly catKeys    = Object.keys(CAT_NAMES).map(Number);

  classifForm: GmClassificationRequest = {
    Recency: 12, Customer_Age_Days: 365, Pct_Weekend: 0.25,
    Frequency: 5, Nb_Categories: 3, Monetary: 1250.75, Avg_Price: 250.15,
  };

  regForm: GmRegressionRequest = {
    est_weekend: 0, mois: 6, trimestre: 2, quantite: 10,
    categorie_id_2: 1, categorie_id_3: 0, categorie_id_4: 0, categorie_id_5: 0,
    categorie_id_6: 0, categorie_id_7: 0, categorie_id_8: 0, categorie_id_9: 0,
    categorie_id_10: 0, categorie_id_11: 0, categorie_id_12: 0,
    canal_id_3: 1, canal_id_4: 0,
    gouvernorat_Ben_Arous: 1, gouvernorat_Bizerte: 0, gouvernorat_INCONNU: 0,
    gouvernorat_Monastir: 0, gouvernorat_Nabeul: 0, gouvernorat_Sfax: 0,
    gouvernorat_Sousse: 0, gouvernorat_Tunis: 0,
  };

  clusterForm: GmClusteringRequest = {
    Recency: 8, Frequency: 12, Monetary: 2450.0,
    Avg_Order_Value: 204.17, Nb_Categories: 4, Pct_Weekend: 0.33, Is_Online_Buyer: 1,
  };

  anomalyForm: GmAnomalyRequest = {
    prix_unitaire: 49.99, quantite: 2, montant_total: 99.98, mois: 12, est_weekend: 1,
  };

  readonly gouvernorats = ['Ben_Arous','Bizerte','INCONNU','Monastir','Nabeul','Sfax','Sousse','Tunis'];
  readonly gouvernoratLabels: Record<string, string> = {
    Ben_Arous: 'Ben Arous', Bizerte: 'Bizerte', INCONNU: 'Non renseigné',
    Monastir: 'Monastir', Nabeul: 'Nabeul', Sfax: 'Sfax', Sousse: 'Sousse', Tunis: 'Tunis',
  };
  readonly canaux = [3, 4];
  readonly canalLabels: Record<number, string> = { 3: 'Boutique en ligne', 4: 'Point de vente' };
  readonly moisLabels = ['Janvier','Février','Mars','Avril','Mai','Juin',
                         'Juillet','Août','Septembre','Octobre','Novembre','Décembre'];

  selectedGouv  = 'Ben_Arous';
  selectedCat   = 2;
  selectedCanal = 3;

  constructor(private api: ApiService) {}

  setTab(tab: GmTab): void {
    this.activeTab.set(tab);
    this.errorMsg.set(null);
    this.classifResult.set(null);
    this.regrResult.set(null);
    this.clusterResult.set(null);
    this.anomalyResult.set(null);
  }

  syncRegForm(): void {
    this.gouvernorats.forEach(g =>
      ((this.regForm as any)[`gouvernorat_${g}`] = this.selectedGouv === g ? 1 : 0));
    this.catKeys.forEach(c =>
      ((this.regForm as any)[`categorie_id_${c}`] = this.selectedCat === c ? 1 : 0));
    this.canaux.forEach(c =>
      ((this.regForm as any)[`canal_id_${c}`] = this.selectedCanal === c ? 1 : 0));
  }

  syncMontant(): void {
    this.anomalyForm.montant_total =
      Math.round(this.anomalyForm.prix_unitaire * this.anomalyForm.quantite * 100) / 100;
  }

  submit(): void {
    if (this.activeTab() === 'regression') this.syncRegForm();
    this.loading.set(true);
    this.errorMsg.set(null);
    const done = (err?: Error) => { if (err) this.errorMsg.set(err.message); this.loading.set(false); };

    switch (this.activeTab()) {
      case 'classification': this.api.gmClassify(this.classifForm).subscribe({ next: r => { this.classifResult.set(r); done(); }, error: done }); break;
      case 'regression':     this.api.gmRegress(this.regForm).subscribe({ next: r => { this.regrResult.set(r); done(); }, error: done }); break;
      case 'clustering':     this.api.gmCluster(this.clusterForm).subscribe({ next: r => { this.clusterResult.set(r); done(); }, error: done }); break;
      case 'anomaly':        this.api.gmAnomaly(this.anomalyForm).subscribe({ next: r => { this.anomalyResult.set(r); done(); }, error: done }); break;
    }
  }

  getProfil(p: number)  { return PROFIL_CLIENT[p]  ?? { label:`Profil ${p}`, desc:'', action:'', icon:'❓', color:'result-reveal--blue' }; }
  getSegment(c: number) { return SEGMENT_CLIENT[c]  ?? { label:`Segment ${c}`, desc:'', action:'', icon:'📦', color:'result-reveal--blue' }; }
  scoreBar(s: number): number { return Math.min(100, Math.max(0, ((s + 0.5) / 0.7) * 100)); }
}