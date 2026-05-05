import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent {

  overviewUrl: SafeResourceUrl;

  readonly features = [
    {
      icon: 'bar',
      title: 'Power BI Embarqué',
      desc: 'Rapports interactifs par rôle — Finance, Achats, B2B, B2C — avec accès sécurisé et filtres dynamiques.'
    },
    {
      icon: 'wave',
      title: 'Prédictions ML',
      desc: 'Modèles de machine learning pour anticiper les tendances de vente, les risques et les opportunités.'
    },
    {
      icon: 'kpi',
      title: 'KPIs en temps réel',
      desc: 'Tableau de bord live avec métriques clés, alertes automatiques et comparaisons période par période.'
    }
  ];

  readonly roles = [
    { label: 'Direction générale', route: '/gm' },
    { label: 'Finance',            route: '/financier' },
    { label: 'Achats',             route: '/purchase' },
    { label: 'Ventes B2B',         route: '/b2b' },
    { label: 'Ventes B2C',         route: '/commercial' },
    { label: 'Marketing',          route: '/marketing' }
  ];

  constructor(private sanitizer: DomSanitizer) {
    const raw = 'https://app.powerbi.com/reportEmbed'
      + '?reportId=84068065-f618-44eb-b89f-4676655167c8'
      + '&pageName=5b093f9e77da4c3adfe5'
      + '&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730'
      + '&autoAuth=true'
      + '&filterPaneEnabled=false'
      + '&navContentPaneEnabled=false';

    this.overviewUrl = this.sanitizer.bypassSecurityTrustResourceUrl(raw);
  }
  scrollToFeatures() {
  document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' });
}
}