import { Routes } from '@angular/router';
 
export const routes: Routes = [
  {
    path: '',
    redirectTo: 'purchase',
    pathMatch: 'full',
  },
  {
    path: 'purchase',
    loadComponent: () =>
      import('./components/purchase/purchase.component').then(
        (m) => m.PurchaseComponent
      ),
  },
  {
    path: 'commercial',
    loadComponent: () =>
      import('./components/commercial/commercial.component').then(
        (m) => m.CommercialComponent
      ),
  },
  {
    path: 'marketing',
    loadComponent: () =>
      import('./components/marketing/marketing.component').then(
        (m) => m.MarketingComponent
      ),
  },
  {
    path: 'gm',
    loadComponent: () =>
      import('./components/gm/gm.component').then((m) => m.GmComponent),
  },
  {
    path: 'b2b',
    loadComponent: () =>
      import('./components/b2b/b2b.component').then((m) => m.B2bComponent),
  },
  {
    path: 'financier',
    loadComponent: () =>
      import('./components/financier/financier.component').then(
        (m) => m.FinancierComponent
      ),
  },
];