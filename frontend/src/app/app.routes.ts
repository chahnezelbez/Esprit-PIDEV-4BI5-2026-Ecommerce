import { Routes } from '@angular/router';
import { roleGuard } from './gards/role.guard';

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
    canActivate: [roleGuard(['achat'])],
  },
  {
    path: 'commercial',
    loadComponent: () =>
      import('./components/commercial/commercial.component').then(
        (m) => m.CommercialComponent
      ),
    canActivate: [roleGuard(['vente_b2c'])],
  },
  {
    path: 'marketing',
    loadComponent: () =>
      import('./components/marketing/marketing.component').then(
        (m) => m.MarketingComponent
      ),
    canActivate: [roleGuard(['marketing'])],
  },
  {
    path: 'gm',
    loadComponent: () =>
      import('./components/gm/gm.component').then((m) => m.GmComponent),
    canActivate: [roleGuard(['general_manager'])],
  },
  {
    path: 'b2b',
    loadComponent: () =>
      import('./components/b2b/b2b.component').then((m) => m.B2bComponent),
    canActivate: [roleGuard(['vente_b2b'])],
  },
  {
    path: 'financier',
    loadComponent: () =>
      import('./components/financier/financier.component').then(
        (m) => m.FinancierComponent
      ),
    canActivate: [roleGuard(['financier'])],
  },
  // ❌ plus de route unauthorized
];