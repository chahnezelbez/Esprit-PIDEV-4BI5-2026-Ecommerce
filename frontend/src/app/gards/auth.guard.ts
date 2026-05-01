import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { KeycloakService } from 'keycloak-angular';

export const authGuard: CanActivateFn = async () => {
  const keycloak = inject(KeycloakService);
  const router = inject(Router);

  const loggedIn = await keycloak.isLoggedIn();
  if (loggedIn) {
    return true;
  } else {
    // Redirige automatiquement vers l'écran de connexion Keycloak
    await keycloak.login();
    return false;
  }
};